"""
Matching pipeline shared by local and remote entrypoints.

Loads two surveys from .sav files, matches target variables against
the candidate survey using embeddings and LLM-based recode design,
then saves the recoded results.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from survey_io import SurveyLoader, SurveyWriter

from variablematcher import MatchResult, Survey, VariableMatcher


@dataclass(frozen=True)
class TaskConfig:
    """Validated configuration for a single matching run."""

    dataset: str
    target_name: str
    candidate_name: str

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must be a non-empty string")


class Pipeline:
    """
    Pipeline for running the full matching process.

    Parameters
    ----------
    output_dir : str
        Root directory for all pipeline outputs.
    config : TaskConfig
        Validated run configuration.
    """

    def __init__(self, output_dir: str, config: TaskConfig) -> None:
        self.output_dir = output_dir
        self.config = config
        self._data_root = "/app/data/datasets"

    def run(self) -> None:
        """Execute the full matching pipeline."""
        target, candidate = self._load_surveys()
        result = self._match(target, candidate)
        self._save(result)

    def _load_surveys(self) -> tuple[Survey, Survey]:
        """Load the target and candidate surveys from .sav files."""
        loader = SurveyLoader(root=self._data_root)

        print(f"Loading target survey: {self.config.target_name}")
        target = loader.load(self.config.dataset, self.config.target_name)
        print(f"Target has {len(target.variables)} variables")

        print(f"Loading candidate survey: {self.config.candidate_name}")
        candidate = loader.load(
            self.config.dataset, self.config.candidate_name
        )
        print(f"Candidate has {len(candidate.variables)} variables")

        return target, candidate

    def _match(self, target: Survey, candidate: Survey) -> MatchResult:
        """Match target variables against the candidate survey."""
        print("Initialising matcher...")
        matcher = VariableMatcher()

        print("Running fit...")
        matcher.fit(target, candidate)

        print("Running predict...")
        return matcher.predict()

    def _save_matches_csv(
        self,
        result: MatchResult,
        run_dir: Path,
    ) -> pd.DataFrame:
        """Save match results summary to CSV."""
        rows = []
        for m in result.matches:
            rows.append(
                {
                    "target_variable": m.target.variable,
                    "candidate_variable": (
                        m.candidate.variable if m.candidate else None
                    ),
                    "is_match": m.is_match,
                    "target_groups": (
                        json.dumps(m.target.groups)
                        if m.target.groups
                        else None
                    ),
                    "candidate_groups": (
                        json.dumps(m.candidate.groups)
                        if m.candidate and m.candidate.groups
                        else None
                    ),
                }
            )

        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "matches.csv", index=False)
        return df

    def _save(self, result: MatchResult) -> None:
        """Save match results and recoded surveys."""
        print("Saving results...")
        run_dir = Path(self.output_dir)
        os.makedirs(run_dir, exist_ok=True)

        self._save_matches_csv(result, run_dir)

        writer = SurveyWriter(root=str(run_dir / "datasets"))
        for survey in (result.target, result.candidate):
            try:
                path = writer.save(survey)
                print(f"Wrote {survey.name}: {path}")
            except Exception as e:
                print(f"{survey.name} save failed: {e}")

        print(f"Done! Output saved to: {run_dir}")
