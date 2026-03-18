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

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError(
                "dataset must be a non-empty string"
            )

    @classmethod
    def from_json(
        cls, path: str = "config.json"
    ) -> "TaskConfig":
        """Load and validate configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


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

    def __init__(
        self, output_dir: str, config: TaskConfig
    ) -> None:
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

        print(f"Loading target survey from: {self.config.dataset}")
        target = loader.load_target(self.config.dataset)
        print(f"Target has {len(target)} variables")

        print(f"Loading candidate survey from: {self.config.dataset}")
        candidate = loader.load_candidate(self.config.dataset)
        print(f"Candidate has {len(candidate)} variables")

        return target, candidate

    def _match(
        self, target: Survey, candidate: Survey
    ) -> MatchResult:
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
                    "target_variable": m.target_variable,
                    "candidate_variable": m.candidate_variable,
                    "similarity_score": m.similarity_score,
                    "match_confidence": m.match_confidence,
                    "is_match": m.is_match,
                    "needs_recode": m.needs_recode,
                    "standardised_label": m.standardised_label,
                    "reasoning": m.reasoning,
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

        n_matched = len(result.matched)
        n_total = len(result.matches)
        print(f"Matched {n_matched}/{n_total} target variables")

        writer = SurveyWriter(root=str(run_dir))

        # Save recoded target
        try:
            path = writer.save_target(result.target)
            print(f"Wrote recoded target: {path}")
        except ValueError:
            print("Target survey has no data \u2014 skipping .sav write")

        # Save recoded candidate
        try:
            path = writer.save_candidate(result.candidate)
            print(f"Wrote recoded candidate: {path}")
        except ValueError:
            print(
                "Candidate survey has no data — skipping .sav write"
            )

        print(f"Done! Output saved to: {run_dir}")
