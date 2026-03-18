"""
Variable matching module.

Provides the VariableMatcher class with fit/predict API for matching
target survey variables against a candidate survey.
"""

import contextlib
import copy
import functools
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from google import genai
from google.api_core.exceptions import Conflict
from google.cloud import bigquery
from google.cloud.bigquery import Row
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from .config import GCPConfig
from .encoder import EmbeddingEncoder
from .models import (
    Candidate,
    MatchResult,
    RecodeResult,
    ValueRecode,
    Variable,
    VariableMatch,
)
from .prompts import PromptLoader
from .survey import Survey


@dataclass
class _CandidateSet:
    """Internal: a target variable and its shortlisted candidates."""

    target: Variable
    candidates: list[Candidate]
    recode: RecodeResult | None = None


class VariableMatcher:
    """
    Match target survey variables against a candidate survey.

    Uses ``fit`` / ``predict``: *fit* encodes surveys and computes
    the similarity matrix; *predict* selects candidates, runs the
    LLM, and applies recodes via ``Survey.add_recode()``.

    Parameters
    ----------
    gcp_config : GCPConfig or None, optional
        GCP configuration. Defaults to ``GCPConfig()``.
    min_ratio : float, default 0.8
        Candidates must score at least this fraction of the
        best similarity score for each target variable.
    top_k : int, default 5
        Maximum number of candidates per target variable.
    include_values_in_encoding : bool, default True
        Whether to include value labels in the encoding text.
    max_wait : int, default 3600
        Maximum seconds to wait for a batch LLM job.
    """

    def __init__(
        self,
        gcp_config: GCPConfig | None = None,
        min_ratio: float = 0.8,
        top_k: int = 5,
        include_values_in_encoding: bool = True,
        max_wait: int = 3600,
    ) -> None:
        self.gcp_config = gcp_config or GCPConfig()
        self.min_ratio = min_ratio
        self.top_k = top_k
        self.include_values_in_encoding = include_values_in_encoding
        self.max_wait = max_wait

        self.encoder = EmbeddingEncoder(gcp_config=self.gcp_config)
        self.prompts = PromptLoader.from_yaml()

        # State populated by fit()
        self._target: Survey | None = None
        self._candidate: Survey | None = None
        self._target_embeddings: dict[str, list[float]] = {}
        self._candidate_embeddings: dict[str, list[float]] = {}
        self._target_vars: list[Variable] = []
        self._candidate_vars: list[Variable] = []
        self._sim_matrix: NDArray[np.floating] | None = None

    # -- Public API --------------------------------------------------------

    def fit(
        self,
        target: Survey,
        candidate: Survey,
        variable_names: list[str] | None = None,
    ) -> "VariableMatcher":
        """
        Encode both surveys and compute the similarity matrix.

        Parameters
        ----------
        target : Survey
            The target survey whose variables will be matched.
        candidate : Survey
            The candidate survey to search for matches in.
        variable_names : list of str or None, optional
            If provided, only match these target variables.

        Returns
        -------
        VariableMatcher
            Self, for chaining.
        """
        self._target = target
        self._candidate = candidate

        self._target_embeddings = self.encoder.encode_survey(
            target,
            include_values=self.include_values_in_encoding,
        )
        self._candidate_embeddings = self.encoder.encode_survey(
            candidate,
            include_values=self.include_values_in_encoding,
        )

        if variable_names:
            self._target_vars = [
                v
                for v in target.variables
                if v.name in variable_names
            ]
        else:
            self._target_vars = list(target.variables)

        self._candidate_vars = list(candidate.variables)

        if self._target_vars and self._candidate_vars:
            self._sim_matrix = self._compute_similarity_matrix(
                self._target_vars, self._candidate_vars
            )
        else:
            self._sim_matrix = None

        return self

    def predict(self) -> MatchResult:
        """
        Select candidates, run LLM verification, and apply recodes.

        Must call ``fit()`` first. Returns deep copies of both
        surveys — the originals are not modified.

        Returns
        -------
        MatchResult
            Contains the (possibly recoded) target and candidate
            surveys along with per-variable match metadata.
        """
        if self._target is None or self._candidate is None:
            raise RuntimeError("Call fit() before predict().")

        target_copy = copy.deepcopy(self._target)
        candidate_copy = copy.deepcopy(self._candidate)

        if self._sim_matrix is None or not self._target_vars:
            return MatchResult(
                target=target_copy,
                candidate=candidate_copy,
            )

        candidate_sets = self._build_candidate_sets()

        # LLM verification
        to_verify = [
            cs for cs in candidate_sets if cs.candidates
        ]
        if to_verify:
            if len(to_verify) == 1:
                self._verify_single(to_verify[0])
            else:
                self._verify_candidates(to_verify)

        # Build per-variable match metadata
        matches = self._build_variable_matches(candidate_sets)

        # Apply recodes to copies via Survey.add_recode()
        self._apply_recodes(
            target_copy, candidate_copy, candidate_sets
        )

        return MatchResult(
            target=target_copy,
            candidate=candidate_copy,
            matches=matches,
        )

    # -- Candidate selection -----------------------------------------------

    def _compute_similarity_matrix(
        self,
        target_vars: list[Variable],
        candidate_vars: list[Variable],
    ) -> NDArray[np.floating]:
        target_embs = np.array(
            [self._target_embeddings[v.name] for v in target_vars]
        )
        candidate_embs = np.array(
            [self._candidate_embeddings[v.name] for v in candidate_vars]
        )
        return cosine_similarity(target_embs, candidate_embs)

    def _build_candidate_sets(self) -> list[_CandidateSet]:
        sets = []
        for i, target_var in enumerate(self._target_vars):
            similarities = self._sim_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]

            best_score = float(similarities[sorted_indices[0]])
            cutoff = best_score * self.min_ratio

            candidates = []
            for rank, idx in enumerate(
                sorted_indices[: self.top_k], start=1
            ):
                score = float(similarities[idx])
                if score >= cutoff:
                    candidates.append(
                        Candidate(
                            variable=self._candidate_vars[idx],
                            similarity_score=score,
                            rank=rank,
                        )
                    )

            sets.append(
                _CandidateSet(
                    target=target_var, candidates=candidates
                )
            )

        return sets

    # -- LLM verification -------------------------------------------------

    @functools.cached_property
    def _genai_client(self) -> genai.Client:
        return self.gcp_config.get_genai_client()

    @functools.cached_property
    def _bq_client(self) -> bigquery.Client:
        client = self.gcp_config.get_bq_client()
        self._ensure_dataset_exists(client)
        return client

    def _verify_candidates(
        self,
        candidate_sets: list[_CandidateSet],
    ) -> None:
        table_name = f"verification_{int(time.time())}"
        self._run_batch_job_for_candidates(
            candidate_sets, table_name
        )
        recodes = self._load_recode_results(table_name)
        for cs, recode in zip(
            candidate_sets, recodes, strict=True
        ):
            cs.recode = recode

    def _verify_single(
        self, candidate_set: _CandidateSet
    ) -> None:
        prompt = self._create_verification_prompt(
            candidate_set
        )
        response = self._genai_client.models.generate_content(
            model=self.gcp_config.model_name,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        candidate_set.recode = self._parse_recode_response(
            response.text
        )

    # -- Prompt building ---------------------------------------------------

    def _create_verification_prompt(
        self,
        candidate_set: _CandidateSet,
    ) -> str:
        instruction = self.prompts.match_instruction
        response_format = self.prompts.match_response_format

        target_info = self._format_variable_info(
            candidate_set.target, "Target"
        )

        candidate_sections = []
        for candidate in candidate_set.candidates:
            section = self._format_variable_info(
                candidate.variable,
                f"Candidate {candidate.rank}",
            )
            section += (
                f"\n  Similarity score:"
                f" {candidate.similarity_score:.3f}"
            )
            candidate_sections.append(section)

        candidates_text = "\n\n".join(candidate_sections)

        return (
            f"{instruction}\n\n"
            f"{target_info}\n\n"
            f"{candidates_text}\n\n"
            f"{response_format}"
        )

    @staticmethod
    def _format_variable_info(
        variable: Any, prefix: str
    ) -> str:
        lines = [
            f"{prefix} Variable:",
            f"  Name: {variable.name}",
            f"  Question: "
            f"{variable.question or variable.label or 'N/A'}",
        ]
        if variable.values:
            lines.append("  Values:")
            for val in variable.values:
                statement = val.statement or val.label
                lines.append(
                    f"    - Code {val.code}: {statement}"
                )
        return "\n".join(lines)

    # -- Batch job ---------------------------------------------------------

    def _run_batch_job_for_candidates(
        self,
        candidate_sets: list[_CandidateSet],
        table_name: str,
    ) -> None:
        rows = []
        for idx, cs in enumerate(candidate_sets):
            prompt = self._create_verification_prompt(cs)
            request = {
                "contents": [
                    {
                        "parts": [{"text": prompt}],
                        "role": "user",
                    }
                ],
                "generationConfig": {
                    "responseMimeType": "application/json"
                },
            }
            rows.append(
                {
                    "row_index": idx,
                    "target_var": cs.target.name,
                    "request": json.dumps(request),
                }
            )

        schema = [
            bigquery.SchemaField("row_index", "INTEGER"),
            bigquery.SchemaField("target_var", "STRING"),
            bigquery.SchemaField("request", "STRING"),
        ]

        self._run_batch_job(
            rows=rows, schema=schema, table_name=table_name
        )

    def _run_batch_job(
        self,
        rows: list[dict[str, Any]],
        schema: list[bigquery.SchemaField],
        table_name: str,
    ) -> None:
        gcp = self.gcp_config
        full_table = (
            f"{gcp.project_id}.{gcp.bq_dataset}.{table_name}"
        )

        table = bigquery.Table(full_table, schema=schema)
        with contextlib.suppress(Conflict):
            self._bq_client.create_table(table)
        self._bq_client.insert_rows_json(full_table, rows)

        table_uri = f"bq://{full_table}"

        batch_job = self._genai_client.batches.create(
            model=gcp.model_name,
            src=table_uri,
            config={
                "display_name": table_name,
                "dest": table_uri,
            },
        )

        job_name = batch_job.name
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        start_time = time.time()
        while True:
            if time.time() - start_time > self.max_wait:
                raise TimeoutError(
                    f"Batch job {job_name} timed out "
                    f"after {self.max_wait}s"
                )
            batch_job = self._genai_client.batches.get(
                name=job_name
            )
            if batch_job.state.name in completed_states:
                break
            time.sleep(gcp.poll_interval)

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(
                f"Batch job failed: {batch_job.state.name}"
            )

    # -- Response parsing --------------------------------------------------

    def _load_recode_results(
        self, table_name: str
    ) -> list[RecodeResult]:
        gcp = self.gcp_config
        full_table = (
            f"{gcp.project_id}.{gcp.bq_dataset}.{table_name}"
        )
        query = (
            f"SELECT response FROM `{full_table}` "
            f"ORDER BY row_index"
        )
        query_job = self._bq_client.query(query)
        return [
            self._parse_row_response(row)
            for row in query_job.result()
        ]

    def _parse_row_response(self, row: Row) -> RecodeResult:
        try:
            resp = row.response
            if isinstance(resp, str):
                resp = json.loads(resp)
            text = resp["candidates"][0]["content"]["parts"][
                0
            ]["text"]
            return self._parse_recode_response(
                text, raw_response=resp
            )
        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
            TypeError,
        ) as e:
            return RecodeResult(
                is_match=False,
                match_confidence=0.0,
                reasoning=f"Failed to parse response: {e}",
            )

    @staticmethod
    def _parse_recode_response(
        response_text: str,
        raw_response: dict[str, Any] | None = None,
    ) -> RecodeResult:
        try:
            data = json.loads(response_text)

            candidate_recodes = None
            target_recodes = None
            needs_recode = data.get("needs_recode", False)

            if data.get("is_match") and needs_recode:
                if "candidate_recodes" in data:
                    candidate_recodes = [
                        ValueRecode(
                            original_code=r["original_code"],
                            original_label=r["original_label"],
                            new_code=r["new_code"],
                            new_label=r["new_label"],
                        )
                        for r in data["candidate_recodes"]
                    ]
                if "target_recodes" in data:
                    target_recodes = [
                        ValueRecode(
                            original_code=r["original_code"],
                            original_label=r["original_label"],
                            new_code=r["new_code"],
                            new_label=r["new_label"],
                        )
                        for r in data["target_recodes"]
                    ]

            return RecodeResult(
                is_match=data.get("is_match", False),
                match_confidence=float(
                    data.get("match_confidence", 0.0)
                ),
                needs_recode=needs_recode,
                matched_variable=data.get("matched_variable"),
                reasoning=data.get("reasoning"),
                standardised_label=data.get(
                    "standardised_label"
                ),
                candidate_recodes=candidate_recodes,
                target_recodes=target_recodes,
                raw_response=raw_response,
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as e:
            return RecodeResult(
                is_match=False,
                match_confidence=0.0,
                reasoning=(
                    f"Failed to parse JSON response: {e}"
                ),
                raw_response=raw_response,
            )

    # -- Result building & recode application ------------------------------

    @staticmethod
    def _build_variable_matches(
        candidate_sets: list[_CandidateSet],
    ) -> list[VariableMatch]:
        matches = []
        for cs in candidate_sets:
            if cs.recode and cs.recode.is_match:
                matched_name = cs.recode.matched_variable
                sim_score = 0.0
                for c in cs.candidates:
                    if c.variable.name == matched_name:
                        sim_score = c.similarity_score
                        break
                if not sim_score and cs.candidates:
                    sim_score = (
                        cs.candidates[0].similarity_score
                    )

                matches.append(
                    VariableMatch(
                        target_variable=cs.target.name,
                        candidate_variable=matched_name,
                        similarity_score=sim_score,
                        match_confidence=(
                            cs.recode.match_confidence
                        ),
                        is_match=True,
                        needs_recode=cs.recode.needs_recode,
                        reasoning=cs.recode.reasoning,
                        standardised_label=(
                            cs.recode.standardised_label
                        ),
                        target_recodes=(
                            cs.recode.target_recodes
                        ),
                        candidate_recodes=(
                            cs.recode.candidate_recodes
                        ),
                    )
                )
            else:
                sim_score = (
                    cs.candidates[0].similarity_score
                    if cs.candidates
                    else 0.0
                )
                matches.append(
                    VariableMatch(
                        target_variable=cs.target.name,
                        candidate_variable=None,
                        similarity_score=sim_score,
                        match_confidence=(
                            cs.recode.match_confidence
                            if cs.recode
                            else 0.0
                        ),
                        is_match=False,
                        needs_recode=False,
                        reasoning=(
                            cs.recode.reasoning
                            if cs.recode
                            else None
                        ),
                    )
                )

        return matches

    @staticmethod
    def _apply_recodes(
        target: Survey,
        candidate: Survey,
        candidate_sets: list[_CandidateSet],
    ) -> None:
        """Build Specification dicts and delegate to Survey.add_recode()."""
        target_specs: list[dict] = []
        candidate_specs: list[dict] = []

        for cs in candidate_sets:
            if (
                not cs.recode
                or not cs.recode.is_match
                or not cs.recode.needs_recode
            ):
                continue

            if cs.recode.target_recodes:
                groups: dict[str, list] = defaultdict(list)
                for r in cs.recode.target_recodes:
                    groups[r.new_label].append(
                        r.original_code
                    )
                target_specs.append(
                    {
                        "source_variable": cs.target.name,
                        "groups": dict(groups),
                        "new_label": (
                            cs.recode.standardised_label
                        ),
                    }
                )

            if (
                cs.recode.candidate_recodes
                and cs.recode.matched_variable
            ):
                groups = defaultdict(list)
                for r in cs.recode.candidate_recodes:
                    groups[r.new_label].append(
                        r.original_code
                    )
                candidate_specs.append(
                    {
                        "source_variable": (
                            cs.recode.matched_variable
                        ),
                        "groups": dict(groups),
                        "new_label": (
                            cs.recode.standardised_label
                        ),
                    }
                )

        if target_specs:
            target.add_recode(target_specs)

        if candidate_specs:
            candidate.add_recode(candidate_specs)

    # -- Infrastructure ----------------------------------------------------

    def _ensure_dataset_exists(
        self, client: bigquery.Client
    ) -> None:
        gcp = self.gcp_config
        dataset_id = f"{gcp.project_id}.{gcp.bq_dataset}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = gcp.location
        with contextlib.suppress(Conflict):
            client.create_dataset(dataset)
