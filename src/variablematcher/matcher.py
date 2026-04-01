"""
Variable matching module.

Provides the VariableMatcher class with fit/predict API for matching
target survey variables against a candidate survey.
"""

import copy
import json
import time

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity
from variablerecoder import Specification

from .models import (
    MatchResult,
    MatchSide,
    Variable,
    VariableMatch,
)
from .prompts import PromptLoader
from .survey import Survey
from .utils import GeminiBatchClient, TextEmbedder


class VariableMatcher:
    """
    Match target survey variables against a candidate survey.

    Uses ``fit`` / ``predict``: *fit* encodes surveys and computes
    the similarity matrix; *predict* selects candidates, runs the
    LLM, and applies recodes via ``Survey.add_recode()``.

    Parameters
    ----------
    min_ratio : float, default 0.8
        Candidates must score at least this fraction of the
        best similarity score for each target variable.
    top_k : int, default 5
        Maximum number of candidates per target variable.
    """

    def __init__(
        self,
        min_ratio: float = 0.8,
        top_k: int = 5,
    ) -> None:
        self.min_ratio = min_ratio
        self.top_k = top_k

        self._embedder = TextEmbedder()
        self.prompts = PromptLoader.from_yaml()
        self._llm = GeminiBatchClient()

        self._target: Survey | None = None
        self._candidate: Survey | None = None
        self._target_embeddings: dict[str, list[float]] = {}
        self._candidate_embeddings: dict[str, list[float]] = {}
        self._target_vars: list[Variable] = []
        self._candidate_vars: list[Variable] = []
        self._sim_matrix: NDArray[np.floating] | None = None

    def fit(
        self,
        target: Survey,
        candidate: Survey,
    ) -> "VariableMatcher":
        """
        Encode both surveys and compute the similarity matrix.

        Variables without a question label are excluded.

        Parameters
        ----------
        target : Survey
            The target survey whose variables will be matched.
        candidate : Survey
            The candidate survey to search for matches in.

        Returns
        -------
        VariableMatcher
            Self, for chaining.
        """
        self._target = target
        self._candidate = candidate

        self._target_vars = [v for v in target.variables if v.question]
        self._candidate_vars = [v for v in candidate.variables if v.question]

        self._target_embeddings = self._encode_variables(self._target_vars)
        self._candidate_embeddings = self._encode_variables(self._candidate_vars)

        if self._target_vars and self._candidate_vars:
            self._sim_matrix = self._compute_similarity_matrix(
                self._target_vars, self._candidate_vars
            )
        else:
            self._sim_matrix = None

        return self

    def _encode_variables(
        self, variables: list[Variable],
    ) -> dict[str, list[float]]:
        texts = [v.question for v in variables]
        if not texts:
            return {}
        embeddings = self._embedder.encode_texts(texts)
        return {
            v.code: emb for v, emb in zip(variables, embeddings, strict=True)
        }

    def predict(self) -> MatchResult:
        """
        Select candidates, run LLM verification, and apply recodes.

        Must call ``fit()`` first. Returns deep copies of both
        surveys — the originals are not modified.

        Returns
        -------
        MatchResult
            Contains the filtered, positionally aligned target and
            candidate surveys, the column mapping, and per-variable
            match metadata.
        """
        target_copy = copy.deepcopy(self._target)
        candidate_copy = copy.deepcopy(self._candidate)

        candidate_sets = self._build_candidate_sets()

        to_verify = {
            tv: cands for tv, cands in candidate_sets.items() if cands
        }
        matches = self._verify_candidates(to_verify) if to_verify else []
        confirmed = [m for m in matches if m.is_match]

        self._process_matches(target_copy, candidate_copy, confirmed)

        return MatchResult(
            target=target_copy,
            candidate=candidate_copy,
            matches=matches,
        )

    def _compute_similarity_matrix(
        self,
        target_vars: list[Variable],
        candidate_vars: list[Variable],
    ) -> NDArray[np.floating]:
        target_embs = np.array(
            [self._target_embeddings[v.code] for v in target_vars]
        )
        candidate_embs = np.array(
            [self._candidate_embeddings[v.code] for v in candidate_vars]
        )
        sim = cosine_similarity(target_embs, candidate_embs)

        target_cat = np.array([v.is_categorical for v in target_vars])
        candidate_cat = np.array([v.is_categorical for v in candidate_vars])
        dtype_mask = target_cat[:, None] == candidate_cat[None, :]
        sim[~dtype_mask] = 0.0

        return sim

    def _build_candidate_sets(
        self,
    ) -> dict[Variable, list[Variable]]:
        result: dict[Variable, list[Variable]] = {}
        for i, target_var in enumerate(self._target_vars):
            similarities = self._sim_matrix[i]
            sorted_indices = np.argsort(similarities)[::-1]

            best_score = float(similarities[sorted_indices[0]])
            cutoff = best_score * self.min_ratio

            candidates = []
            for idx in sorted_indices[: self.top_k]:
                score = float(similarities[idx])
                if score >= cutoff:
                    candidates.append(self._candidate_vars[idx])

            result[target_var] = candidates

        return result

    def _verify_candidates(
        self,
        candidate_sets: dict[Variable, list[Variable]],
    ) -> list[VariableMatch]:
        prompts = {
            tv.code: self._create_verification_prompt(tv, cands)
            for tv, cands in candidate_sets.items()
        }
        table_name = f"verification_{int(time.time())}"
        responses = self._llm.generate(prompts, table_name)
        return [
            self._parse_llm_response(responses.get(tv.code), tv, cands)
            for tv, cands in candidate_sets.items()
        ]

    def _create_verification_prompt(
        self,
        target: Variable,
        candidates: list[Variable],
    ) -> str:
        instruction = self.prompts.match_instruction
        response_format = self.prompts.match_response_format

        target_info = self._format_variable_info(target, "Target")

        candidate_sections = []
        for i, candidate in enumerate(candidates, 1):
            section = self._format_variable_info(
                candidate,
                f"[{i}] Candidate",
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
    def _format_variable_info(variable: Variable, prefix: str) -> str:
        lines = [
            f"{prefix} Variable:",
            f"  Question: {variable.question or 'N/A'}",
        ]
        if variable.values:
            lines.append("  Values:")
            for i, val in enumerate(variable.values, 1):
                statement = val.statement
                lines.append(f"    - V{i}: {statement}")
        return "\n".join(lines)

    @staticmethod
    def _assignments_to_groups(
        variable: Variable,
        assignments: dict[str, str],
    ) -> dict[str, list]:
        """Convert per-value group assignments to group → codes mapping."""
        values = variable.values or []
        ref_map = {f"V{i}": val.code for i, val in enumerate(values, 1)}
        groups: dict[str, list] = {}
        for vref, label in assignments.items():
            if vref in ref_map:
                groups.setdefault(label, []).append(ref_map[vref])
        return groups

    def _parse_llm_response(
        self,
        text: str | None,
        target: Variable,
        candidates: list[Variable],
    ) -> VariableMatch:
        """Parse LLM JSON, resolve rank and V-refs, return a VariableMatch."""
        if text is None:
            return VariableMatch(
                is_categorical=target.is_categorical,
                target=MatchSide(variable=target.code),
            )

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return VariableMatch(
                is_categorical=target.is_categorical,
                target=MatchSide(variable=target.code),
            )

        rank = data.get("matched_candidate")
        if rank is None:
            return VariableMatch(
                is_categorical=target.is_categorical,
                target=MatchSide(variable=target.code),
            )

        rank = int(rank)
        if rank < 1 or rank > len(candidates):
            return VariableMatch(
                is_categorical=target.is_categorical,
                target=MatchSide(variable=target.code),
            )

        matched = candidates[rank - 1]

        target_groups = None
        candidate_groups = None
        recode = data.get("recode")
        if recode:
            raw_ta = recode.get("target_assignments")
            raw_ca = recode.get("candidate_assignments")
            if raw_ta:
                target_groups = self._assignments_to_groups(target, raw_ta)
            if raw_ca:
                candidate_groups = self._assignments_to_groups(matched, raw_ca)

        return VariableMatch(
            is_categorical=target.is_categorical,
            target=MatchSide(
                variable=target.code,
                groups=target_groups,
            ),
            candidate=MatchSide(
                variable=matched.code,
                groups=candidate_groups,
            ),
        )

    @staticmethod
    def _process_matches(
        target: Survey,
        candidate: Survey,
        matches: list[VariableMatch],
    ) -> None:
        """Build specs, apply recodes, filter and order both surveys."""
        def _groups(groups: dict | None, is_cat: bool) -> dict | None:
            if groups:
                seen: set = set()
                clean: dict[str, list] = {}
                for label, codes in groups.items():
                    unique = [c for c in codes if c not in seen]
                    seen.update(unique)
                    if unique:
                        clean[label] = unique
                return clean if clean else ({} if is_cat else None)
            return {} if is_cat else None

        target_specs = [
            Specification(
                source_code=m.target.variable,
                groups=_groups(m.target.groups, m.is_categorical),
            )
            for m in matches
        ]
        candidate_specs = [
            Specification(
                source_code=m.candidate.variable,
                groups=_groups(m.candidate.groups, m.is_categorical),
            )
            for m in matches
        ]

        target.add_recode(target_specs)
        candidate.add_recode(candidate_specs)

        target.filter_to([s.new_code for s in target_specs])
        candidate.filter_to([s.new_code for s in candidate_specs])
