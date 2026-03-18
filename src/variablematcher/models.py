"""
Data models for survey variables, values, and match results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .survey import Survey


@dataclass
class BaseModel:
    """Base dataclass with custom string representation."""

    def __repr__(self) -> str:
        parts = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


@dataclass
class Value(BaseModel):
    """
    A survey value or response option.

    Parameters
    ----------
    code : int | float | str
        The numeric or string code for this value.
    label : str
        The label for this value.
    statement : str or None, optional
        A standardised statement describing this value.
    embedding : list of float or None, optional
        The embedding vector for this value's text.
    """

    code: int | float | str
    label: str
    statement: str | None = None


@dataclass
class Variable(BaseModel):
    """
    A survey variable or question.

    Parameters
    ----------
    name : str
        The variable name (column name).
    label : str or None, optional
        The original variable label.
    question : str or None, optional
        The standardised question text.
    values : list of Value or None, optional
        Possible values/response options.
    """

    name: str
    label: str | None = None
    question: str | None = None
    values: list[Value] | None = None

    @property
    def value_codes(self) -> dict[str, int | float | str]:
        """Mapping of value labels to codes."""
        if not self.values:
            return {}
        return {v.label: v.code for v in self.values}


@dataclass
class ValueRecode(BaseModel):
    """
    Maps an original value code to a new standardised code.

    Parameters
    ----------
    original_code : int | float | str
        The original code in the survey variable.
    original_label : str
        The original label in the survey variable.
    new_code : int
        The new standardised code.
    new_label : str
        The new standardised label.
    """

    original_code: int | float | str
    original_label: str
    new_code: int
    new_label: str


# -- Public result types -----------------------------------------------


@dataclass
class VariableMatch(BaseModel):
    """
    Per-variable match decision from the LLM.

    Parameters
    ----------
    target_variable : str
        Name of the target variable.
    candidate_variable : str or None
        Name of the matched candidate variable, or None.
    similarity_score : float
        Cosine similarity between target and matched candidate.
    match_confidence : float
        LLM confidence score (0-1).
    is_match : bool
        Whether the LLM confirmed a semantic match.
    needs_recode : bool
        Whether values need realignment.
    reasoning : str or None, optional
        LLM explanation of match/recode decisions.
    standardised_label : str or None, optional
        Unified variable label for the recoded variable.
    target_recodes : list of ValueRecode or None, optional
        Recode instructions for the target variable.
    candidate_recodes : list of ValueRecode or None, optional
        Recode instructions for the candidate variable.
    """

    target_variable: str
    candidate_variable: str | None
    similarity_score: float
    match_confidence: float
    is_match: bool
    needs_recode: bool
    reasoning: str | None = None
    standardised_label: str | None = None
    target_recodes: list[ValueRecode] | None = None
    candidate_recodes: list[ValueRecode] | None = None


@dataclass
class MatchResult(BaseModel):
    """
    Result of matching a target survey against a candidate survey.

    Contains the two (possibly recoded) Survey objects and
    per-variable match metadata.

    Parameters
    ----------
    target : Survey
        The target survey (recoded if matches were found).
    candidate : Survey
        The candidate survey (recoded if matches were found).
    matches : list of VariableMatch
        Per-variable match decisions.
    """

    target: Survey
    candidate: Survey
    matches: list[VariableMatch] = field(
        default_factory=list
    )

    @property
    def matched(self) -> list[VariableMatch]:
        """Only the successful matches."""
        return [m for m in self.matches if m.is_match]

    @property
    def unmatched(self) -> list[VariableMatch]:
        """Only the unmatched target variables."""
        return [m for m in self.matches if not m.is_match]


# -- Internal types (used by VariableMatcher) --------------------------


@dataclass
class Candidate(BaseModel):
    """A shortlisted candidate variable for matching (internal)."""

    variable: Variable
    similarity_score: float
    rank: int


@dataclass
class RecodeResult(BaseModel):
    """Parsed LLM recode response (internal)."""

    is_match: bool
    match_confidence: float
    needs_recode: bool = False
    matched_variable: str | None = None
    reasoning: str | None = None
    standardised_label: str | None = None
    candidate_recodes: list[ValueRecode] | None = None
    target_recodes: list[ValueRecode] | None = None
    raw_response: dict[str, Any] | None = field(
        default=None, repr=False
    )
