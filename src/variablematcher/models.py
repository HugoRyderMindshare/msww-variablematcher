"""Data models for survey variables, values, and match results."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .survey import Survey


@dataclass
class BaseModel:
    """Base dataclass with a repr that omits None-valued fields."""

    def __repr__(self) -> str:
        """Return a string showing only fields with non-None values."""
        parts = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return f"{self.__class__.__name__}({', '.join(parts)})"


@dataclass
class Value(BaseModel):
    """A single response option for a categorical survey variable.

    Parameters
    ----------
    code : int | float | str
        The numeric or string code stored in the SPSS data column.
    statement : str
        The standardised statement describing this response option.
    """

    code: int | float | str
    statement: str


@dataclass
class Variable(BaseModel):
    """A survey variable derived from SPSS metadata.

    Assumes the input SAV has already been standardised: the
    variable label is the standardised question text, and each
    value label is a standardised statement.

    Parameters
    ----------
    code : str
        The SPSS column name (e.g. ``'Q5a'``).
    question : str or None
        The standardised question text. Variables without a
        question are excluded from matching.
    values : tuple of Value or None
        Response options. Present for categorical variables,
        ``None`` for continuous.
    """

    code: str
    question: str | None = None
    values: tuple[Value, ...] | None = None

    def __hash__(self) -> int:
        return hash(self.code)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variable):
            return NotImplemented
        return self.code == other.code

    @property
    def value_codes(self) -> dict[str, int | float | str]:
        """Map each value statement to its code.

        Returns
        -------
        dict of str to int | float | str
            ``{statement: code}`` for every value, or empty dict
            if the variable has no values.
        """
        if not self.values:
            return {}
        return {v.statement: v.code for v in self.values}

    @property
    def is_categorical(self) -> bool:
        """Whether this variable has response options.

        Returns ``True`` if ``values`` is non-empty, indicating a
        categorical variable. Continuous variables have no values.
        """
        return bool(self.values)


@dataclass
class MatchSide(BaseModel):
    """One side (target or candidate) of a variable match.

    Parameters
    ----------
    variable : str
        The SPSS column name.
    groups : dict or None
        Recode group mapping. Keys are group labels, values are
        lists of SPSS value codes.
    """

    variable: str
    groups: dict[str, list] | None = None


@dataclass
class VariableMatch(BaseModel):
    """The outcome of matching one target variable.

    Produced by the LLM verification step. If the LLM confirmed
    a match, ``candidate`` is set and optional recode group
    mappings may be present on either side.

    Parameters
    ----------
    is_categorical : bool
        Whether the matched variables are categorical.
    target : MatchSide
        Target side of the match.
    candidate : MatchSide or None
        Candidate side, or ``None`` if unmatched.
    """

    is_categorical: bool
    target: MatchSide
    candidate: MatchSide | None = None

    @property
    def is_match(self) -> bool:
        """Whether the LLM confirmed a match for this target."""
        return self.candidate is not None


@dataclass
class MatchResult(BaseModel):
    """Result of matching a target survey against a candidate survey.

    Contains the filtered, positionally aligned Survey objects
    and per-variable match metadata.

    Parameters
    ----------
    target : Survey
        The target survey, filtered and ordered to matched columns.
    candidate : Survey
        The candidate survey, filtered and ordered to matched columns.
    matches : list of VariableMatch
        Per-variable match decisions from the LLM.
    """

    target: Survey
    candidate: Survey
    matches: list[VariableMatch] = field(default_factory=list)
