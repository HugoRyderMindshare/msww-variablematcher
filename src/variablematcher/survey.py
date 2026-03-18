"""Survey data structure module."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

import pandas as pd
from variablerecoder import Specification, VariableRecoder

from .models import Value, Variable


class Survey:
    """
    Container for survey data and metadata.

    The single source of truth is the ``(_data, _meta)`` pair.
    ``variables`` is a derived property rebuilt from ``_meta``
    on every access.

    Parameters
    ----------
    data : pd.DataFrame or None
        The survey DataFrame.
    meta : object or None
        The pyreadstat metadata object.
    name : str or None, optional
        Name identifier for this survey.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        meta: object = None,
        name: str | None = None,
    ) -> None:
        self._data = data
        self._meta = meta
        self.name = name

    @classmethod
    def from_sav(
        cls,
        df: pd.DataFrame,
        meta: object,
        name: str | None = None,
    ) -> Survey:
        """
        Create a Survey from a pyreadstat DataFrame and metadata.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame returned by ``pyreadstat.read_sav()``.
        meta : pyreadstat metadata object
            The metadata returned by ``pyreadstat.read_sav()``.
        name : str or None, optional
            Name identifier for this survey.

        Returns
        -------
        Survey
        """
        return cls(data=df, meta=meta, name=name)

    @classmethod
    def from_dict(
        cls,
        data: Iterable[dict[str, dict[int | str, str] | list[str]]],
        name: str | None = None,
    ) -> Survey:
        """
        Create a Survey from a list of variable-values dictionaries.

        Builds a minimal metadata-like object so ``variables`` works.
        No DataFrame is stored; ``to_sav()`` will raise.

        Parameters
        ----------
        data : Iterable of dict
            Each key is a variable name, value is ``{code: label}``
            or ``[labels]``.
        name : str or None, optional
            Name identifier for this survey.

        Returns
        -------
        Survey
        """
        meta = _DictMeta.from_dict(data)
        return cls(data=None, meta=meta, name=name)

    # -- Derived property --------------------------------------------------

    @property
    def variables(self) -> tuple[Variable, ...]:
        """
        Build and return Variable objects from ``_meta``.

        Rebuilt on every access — always consistent with the
        current state of ``_meta``.
        """
        if self._meta is None:
            return ()

        result = []
        for var_name in self._meta.column_names:
            var_label = self._meta.column_names_to_labels.get(
                var_name
            )

            values = None
            if var_name in self._meta.variable_value_labels:
                vvl = self._meta.variable_value_labels[var_name]
                values = [
                    Value(code=code, label=str(label))
                    for code, label in vvl.items()
                ]

            result.append(
                Variable(
                    name=var_name,
                    question=var_label,
                    values=values,
                )
            )

        return tuple(result)

    @property
    def variable_names(self) -> tuple[str, ...]:
        if self._meta is None:
            return ()
        return tuple(self._meta.column_names)

    @property
    def is_empty(self) -> bool:
        return len(self.variable_names) == 0

    # -- Recode ------------------------------------------------------------

    def add_recode(
        self,
        specifications: list[Specification] | list[dict],
    ) -> None:
        """
        Apply recode specifications to the underlying data/meta.

        Parameters
        ----------
        specifications : list of Specification or list of dict
            If dicts, each must have keys accepted by
            ``Specification(**d)``.

        Raises
        ------
        ValueError
            If no data is available.
        """
        if self._data is None:
            raise ValueError(
                "No data available. Use Survey.from_sav() to "
                "load data before recoding."
            )

        specs = [
            s
            if isinstance(s, Specification)
            else Specification(**s)
            for s in specifications
        ]

        recoder = VariableRecoder(self._data, self._meta)
        recoder.add(specs)
        self._data, self._meta = recoder.build()

    # -- Export ------------------------------------------------------------

    def to_sav(self) -> tuple[pd.DataFrame, object]:
        """
        Return ``(DataFrame, metadata)`` in pyreadstat read format.

        Returns
        -------
        tuple of (pd.DataFrame, pyreadstat metadata)

        Raises
        ------
        ValueError
            If no data is available.
        """
        if self._data is None:
            raise ValueError(
                "No data available. Use Survey.from_sav() to "
                "load data."
            )

        return self._data.copy(), self._meta

    # -- Lookup & filter ---------------------------------------------------

    def get_variable_by_name(
        self, name: str
    ) -> Variable | None:
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def filter_variables(
        self,
        names: list[str] | None = None,
        has_values: bool | None = None,
        label_contains: str | None = None,
    ) -> list[Variable]:
        result = list(self.variables)

        if names is not None:
            result = [v for v in result if v.name in names]

        if has_values is not None:
            if has_values:
                result = [v for v in result if v.values]
            else:
                result = [v for v in result if not v.values]

        if label_contains is not None:
            label_lower = label_contains.lower()
            result = [
                v
                for v in result
                if v.label and label_lower in v.label.lower()
            ]

        return result

    # -- Dunder methods ----------------------------------------------------

    def __hash__(self) -> int:
        content_parts = []
        for var in self.variables:
            var_part = f"{var.name}:{var.label or ''}"
            if var.values:
                value_parts = [
                    f"{v.code}={v.label}" for v in var.values
                ]
                var_part += f"[{','.join(value_parts)}]"
            content_parts.append(var_part)

        content = "|".join(content_parts)
        hash_hex = hashlib.md5(
            content.encode(), usedforsecurity=False
        ).hexdigest()
        return int(hash_hex[:16], 16)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Survey):
            return NotImplemented
        return hash(self) == hash(other)

    def __len__(self) -> int:
        return len(self.variable_names)

    def __getitem__(self, key: str | int) -> Variable:
        variables = self.variables
        if isinstance(key, int):
            return variables[key]
        for var in variables:
            if var.name == key:
                return var
        raise KeyError(f"Variable not found: {key}")




class _DictMeta:
    """Minimal metadata stand-in for ``from_dict()`` surveys."""

    def __init__(
        self,
        column_names: list[str],
        column_names_to_labels: dict[str, str],
        variable_value_labels: dict[str, dict],
    ) -> None:
        self.column_names = column_names
        self.column_names_to_labels = column_names_to_labels
        self.variable_value_labels = variable_value_labels

    @classmethod
    def from_dict(
        cls,
        data: Iterable[dict[str, dict[int | str, str] | list[str]]],
    ) -> _DictMeta:
        column_names: list[str] = []
        column_names_to_labels: dict[str, str] = {}
        variable_value_labels: dict[str, dict] = {}

        for item in data:
            for var_name, value_data in item.items():
                column_names.append(var_name)
                column_names_to_labels[var_name] = var_name

                if isinstance(value_data, dict):
                    variable_value_labels[var_name] = {
                        code: str(label)
                        for code, label in value_data.items()
                    }
                else:
                    variable_value_labels[var_name] = {
                        idx + 1: str(label)
                        for idx, label in enumerate(value_data)
                    }

        return cls(
            column_names=column_names,
            column_names_to_labels=column_names_to_labels,
            variable_value_labels=variable_value_labels,
        )
