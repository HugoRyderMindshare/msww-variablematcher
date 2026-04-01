"""Survey data structure module."""

from __future__ import annotations

import warnings

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
                "No data available. Use Survey.from_sav() to load data."
            )

        return self._data.copy(), self._meta

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
            var_label = self._meta.column_names_to_labels.get(var_name)

            values = None
            if var_name in self._meta.variable_value_labels:
                vvl = self._meta.variable_value_labels[var_name]
                values = [
                    Value(code=code, statement=str(label))
                    for code, label in vvl.items()
                ]

            result.append(
                Variable(
                    code=var_name,
                    question=var_label,
                    values=values,
                )
            )

        return tuple(result)

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
            s if isinstance(s, Specification) else Specification(**s)
            for s in specifications
        ]

        for spec in specs:
            recoder = VariableRecoder(self._data, self._meta)
            try:
                recoder.add(spec)
            except ValueError as e:
                warnings.warn(
                    f"Skipping recode for '{spec.source_code}': {e}",
                    stacklevel=2,
                )
                continue
            self._data, self._meta = recoder.build()

    def filter_to(self, columns: list[str]) -> None:
        """Filter and reorder data and metadata to the given columns.

        Modifies the survey in-place. Only the listed columns
        are kept, in the order provided.

        Parameters
        ----------
        columns : list of str
            Column names to keep, in the desired order.
        """
        if self._data is None or self._meta is None:
            raise ValueError(
                "No data available. Use Survey.from_sav() to "
                "load data before filtering."
            )

        self._data = self._data[columns]

        label_map = dict(
            zip(self._meta.column_names, self._meta.column_labels)
        )
        self._meta.column_names = list(columns)
        self._meta.column_labels = [label_map.get(c, "") for c in columns]

        col_set = set(columns)
        for attr in (
            "column_names_to_labels",
            "variable_value_labels",
            "variable_measure",
            "variable_display_width",
            "variable_storage_width",
            "variable_alignment",
            "readstat_variable_types",
            "original_variable_types",
            "missing_ranges",
            "missing_user_values",
        ):
            d = getattr(self._meta, attr, None)
            if isinstance(d, dict):
                setattr(
                    self._meta,
                    attr,
                    {k: v for k, v in d.items() if k in col_set},
                )

        self._meta.number_columns = len(columns)
