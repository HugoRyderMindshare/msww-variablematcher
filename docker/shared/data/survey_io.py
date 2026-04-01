"""Survey I/O for SPSS datasets.

Provides ``SurveyLoader`` and ``SurveyWriter`` for reading and
writing ``Survey`` objects to ``.sav`` files via pyreadstat.

Convention: each dataset lives in ``{root}/{dataset}/`` and
contains one ``.sav`` file per survey.
"""

import os

import pyreadstat

from variablematcher import Survey


class SurveyLoader:
    """Loads Survey objects from .sav files in a dataset folder.

    Parameters
    ----------
    root : str
        Path to the top-level datasets directory.
    """

    def __init__(self, root: str = "data/datasets") -> None:
        self.root = root

    def load(self, dataset: str, name: str) -> Survey:
        """Load a survey from ``{root}/{dataset}/{name}.sav``."""
        path = os.path.join(self.root, dataset, f"{name}.sav")
        df, meta = pyreadstat.read_sav(path)
        return Survey.from_sav(df, meta, name=name)


class SurveyWriter:
    """Writes Survey objects to an output directory.

    Parameters
    ----------
    root : str
        Path to the output directory.
    """

    def __init__(self, root: str = "data/datasets") -> None:
        self.root = root

    def save(self, survey: Survey) -> str:
        """Save the survey to ``{root}/{survey.name}.sav``."""
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, f"{survey.name}.sav")

        df, meta = survey.to_sav()
        pyreadstat.write_sav(
            df,
            path,
            column_labels=meta.column_labels,
            variable_value_labels=meta.variable_value_labels,
        )
        return path
