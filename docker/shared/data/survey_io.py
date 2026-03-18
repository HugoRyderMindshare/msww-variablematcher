"""Survey I/O for SPSS datasets.

Provides ``SurveyLoader`` and ``SurveyWriter`` for reading and
writing ``Survey`` objects to ``.sav`` files via pyreadstat.

Convention: each dataset lives in ``{root}/{dataset}/`` and
contains ``target.sav`` and ``candidate.sav``.
"""

import os

import pyreadstat

from variablematcher import Survey


class SurveyLoader:
    """Loads target and candidate Survey objects from a dataset folder.

    Parameters
    ----------
    root : str
        Path to the top-level datasets directory.
    """

    def __init__(self, root: str = "data/datasets") -> None:
        self.root = root

    def load_target(self, dataset: str) -> Survey:
        """Load the target survey from ``{root}/{dataset}/target.sav``."""
        return self._load(dataset, "target")

    def load_candidate(self, dataset: str) -> Survey:
        """Load the candidate survey from ``{root}/{dataset}/candidate.sav``."""
        return self._load(dataset, "candidate")

    def _load(self, dataset: str, name: str) -> Survey:
        path = os.path.join(self.root, dataset, f"{name}.sav")
        df, meta = pyreadstat.read_sav(path)
        return Survey.from_sav(df, meta, name=name)


class SurveyWriter:
    """Writes target and candidate Survey objects to an output directory.

    Parameters
    ----------
    root : str
        Path to the output directory.
    """

    def __init__(self, root: str = "data/datasets") -> None:
        self.root = root

    def save_target(self, survey: Survey) -> str:
        """Save the target survey to ``{root}/target_recoded.sav``."""
        return self._save(survey, "target_recoded")

    def save_candidate(self, survey: Survey) -> str:
        """Save the candidate survey to ``{root}/candidate_recoded.sav``."""
        return self._save(survey, "candidate_recoded")

    def _save(self, survey: Survey, name: str) -> str:
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, f"{name}.sav")

        df, meta = survey.to_sav()
        pyreadstat.write_sav(
            df,
            path,
            column_labels=meta.column_labels,
            variable_value_labels=meta.variable_value_labels,
        )
        return path
