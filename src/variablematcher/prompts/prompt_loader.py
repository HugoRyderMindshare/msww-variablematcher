"""Prompt loader for the matching engine."""

from dataclasses import dataclass
from pathlib import Path

import yaml  # pyright: ignore[reportMissingModuleSource]

_DEFAULT_PATH = Path(__file__).parent / "prompts.yaml"


@dataclass(frozen=True)
class PromptLoader:
    """
    Typed, read-only access to prompt templates.

    Parameters
    ----------
    match_instruction : str
        Instruction for variable match verification and recode design.
    match_response_format : str
        Expected JSON response format for match verification.
    """

    match_instruction: str
    match_response_format: str

    @classmethod
    def from_yaml(
        cls, path: Path = _DEFAULT_PATH
    ) -> "PromptLoader":
        """
        Load prompts from a YAML file.

        Parameters
        ----------
        path : Path, optional
            Path to the YAML file. Defaults to the bundled
            ``prompts.yaml`` shipped with this package.

        Returns
        -------
        PromptLoader
            Frozen instance exposing every prompt as a typed
            attribute.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If a required prompt section or key is missing.
        """
        data = yaml.safe_load(path.read_text())
        return cls(
            match_instruction=data["match_verification"][
                "instruction"
            ],
            match_response_format=data["match_verification"][
                "response_format"
            ],
        )
