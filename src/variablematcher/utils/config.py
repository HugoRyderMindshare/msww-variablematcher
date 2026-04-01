"""GCP configuration for Vertex AI and BigQuery services."""

import os
from dataclasses import dataclass, field

from google import genai


@dataclass
class GCPConfig:
    """Configuration for Google Cloud Platform services.

    Parameters
    ----------
    project_id : str, optional
        GCP project ID. Defaults to GCP_PROJECT_ID env var.
    location : str, optional
        GCP region. Defaults to GCP_LOCATION env var.
    bq_dataset : str, optional
        BigQuery dataset for batch jobs. Defaults to BQ_DATASET env var
        or 'variable_matcher'.
    model_name : str, default 'gemini-2.5-flash'
        Gemini model for verification.
    poll_interval : int, default 30
        Seconds between batch job status checks.
    """

    project_id: str = field(
        default_factory=lambda: os.getenv("GCP_PROJECT_ID", "")
    )
    location: str = field(
        default_factory=lambda: os.getenv("GCP_LOCATION", "")
    )
    bq_dataset: str = field(
        default_factory=lambda: os.getenv("BQ_DATASET", "variable_matcher")
    )
    model_name: str = "gemini-2.5-flash"
    poll_interval: int = 30

    def _validate_project_id(self) -> None:
        if not self.project_id:
            raise ValueError(
                "project_id is required. "
                "Set GCP_PROJECT_ID environment variable."
            )

    def _validate_location(self) -> None:
        if not self.location:
            raise ValueError(
                "location is required. Set GCP_LOCATION environment variable."
            )

    def get_genai_client(self) -> genai.Client:
        """Create a GenAI client for Vertex AI."""
        self._validate_project_id()
        self._validate_location()
        return genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )
