"""
Configuration module for GCP services.

This module provides configuration management for Google Cloud Platform services
used in variable matching, including Vertex AI embeddings and BigQuery batch
processing.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from google import genai
from google.cloud import bigquery


@dataclass
class GCPConfig:
    """
    Configuration for Google Cloud Platform services.

    This dataclass holds configuration parameters for connecting to GCP services
    including Vertex AI (Gemini) for verification, embeddings API, and BigQuery.

    Parameters
    ----------
    project_id : str, optional
        GCP project ID. Defaults to GCP_PROJECT_ID environment variable.
    location : str, optional
        GCP location/region. Defaults to GCP_LOCATION environment variable
        or 'us-central1'.
    bq_dataset : str, optional
        BigQuery dataset name for caching and batch jobs. Defaults to BQ_DATASET
        environment variable or 'variable_matcher'.
    model_name : str, default 'gemini-2.5-flash'
        Name of the Gemini model to use for verification.
    embedding_model : str, default 'text-embedding-005'
        Name of the embedding model to use for encoding.
    poll_interval : int, default 30
        Interval in seconds between batch job status checks.

    Examples
    --------
    >>> config = GCPConfig(project_id='my-gcp-project')
    >>> client = config.get_genai_client()
    """

    project_id: str = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID", ""))
    location: str = field(
        default_factory=lambda: os.getenv("GCP_LOCATION", "us-central1")
    )
    bq_dataset: str = field(
        default_factory=lambda: os.getenv("BQ_DATASET", "variable_matcher")
    )
    model_name: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"
    embedding_dimensions: int = 768
    poll_interval: int = 30

    def _validate_project_id(
        self,
    ) -> None:
        """
        Validate that project_id is set.

        Raises
        ------
        ValueError
            If project_id is empty or not set.
        """
        if not self.project_id:
            raise ValueError(
                "project_id is required. Set GCP_PROJECT_ID environment variable."
            )

    def get_genai_client(
        self,
    ) -> genai.Client:
        """
        Create and return a Google GenAI client for Vertex AI.

        Returns
        -------
        genai.Client
            Configured GenAI client for Vertex AI.

        Raises
        ------
        ValueError
            If project_id is not configured.

        Examples
        --------
        >>> config = GCPConfig(project_id='my-project')
        >>> client = config.get_genai_client()
        """
        self._validate_project_id()

        return genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def get_bq_client(
        self,
    ) -> bigquery.Client:
        """
        Create and return a BigQuery client.

        Returns
        -------
        bigquery.Client
            Configured BigQuery client.

        Raises
        ------
        ValueError
            If project_id is not configured.

        Examples
        --------
        >>> config = GCPConfig(project_id='my-project')
        >>> bq_client = config.get_bq_client()
        """
        self._validate_project_id()

        return bigquery.Client(project=self.project_id)


