"""
Gemini client wrapping the GenAI API with BigQuery batch support.

Callers supply ``{key: prompt}`` and get back ``{key: response_text}``.
"""

from __future__ import annotations

import contextlib
import functools
import json
import time
import warnings
from typing import Any

from google import genai
from google.api_core.exceptions import Conflict
from google.cloud import bigquery

from .config import GCPConfig


class GeminiBatchClient:
    """Wraps the Gemini GenAI API with BigQuery batch support."""

    def __init__(self) -> None:
        self._config = GCPConfig()

    @functools.cached_property
    def _genai_client(self) -> genai.Client:
        return self._config.get_genai_client()

    @functools.cached_property
    def _bq_client(self) -> bigquery.Client:
        client = bigquery.Client(project=self._config.project_id)
        self._ensure_dataset_exists(client)
        return client

    def _ensure_dataset_exists(self, client: bigquery.Client) -> None:
        dataset_id = f"{self._config.project_id}.{self._config.bq_dataset}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = self._config.location
        with contextlib.suppress(Conflict):
            client.create_dataset(dataset)

    def generate(
        self,
        prompts: dict[str, str],
        table_name: str | None = None,
    ) -> dict[str, str]:
        """
        Submit prompts as a BQ batch job and return responses.

        Keys whose responses could not be extracted are omitted.
        """
        if table_name is None:
            table_name = f"batch_{int(time.time())}"

        keys = list(prompts.keys())
        rows = self._build_rows(keys, prompts)
        schema = [
            bigquery.SchemaField("row_index", "INTEGER"),
            bigquery.SchemaField("request", "STRING"),
        ]

        self._submit(rows, schema, table_name)
        results = self._load_responses(keys, table_name)

        missing = set(keys) - set(results.keys())
        if missing:
            warnings.warn(
                f"Batch response missing {len(missing)} key(s): {missing}",
                stacklevel=2,
            )

        return results

    @staticmethod
    def _build_rows(
        keys: list[str], prompts: dict[str, str]
    ) -> list[dict[str, Any]]:
        rows = []
        for idx, key in enumerate(keys):
            request = {
                "contents": [
                    {
                        "parts": [{"text": prompts[key]}],
                        "role": "user",
                    }
                ],
                "generationConfig": {"responseMimeType": "application/json"},
            }
            rows.append(
                {
                    "row_index": idx,
                    "request": json.dumps(request),
                }
            )
        return rows

    def _submit(
        self,
        rows: list[dict[str, Any]],
        schema: list[bigquery.SchemaField],
        table_name: str,
    ) -> None:
        full_table = (
            f"{self._config.project_id}.{self._config.bq_dataset}.{table_name}"
        )

        table = bigquery.Table(full_table, schema=schema)
        with contextlib.suppress(Conflict):
            self._bq_client.create_table(table)

        max_bytes = 9 * 1024 * 1024
        chunk: list[dict[str, Any]] = []
        chunk_bytes = 0

        for row in rows:
            row_bytes = len(json.dumps(row).encode())
            if chunk and chunk_bytes + row_bytes > max_bytes:
                self._bq_client.insert_rows_json(full_table, chunk)
                chunk = []
                chunk_bytes = 0
            chunk.append(row)
            chunk_bytes += row_bytes

        if chunk:
            self._bq_client.insert_rows_json(full_table, chunk)

        table_uri = f"bq://{full_table}"
        batch_job = self._genai_client.batches.create(
            model=self._config.model_name,
            src=table_uri,
            config={
                "display_name": table_name,
                "dest": table_uri,
            },
        )

        job_name = batch_job.name
        completed_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }

        while True:
            batch_job = self._genai_client.batches.get(name=job_name)
            if batch_job.state.name in completed_states:
                break
            time.sleep(self._config.poll_interval)

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Batch job failed: {batch_job.state.name}")

    def _load_responses(
        self,
        keys: list[str],
        table_name: str,
    ) -> dict[str, str]:
        full_table = (
            f"{self._config.project_id}.{self._config.bq_dataset}.{table_name}"
        )
        query = f"SELECT response FROM `{full_table}` ORDER BY row_index"
        results: dict[str, str] = {}
        for idx, row in enumerate(self._bq_client.query(query).result()):
            text = self._extract_text(row.response)
            if text is not None:
                results[keys[idx]] = text
        return results

    @staticmethod
    def _extract_text(response: Any) -> str | None:
        try:
            if isinstance(response, str):
                response = json.loads(response)
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
            TypeError,
        ):
            return None
