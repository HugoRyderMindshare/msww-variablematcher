"""
Content-agnostic text embedding via Google GenAI.

Wraps the GenAI embedding API with async concurrency control.
Callers supply plain strings and get back embedding vectors.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Sequence
from dataclasses import dataclass

from google import genai

from .config import GCPConfig


@dataclass(frozen=True)
class EmbedderConfig:
    """Static configuration for the text embedder."""

    model: str = "gemini-embedding-001"
    max_concurrency: int = 10


class TextEmbedder:
    """Encode plain text strings into embedding vectors."""

    CONFIG = EmbedderConfig()

    def __init__(self) -> None:
        self._config = GCPConfig()
        self._loop = asyncio.new_event_loop()

    @functools.cached_property
    def _client(self) -> genai.Client:
        return self._config.get_genai_client()

    def encode_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a list of texts into embedding vectors.

        Parameters
        ----------
        texts : Sequence of str
            Texts to encode.

        Returns
        -------
        list of list of float
            One embedding vector per input text.
        """
        if not texts:
            return []
        return self._loop.run_until_complete(self._encode_texts_async(texts))

    async def _encode_texts_async(
        self, texts: Sequence[str]
    ) -> list[list[float]]:
        texts = list(texts)
        cfg = self.CONFIG
        embed_config = {"auto_truncate": False}
        sem = asyncio.Semaphore(cfg.max_concurrency)

        async def _embed(idx: int, text: str) -> tuple[int, list[float]]:
            async with sem:
                resp = await self._client.aio.models.embed_content(
                    model=cfg.model,
                    contents=text,
                    config=embed_config,
                )
                return idx, resp.embeddings[0].values

        tasks = [_embed(i, t) for i, t in enumerate(texts)]
        results_unordered = await asyncio.gather(*tasks)
        results = [None] * len(texts)
        for idx, values in results_unordered:
            results[idx] = values
        return results
