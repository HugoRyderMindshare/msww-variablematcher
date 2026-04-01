"""Content-agnostic utility modules."""

from .embedding import TextEmbedder
from .gemini_client import GeminiBatchClient

__all__ = ["GeminiBatchClient", "TextEmbedder"]
