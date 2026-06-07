"""Deterministic, offline embedding client for tests (Phase 6C).

:class:`HashEmbeddingClient` derives each vector purely from the SHA-256 of the
input text, so the same text always yields the same vector -- across instances,
processes, and ``PYTHONHASHSEED`` values (it never uses Python's salted ``hash()``).
Vectors are L2-normalized, so the in-memory index's dot-product equals cosine
similarity. This is the embedding analog of the Stage ``*FakeClient`` doubles: it
makes the whole ingestion path runnable with no network and no API key.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from hashlib import sha256

from app.vector.interfaces import EmbeddingVector

__all__ = ["HashEmbeddingClient"]


class HashEmbeddingClient:
    """A deterministic, content-seeded embedding client (test double)."""

    def __init__(self, *, dim: int = 16, model_id: str = "hash-fake-v1") -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed_one(self, text: str) -> EmbeddingVector:
        seed = int(sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        raw = [rng.uniform(-1.0, 1.0) for _ in range(self._dim)]
        norm = math.sqrt(sum(component * component for component in raw))
        if norm == 0.0:
            # Astronomically unlikely; fall back to a deterministic unit basis vector.
            values = tuple(1.0 if i == 0 else 0.0 for i in range(self._dim))
        else:
            values = tuple(component / norm for component in raw)
        return EmbeddingVector(values=values, dim=self._dim, model_id=self._model_id)

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed a batch; output length and order match the input exactly."""

        return [self.embed_one(text) for text in texts]
