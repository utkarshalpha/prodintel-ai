"""Vector infrastructure: provider-neutral embedding/index seam (Phase 6C).

Exposes the two Protocols and their DTOs, the deterministic offline fakes, and the
error taxonomy. The ChromaDB adapters are added in a later sub-phase (6C-3) and are
imported lazily there, so importing this package never pulls in ``chromadb``.
"""

from app.vector.embedding_fake import HashEmbeddingClient
from app.vector.errors import (
    EmbeddingClientError,
    EmbeddingDimensionMismatchError,
    VectorError,
    VectorIndexError,
)
from app.vector.index_fake import InMemoryVectorIndex
from app.vector.interfaces import (
    EmbeddingClient,
    EmbeddingVector,
    VectorFilter,
    VectorIndex,
    VectorMatch,
    VectorRecord,
)

__all__ = [
    "EmbeddingClient",
    "VectorIndex",
    "EmbeddingVector",
    "VectorRecord",
    "VectorMatch",
    "VectorFilter",
    "HashEmbeddingClient",
    "InMemoryVectorIndex",
    "VectorError",
    "EmbeddingClientError",
    "VectorIndexError",
    "EmbeddingDimensionMismatchError",
]
