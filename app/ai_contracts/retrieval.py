"""Phase 6D -- Retrieval-layer contracts.

The read-path counterpart to the knowledge-ingestion contracts: a
:class:`RetrievalQuery` describes a framework-aware lookup against one corpus
snapshot, and a :class:`RetrievalResult` returns ranked, evidence-traceable
:class:`RetrievalCitation`s plus a :class:`RetrievalMeta` accounting block.

These are operator/service DTOs (not LLM-emitted), so they extend
:class:`FrozenModel` (strict + immutable) rather than :class:`AIContractBase`; only
the result root carries a pinned ``schema_version``.

The citation's ``content`` is the **authoritative** chunk text from the relational
store (the vector index holds no text), so a cited passage always traces back to the
system of record -- the same discipline as Stage 1 grounding and the ``/why`` walk.
``chunk_id``/``source_id`` are the stable handles a future decision would ground
against (the reserved ``FRAMEWORK_CITATION`` / ``GROUNDS`` provenance edge).
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.ai_contracts.base import FrozenModel
from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType

__all__ = [
    "MAX_TOP_K",
    "DEFAULT_TOP_K",
    "RetrievalQuery",
    "RetrievalCitation",
    "RetrievalMeta",
    "RetrievalResult",
]

#: Upper bound on requested results, so a query can never ask for an unbounded scan.
MAX_TOP_K = 50
DEFAULT_TOP_K = 5


class RetrievalQuery(FrozenModel):
    """A framework-aware retrieval request, pinned to one corpus snapshot."""

    text: str = Field(..., min_length=1, description="The natural-language query to embed and match.")
    corpus_version: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Corpus snapshot to search (required: pins the snapshot and the embedding model).",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description=f"Maximum results to return (1..{MAX_TOP_K}).",
    )
    framework: FrameworkName | None = Field(
        default=None, description="Optional framework filter (framework-aware retrieval)."
    )
    source_type: KnowledgeSourceType | None = Field(
        default=None, description="Optional source-kind filter."
    )
    min_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional cosine-similarity floor in [-1, 1]; matches below it are dropped.",
    )


class RetrievalCitation(FrozenModel):
    """One ranked hit: a retrieved chunk plus its source attribution.

    ``content`` is read from the relational chunk row (the system of record), never
    from the vector store. ``score`` is cosine similarity (``1 - distance``); higher
    is more similar.
    """

    chunk_id: UUID = Field(..., description="Id of the matched knowledge chunk.")
    source_id: UUID = Field(..., description="Id of the chunk's knowledge source.")
    score: float = Field(..., description="Cosine similarity (1 - distance); higher is more similar.")
    content: str = Field(..., min_length=1, description="Authoritative chunk text (from Postgres).")
    ordinal: int = Field(..., ge=0, description="Chunk position within its source.")
    section: str | None = Field(default=None, max_length=255, description="Heading/section path, if any.")
    framework: FrameworkName | None = Field(default=None, description="Per-chunk framework attribution.")
    corpus_version: str = Field(..., min_length=1, max_length=64)
    source_title: str = Field(..., min_length=1, max_length=255)
    source_type: KnowledgeSourceType = Field(..., description="Kind of source the chunk came from.")
    source_uri: str | None = Field(default=None, max_length=2048)
    source_license: str | None = Field(default=None, max_length=128)


class RetrievalMeta(FrozenModel):
    """Accounting for a retrieval run: how the vector matches resolved to citations.

    Every vector match is exactly one of: a returned citation, dropped because no
    relational row resolved (stale), or dropped by the ``min_score`` floor. The
    :class:`RetrievalResult` validator enforces that partition.
    """

    embedding_model_id: str = Field(..., min_length=1, max_length=128)
    top_k: int = Field(..., ge=1, le=MAX_TOP_K)
    vector_match_count: int = Field(..., ge=0, description="Matches the vector index returned.")
    resolved_count: int = Field(..., ge=0, description="Matches that resolved to a relational row.")
    dropped_stale_count: int = Field(..., ge=0, description="Matches with no relational row (stale vectors).")
    dropped_score_count: int = Field(..., ge=0, description="Resolved matches dropped by the min_score floor.")


class RetrievalResult(FrozenModel):
    """Ranked, evidence-traceable retrieval output for one query."""

    schema_version: Literal["1.0"] = Field(default="1.0", description="Pinned contract version.")
    query_text: str = Field(..., min_length=1)
    corpus_version: str = Field(..., min_length=1, max_length=64)
    citations: list[RetrievalCitation] = Field(
        default_factory=list, description="Ranked citations (may be empty)."
    )
    meta: RetrievalMeta = Field(..., description="Run accounting.")

    @model_validator(mode="after")
    def _check_count_partition(self) -> "RetrievalResult":
        """Every vector match is a citation, a stale drop, or a score drop -- no leaks."""

        meta = self.meta
        if meta.resolved_count != meta.vector_match_count - meta.dropped_stale_count:
            raise ValueError(
                "resolved_count must equal vector_match_count - dropped_stale_count"
            )
        if len(self.citations) != meta.resolved_count - meta.dropped_score_count:
            raise ValueError(
                "len(citations) must equal resolved_count - dropped_score_count"
            )
        return self
