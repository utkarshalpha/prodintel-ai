"""Phase 6 -- Knowledge-corpus ingestion contracts.

These are the *input* shape for ingesting an external knowledge source (a PM
framework reference, a book excerpt, a historical decision record, a company
strategy memo) into the RAG corpus. Unlike the Stage 1-4 contracts, they are
**not** emitted by an LLM -- they are operator/pipeline-supplied -- so they extend
:class:`FrozenModel` (strict + immutable) rather than :class:`AIContractBase`:
there is no model call behind them, hence no ``model_meta`` / ``schema_version``.

Two deliberate omissions, both because the value is *computed, never asserted*
(the same discipline as :class:`ConfidenceBlock`'s derived basis):

* ``content_hash`` -- the SHA-256 of chunk text and the source roll-up are computed
  by the ingestion service, so a caller can never supply a hash that disagrees with
  the bytes.
* ``chroma_id`` -- the vector handle is assigned after embedding, not at ingest time.

Only *structural* rules live here (non-empty text, bounded lengths, >= 1 chunk).
*Semantic* rules -- ordinal contiguity, per-source dedup, embedding-model pinning --
belong to the ingestion service/validator (a later Phase 6 sub-phase), exactly as
grounding/provenance/integrity semantics live outside the Stage contracts.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from app.ai_contracts.base import FrozenModel
from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType

__all__ = ["KnowledgeChunkContract", "KnowledgeSourceContract"]


class KnowledgeChunkContract(FrozenModel):
    """One retrievable span of a knowledge source, prior to hashing/embedding.

    ``ordinal`` gives a deterministic order within the source; ``token_count`` is
    the chunk's measured token length (used downstream for batching/cost). The
    optional ``framework`` lets a single mixed source attribute individual chunks
    (e.g. a methodology book whose chapters cover different frameworks).
    """

    ordinal: Annotated[int, Field(ge=0)] = Field(
        ...,
        description="Position of this chunk within its source (0-based, deterministic order key).",
    )
    content: str = Field(..., min_length=1, description="Canonical chunk text (the system-of-record copy).")
    section: str | None = Field(
        default=None,
        max_length=255,
        description="Optional heading/section path this chunk was extracted from.",
    )
    token_count: Annotated[int, Field(ge=1)] = Field(
        ...,
        description="Measured token length of `content` (>= 1 for non-empty text).",
    )
    framework: FrameworkName | None = Field(
        default=None,
        description="Optional per-chunk framework attribution; None when not framework-specific.",
    )


class KnowledgeSourceContract(FrozenModel):
    """A complete knowledge source to ingest: its descriptive metadata plus chunks.

    ``corpus_version`` labels the corpus snapshot and ``embedding_model_id`` pins the
    embedding model; both are carried so any future retrieval is reproducible
    (which snapshot, which embedder). At least one chunk is required -- a source with
    nothing to retrieve is not a meaningful ingest.
    """

    source_type: KnowledgeSourceType = Field(
        ..., description="Category of source: framework | book | historical_decision | company_strategy."
    )
    framework: FrameworkName | None = Field(
        default=None,
        description="Specific framework this source is attributed to, when applicable.",
    )
    title: str = Field(..., min_length=1, max_length=255, description="Human-readable source title.")
    author: str | None = Field(default=None, max_length=255, description="Optional author/origin.")
    source_license: str | None = Field(
        default=None, max_length=128, description="Optional license/usage terms for the source."
    )
    uri: str | None = Field(default=None, max_length=2048, description="Optional canonical URI of the source.")
    corpus_version: str = Field(
        ..., min_length=1, max_length=64, description="Corpus snapshot label (e.g. 'pm-corpus-v3')."
    )
    embedding_model_id: str = Field(
        ..., min_length=1, max_length=128, description="Identifier of the embedding model this source is pinned to."
    )
    chunks: list[KnowledgeChunkContract] = Field(
        ..., min_length=1, description="The source's retrievable chunks (at least one)."
    )
