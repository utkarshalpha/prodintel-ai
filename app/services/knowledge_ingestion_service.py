"""KnowledgeIngestionService -- ingest external knowledge into the RAG corpus (6C-2).

Turns a :class:`KnowledgeSourceContract` into persisted, vector-backed
:class:`KnowledgeSource` / :class:`KnowledgeChunk` rows. It is **not** an LLM stage,
but it follows the same hardened discipline as the Stage services: it owns the
transaction boundary and never holds a DB transaction during the external
(embedding/upsert) call (ADR-010).

The orphan-safe four-phase flow
-------------------------------
1. **READ / dedupe / validate** (own txn -> commit): validate ordinals, pin-check the
   embedding model, compute hashes, look up an existing source, read its chunks.
2. **WRITE rows** with ``chroma_id = NULL`` (fresh txn -> commit): the system of
   record now exists, idempotent by ``(corpus_version, content_hash)``.
3. **EMBED + UPSERT** (no DB txn held): batch-embed chunk text and upsert vectors
   keyed by ``str(chunk.id)``.
4. **WRITE-BACK** (per-batch fresh txn -> commit): set each chunk's ``chroma_id`` to
   its own id (the deterministic vector key).

A crash anywhere between phases 2 and 4 leaves committed rows temporarily missing
their vector -- recoverable, because the vector key is deterministic, so re-upsert
overwrites rather than duplicating. It can never leave an orphan vector with no row:
the row is always committed (phase 2) before any upsert, so the worst case is a
committed row whose ``chroma_id`` is still NULL -- exactly the recoverable state that
:meth:`reembed_pending` (or a re-``ingest``) heals.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from sqlalchemy.orm import Session

from app.ai_contracts.knowledge import KnowledgeSourceContract
from app.models.knowledge import KnowledgeChunk, KnowledgeSource
from app.observability.logging import get_logger
from app.repositories.knowledge_repository import (
    KnowledgeChunkRepository,
    KnowledgeSourceRepository,
)
from app.services.errors import (
    ChunkOrdinalError,
    CorpusEmbeddingModelMismatchError,
    KnowledgeIngestionFailedError,
    KnowledgeSourceNotFoundError,
)
from app.vector.errors import EmbeddingClientError, EmbeddingDimensionMismatchError, VectorIndexError
from app.vector.interfaces import EmbeddingClient, VectorIndex, VectorRecord

__all__ = ["KnowledgeIngestionService", "IngestionResult"]

_logger = get_logger(__name__)
_DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of an ingest/resume run.

    ``chunk_count`` is the source's total chunks; ``embedded_count`` is how many
    chunks this call newly embedded (0 for a skip).
    """

    created: bool
    source_id: uuid.UUID
    chunk_count: int
    embedded_count: int
    skipped: bool
    reason: str | None


class KnowledgeIngestionService:
    """Application service for corpus ingestion and embedding recovery."""

    def __init__(
        self,
        session: Session,
        source_repo: KnowledgeSourceRepository,
        chunk_repo: KnowledgeChunkRepository,
        embedding_client: EmbeddingClient,
        vector_index: VectorIndex,
        *,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._session = session
        self._sources = source_repo
        self._chunks = chunk_repo
        self._embedder = embedding_client
        self._index = vector_index
        self._batch_size = batch_size

    # ------------------------------------------------------------------- ingest
    def ingest(
        self, contract: KnowledgeSourceContract, *, workspace_id: uuid.UUID | None = None
    ) -> IngestionResult:
        """Ingest a knowledge source: dedupe, persist, embed -- idempotent & resumable."""

        # ---- Pure validation + hashing (no DB) ------------------------------
        self._validate_ordinals(contract)
        self._check_declared_model(contract)
        self._check_dimensions()
        ordered = sorted(contract.chunks, key=lambda chunk: chunk.ordinal)
        chunk_hashes = [self._chunk_hash(chunk.content) for chunk in ordered]
        source_hash = self._rollup_hash(
            (chunk.ordinal, digest) for chunk, digest in zip(ordered, chunk_hashes)
        )

        # ---- Phase 1: READ / dedupe (one read txn -> commit) ----------------
        self._check_corpus_model_invariant(contract.corpus_version)
        existing = self._sources.get_by_corpus_version_and_content_hash(
            contract.corpus_version, source_hash
        )
        all_chunks: list[KnowledgeChunk] = []
        pending: list[KnowledgeChunk] = []
        if existing is not None:
            all_chunks = list(self._chunks.list_for_source(existing.id))
            pending = [chunk for chunk in all_chunks if chunk.chroma_id is None]
        self._session.commit()  # release the connection before any embedding (ADR-010)

        if existing is not None:
            if not pending:
                _logger.info(
                    "knowledge_ingest_skipped",
                    extra={"event": "ingest_skipped", "source_id": str(existing.id)},
                )
                return IngestionResult(
                    created=False,
                    source_id=existing.id,
                    chunk_count=len(all_chunks),
                    embedded_count=0,
                    skipped=True,
                    reason="already ingested",
                )
            embedded = self._embed_and_writeback(pending)
            return IngestionResult(
                created=False,
                source_id=existing.id,
                chunk_count=len(all_chunks),
                embedded_count=embedded,
                skipped=False,
                reason="resumed prior ingest",
            )

        # ---- Phase 2: WRITE rows (chroma_id NULL) (fresh txn -> commit) ------
        source = KnowledgeSource(
            workspace_id=workspace_id,
            source_type=contract.source_type,
            framework=contract.framework,
            title=contract.title,
            author=contract.author,
            source_license=contract.source_license,
            uri=contract.uri,
            corpus_version=contract.corpus_version,
            embedding_model_id=self._embedder.model_id,
            content_hash=source_hash,
        )
        chunks = [
            KnowledgeChunk(
                corpus_version=contract.corpus_version,
                embedding_model_id=self._embedder.model_id,
                framework=chunk.framework,
                section=chunk.section,
                ordinal=chunk.ordinal,
                content=chunk.content,
                content_hash=digest,
                token_count=chunk.token_count,
                chroma_id=None,
            )
            for chunk, digest in zip(ordered, chunk_hashes)
        ]
        self._sources.create_with_chunks(source, chunks)
        self._session.commit()
        source_id = source.id
        persisted = list(source.chunks)
        _logger.info(
            "knowledge_db_write",
            extra={"event": "db_write", "entity": "knowledge_source", "chunk_count": len(persisted)},
        )

        # ---- Phase 3 + 4: embed + upsert + write-back -----------------------
        embedded = self._embed_and_writeback(persisted)
        _logger.info(
            "knowledge_ingest_succeeded",
            extra={"event": "ingest_succeeded", "source_id": str(source_id), "embedded": embedded},
        )
        return IngestionResult(
            created=True,
            source_id=source_id,
            chunk_count=len(persisted),
            embedded_count=embedded,
            skipped=False,
            reason=None,
        )

    # ----------------------------------------------------------- reembed_pending
    def reembed_pending(self, source_id: uuid.UUID) -> IngestionResult:
        """Embed + upsert + write-back the source's chunks whose ``chroma_id`` is NULL.

        Ops/recovery entry point. Raises :class:`KnowledgeSourceNotFoundError` if the
        source is absent. Idempotent: a fully-embedded source is a no-op skip.
        """

        source = self._sources.get(source_id)
        if source is None:
            raise KnowledgeSourceNotFoundError(source_id)
        all_chunks = list(self._chunks.list_for_source(source_id))
        pending = [chunk for chunk in all_chunks if chunk.chroma_id is None]
        self._session.commit()  # release the connection before embedding (ADR-010)

        embedded = self._embed_and_writeback(pending) if pending else 0
        return IngestionResult(
            created=False,
            source_id=source_id,
            chunk_count=len(all_chunks),
            embedded_count=embedded,
            skipped=not pending,
            reason="reembed_pending",
        )

    # ------------------------------------------------------------- embed/write
    def _embed_and_writeback(self, chunks: list[KnowledgeChunk]) -> int:
        """Per-batch: embed -> upsert -> write-back -> commit. No txn held during embed."""

        embedded = 0
        for batch in _batched(chunks, self._batch_size):
            try:
                vectors = self._embedder.embed([chunk.content for chunk in batch])
                records = [self._record(chunk, vector) for chunk, vector in zip(batch, vectors)]
                self._index.upsert(records)
            except (EmbeddingClientError, VectorIndexError) as exc:
                # Rows already committed with chroma_id NULL -> resumable.
                _logger.warning(
                    "knowledge_ingest_failed",
                    extra={"event": "ingest_failed", "reason": str(exc)},
                )
                raise KnowledgeIngestionFailedError(str(exc)) from exc

            # Write-back is the only mutation. On failure, discard so chroma_id stays
            # NULL and expire in-memory state, so a resume re-reads the rolled-back rows
            # (a flushed-but-uncommitted chroma_id must not linger in the identity map).
            try:
                updated = self._chunks.set_chroma_ids({chunk.id: chunk.id for chunk in batch})
                self._session.commit()
            except Exception:
                self._session.rollback()
                self._session.expire_all()
                raise
            if updated != len(batch):
                # A chunk vanished between read and write-back (e.g. its source was
                # deleted): surface it rather than silently inflating embedded_count.
                _logger.warning(
                    "knowledge_writeback_short",
                    extra={"event": "writeback_short", "expected": len(batch), "updated": updated},
                )
            embedded += updated
        return embedded

    def _record(self, chunk: KnowledgeChunk, vector) -> VectorRecord:
        metadata: dict[str, str | int | float | bool] = {
            "corpus_version": chunk.corpus_version,
            "source_id": str(chunk.source_id),
            "ordinal": chunk.ordinal,
        }
        if chunk.framework is not None:
            metadata["framework"] = chunk.framework.value
        return VectorRecord(chunk_id=str(chunk.id), vector=vector, metadata=metadata)

    # -------------------------------------------------------------- validation
    @staticmethod
    def _validate_ordinals(contract: KnowledgeSourceContract) -> None:
        ordinals = sorted(chunk.ordinal for chunk in contract.chunks)
        if ordinals != list(range(len(contract.chunks))):
            raise ChunkOrdinalError(
                f"chunk ordinals must be a contiguous 0-based sequence; got {ordinals}"
            )

    def _check_declared_model(self, contract: KnowledgeSourceContract) -> None:
        """The contract's declared model must match the wired embedding client."""

        actual = self._embedder.model_id
        if contract.embedding_model_id != actual:
            raise CorpusEmbeddingModelMismatchError(
                contract.corpus_version, actual, contract.embedding_model_id
            )

    def _check_dimensions(self) -> None:
        """Pre-flight: the embedder's dim must match the index's configured dim.

        The index dim is read defensively -- the vector Protocol does not declare it,
        but the concrete fakes/adapters expose it. The index also validates dim at
        upsert as a backstop.
        """

        index_dim = getattr(self._index, "dim", None)
        if index_dim is not None and self._embedder.dim != index_dim:
            raise EmbeddingDimensionMismatchError(
                f"embedding client dim={self._embedder.dim} != index dim={index_dim}"
            )

    def _check_corpus_model_invariant(self, corpus_version: str) -> None:
        """A corpus_version is single-model: any existing source must share the model."""

        actual = self._embedder.model_id
        existing = self._sources.list(corpus_version=corpus_version, limit=1)
        if existing and existing[0].embedding_model_id != actual:
            raise CorpusEmbeddingModelMismatchError(
                corpus_version, existing[0].embedding_model_id, actual
            )

    # ------------------------------------------------------------------ hashing
    @staticmethod
    def _chunk_hash(content: str) -> str:
        """SHA-256 of the chunk's canonical text (the SignalService idiom)."""

        return sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _rollup_hash(ordered_pairs) -> str:
        """Merkle-style roll-up over ``(ordinal, chunk_hash)`` pairs (order-sensitive)."""

        payload = "\n".join(f"{ordinal}:{digest}" for ordinal, digest in ordered_pairs)
        return sha256(payload.encode("utf-8")).hexdigest()


def _batched(items: list, size: int):
    """Yield consecutive slices of ``items`` of length ``size``."""

    for start in range(0, len(items), size):
        yield items[start : start + size]
