"""RetrievalService -- framework-aware retrieval over the knowledge corpus (6D-iii).

Read-only. Embeds a query, filters and ranks against the vector index, resolves the
top matches to authoritative chunk rows (text from Postgres, never the vector store),
and returns evidence-traceable citations plus a count-partitioned accounting block.

Discipline (mirrors the Stage services):

* **Model-consistency guard** -- the corpus's ``embedding_model_id`` must equal the
  wired embedding client's ``model_id``; otherwise the query would be embedded by a
  different model than the corpus and rank against incomparable vectors. Mismatch
  raises :class:`CorpusEmbeddingModelMismatchError` *before* embedding.
* **ADR-010** -- the guard read is committed (connection released) *before* the embed
  and vector-query network calls; the only other DB touch is the final batch resolve,
  which runs *after* the network calls. No transaction is ever held across network I/O.
* **Count partition** -- every vector match is exactly one of: a returned citation, a
  stale drop (no relational row / malformed id), or a ``min_score`` drop.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.ai_contracts.retrieval import (
    RetrievalCitation,
    RetrievalMeta,
    RetrievalQuery,
    RetrievalResult,
)
from app.models.knowledge import KnowledgeChunk
from app.observability.logging import get_logger
from app.repositories.knowledge_repository import (
    KnowledgeChunkRepository,
    KnowledgeSourceRepository,
)
from app.services.errors import CorpusEmbeddingModelMismatchError, RetrievalFailedError
from app.vector.errors import EmbeddingClientError, VectorIndexError
from app.vector.interfaces import EmbeddingClient, VectorFilter, VectorIndex, VectorMatch

__all__ = ["RetrievalService"]

_logger = get_logger(__name__)


class RetrievalService:
    """Application service for corpus retrieval (read-only)."""

    def __init__(
        self,
        session: Session,
        embedding_client: EmbeddingClient,
        vector_index: VectorIndex,
        chunk_repo: KnowledgeChunkRepository,
        source_repo: KnowledgeSourceRepository,
    ) -> None:
        self._session = session
        self._embedder = embedding_client
        self._index = vector_index
        self._chunks = chunk_repo
        self._sources = source_repo

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run a framework-aware retrieval and return ranked, traceable citations."""

        # ---- Phase 1: model-consistency guard (DB read) -> commit (ADR-010) ----
        sources = self._sources.list(corpus_version=query.corpus_version, limit=1)
        self._session.commit()  # release the connection before any network call
        if not sources:
            # Unknown / empty corpus: nothing to retrieve, model cannot be checked.
            return self._empty_result(query)
        corpus_model = sources[0].embedding_model_id
        if corpus_model != self._embedder.model_id:
            raise CorpusEmbeddingModelMismatchError(
                query.corpus_version, corpus_model, self._embedder.model_id
            )

        # ---- Phase 2: embed + vector search (NO DB transaction held) -----------
        try:
            query_vector = self._embedder.embed_one(query.text)
        except EmbeddingClientError as exc:
            raise RetrievalFailedError(f"query embedding failed: {exc}") from exc
        try:
            raw_matches = list(self._index.query(query_vector, top_k=query.top_k, where=self._build_filter(query)))
        except VectorIndexError as exc:
            raise RetrievalFailedError(f"vector query failed: {exc}") from exc

        # The index contract returns distinct chunk_ids; normalize defensively (keep the
        # first/best-ranked occurrence) so a misbehaving backend that returns duplicates
        # cannot double-count resolved chunks or emit duplicate citations -- the count
        # partition stays semantically sound, not just arithmetically.
        matches = _dedupe_by_chunk_id(raw_matches)

        # ---- Phase 3: resolve rows (DB read) + partition in ranking order ------
        parsed = [(match, self._to_uuid(match.chunk_id)) for match in matches]
        resolvable_ids = [chunk_id for _, chunk_id in parsed if chunk_id is not None]
        rows = {row.id: row for row in self._chunks.get_many_with_source(resolvable_ids)}

        citations: list[RetrievalCitation] = []
        resolved_count = 0
        dropped_stale_count = 0
        dropped_score_count = 0
        for match, chunk_id in parsed:  # iterate in vector-ranking order
            row = rows.get(chunk_id) if chunk_id is not None else None
            if row is None:
                dropped_stale_count += 1  # malformed id or no relational row
                continue
            resolved_count += 1
            if query.min_score is not None and match.score < query.min_score:
                dropped_score_count += 1
                continue
            citations.append(self._citation(match, row))

        meta = RetrievalMeta(
            embedding_model_id=self._embedder.model_id,
            top_k=query.top_k,
            vector_match_count=len(matches),
            resolved_count=resolved_count,
            dropped_stale_count=dropped_stale_count,
            dropped_score_count=dropped_score_count,
        )
        _logger.info(
            "retrieval_succeeded",
            extra={
                "event": "retrieval_succeeded",
                "corpus_version": query.corpus_version,
                "vector_match_count": meta.vector_match_count,
                "citation_count": len(citations),
            },
        )
        return RetrievalResult(
            query_text=query.text,
            corpus_version=query.corpus_version,
            citations=citations,
            meta=meta,
        )

    # ------------------------------------------------------------------ helpers
    def _build_filter(self, query: RetrievalQuery) -> VectorFilter:
        """Metadata filter: corpus_version (always) + framework / source_type if set.

        Note: ``source_type`` is honored only once ingestion writes it into the vector
        metadata (the current 6C-2 metadata omits it); until then a source_type filter
        legitimately yields no matches rather than wrong ones.
        """

        equals: dict[str, str | int | float | bool] = {"corpus_version": query.corpus_version}
        if query.framework is not None:
            equals["framework"] = query.framework.value
        if query.source_type is not None:
            equals["source_type"] = query.source_type.value
        return VectorFilter(equals=equals)

    def _citation(self, match: VectorMatch, chunk: KnowledgeChunk) -> RetrievalCitation:
        source = chunk.source  # eagerly loaded by get_many_with_source
        return RetrievalCitation(
            chunk_id=chunk.id,
            source_id=chunk.source_id,
            score=match.score,
            content=chunk.content,
            ordinal=chunk.ordinal,
            section=chunk.section,
            framework=chunk.framework,
            corpus_version=chunk.corpus_version,
            source_title=source.title,
            source_type=source.source_type,
            source_uri=source.uri,
            source_license=source.source_license,
        )

    def _empty_result(self, query: RetrievalQuery) -> RetrievalResult:
        meta = RetrievalMeta(
            embedding_model_id=self._embedder.model_id,
            top_k=query.top_k,
            vector_match_count=0,
            resolved_count=0,
            dropped_stale_count=0,
            dropped_score_count=0,
        )
        return RetrievalResult(
            query_text=query.text, corpus_version=query.corpus_version, citations=[], meta=meta
        )

    @staticmethod
    def _to_uuid(value: str) -> uuid.UUID | None:
        try:
            return uuid.UUID(str(value))
        except (ValueError, TypeError, AttributeError):
            return None


def _dedupe_by_chunk_id(matches: list[VectorMatch]) -> list[VectorMatch]:
    """Keep the first occurrence of each chunk_id, preserving vector-ranking order."""

    seen: set[str] = set()
    unique: list[VectorMatch] = []
    for match in matches:
        if match.chunk_id in seen:
            continue
        seen.add(match.chunk_id)
        unique.append(match)
    return unique
