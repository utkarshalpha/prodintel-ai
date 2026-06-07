"""Phase 7B -- DecisionFrameworkCitation provenance edge: FK behavior + ORM.

Exercises the edge against in-memory SQLite (FK pragma on, mirroring Postgres):
cascade-delete from the decision side, RESTRICT protection of a cited chunk,
multiple citations per decision, the reserved enum defaults, and the ORM
relationship. Pure persistence -- no service, retrieval, or Stage-4 flow.

A decision grounds on chunks that already exist (ingested before any decision cites
them), so the corpus is seeded and committed first, then decisions + citations.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.ai_contracts.enums import EvidenceType, Relationship
from app.models.decision import Decision, DecisionFrameworkCitation
from app.models.knowledge import KnowledgeChunk, KnowledgeSource

from tests.app_helpers import make_engine, make_session_factory


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _decision() -> Decision:
    return Decision(
        subject_type="feature",
        subject_id=uuid.uuid4(),
        recommendation="build_now",
        title="Build it",
        rationale="Grounded in the RICE framework.",
        priority_rank=1,
        confidence_score=0.8,
        confidence={"score": 0.8, "components": {}, "basis": "strong"},
        model_meta={"model_id": "m", "prompt_version": "v", "input_tokens": 1, "output_tokens": 1, "stop_reason": "end"},
    )


def _seed_chunks(session, n: int = 1):
    """Persist a source + n chunks (committed), returning the chunk ids."""

    source = KnowledgeSource(
        source_type="framework",
        framework="RICE",
        title="RICE Reference",
        corpus_version="pm-corpus-v1",
        embedding_model_id="hash-fake-v1",
        content_hash="0" * 64,
    )
    chunks = [
        KnowledgeChunk(
            corpus_version=source.corpus_version,
            embedding_model_id=source.embedding_model_id,
            ordinal=i,
            content=f"framework passage {i}",
            content_hash=f"{i:064d}",
            token_count=5,
        )
        for i in range(n)
    ]
    source.chunks.extend(chunks)
    session.add(source)
    session.commit()
    return source.id, [chunk.id for chunk in chunks]


def _ground(session, decision: Decision, chunk_ids, *, score=None):
    for chunk_id in chunk_ids:
        decision.framework_citations.append(
            DecisionFrameworkCitation(chunk_id=chunk_id, retrieval_score=score)
        )
    session.add(decision)
    session.commit()


def test_defaults_are_grounds_and_framework_citation(session) -> None:
    _, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    _ground(session, decision, [chunk_id], score=0.91)

    session.expire_all()
    citation = session.get(Decision, decision.id).framework_citations[0]
    assert citation.relationship_type is Relationship.GROUNDS
    assert citation.evidence_type is EvidenceType.FRAMEWORK_CITATION
    assert citation.retrieval_score == 0.91
    assert citation.chunk_id == chunk_id


def test_orm_relationship_round_trips(session) -> None:
    _, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    _ground(session, decision, [chunk_id])

    session.expire_all()
    citation = session.get(Decision, decision.id).framework_citations[0]
    assert citation.decision.id == decision.id  # back-reference


def test_multiple_citations_per_decision(session) -> None:
    _, chunk_ids = _seed_chunks(session, 3)
    decision = _decision()
    _ground(session, decision, chunk_ids)

    session.expire_all()
    reloaded = session.get(Decision, decision.id)
    assert {c.chunk_id for c in reloaded.framework_citations} == set(chunk_ids)
    assert len(reloaded.framework_citations) == 3


def test_deleting_decision_cascades_to_citations(session) -> None:
    _, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    _ground(session, decision, [chunk_id])
    decision_id = decision.id

    session.delete(decision)
    session.commit()

    assert session.get(Decision, decision_id) is None
    assert session.get(KnowledgeChunk, chunk_id) is not None  # cited chunk survives
    assert session.query(DecisionFrameworkCitation).filter_by(decision_id=decision_id).all() == []


def test_cited_chunk_cannot_be_deleted(session) -> None:
    _, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    _ground(session, decision, [chunk_id])

    chunk = session.get(KnowledgeChunk, chunk_id)
    session.delete(chunk)
    with pytest.raises(IntegrityError):
        session.commit()  # RESTRICT: a grounded-on chunk cannot be deleted


def test_cited_chunks_source_cannot_cascade_delete(session) -> None:
    """Deleting the source would cascade to the cited chunk, which RESTRICT blocks."""

    source_id, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    _ground(session, decision, [chunk_id])

    source = session.get(KnowledgeSource, source_id)
    session.delete(source)
    with pytest.raises(IntegrityError):
        session.commit()
