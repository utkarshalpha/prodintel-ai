"""Phase 7C-iii -- persistence of DecisionFrameworkCitation grounding edges.

Two layers:

* repository -- ``DecisionRepository.create_with_links`` now appends framework edges from
  the ``(chunk_id, retrieval_score)`` pairs the service resolved against the frozen pool;
* service -- ``DecisionService`` dedupes ``framework_citation_ids`` (``dict.fromkeys``),
  validates every id is a pool member (else :class:`FrameworkCitationPoolError`), and
  persists the pool's ``retrieval_score`` -- end to end, the model grounds a decision on a
  framework chunk and the edge lands with the right provenance defaults.

The frozen pool from 7C-ii is the single source of truth; a cited chunk must already exist
as a KnowledgeChunk row (the FK is ON DELETE RESTRICT), so the corpus is seeded first.
"""

from __future__ import annotations

import re
import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.ai_contracts.enums import (
    EvidenceType,
    FrameworkName,
    KnowledgeSourceType,
    Relationship,
    StakeholderType,
)
from app.ai_contracts.retrieval import RetrievalCitation, RetrievalMeta, RetrievalResult
from app.models.decision import Decision, DecisionFrameworkCitation
from app.models.knowledge import KnowledgeChunk, KnowledgeSource
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_service import DecisionService, FrameworkCitationPoolError
from app.services.feature_service import FeatureService
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage4.runner import build_stage4_runner

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage4FakeClient,
    make_engine,
    make_session_factory,
)

CORPUS = "pm-corpus-v1"
_CHUNK_ID_RE = re.compile(r"chunk_id:\s*([0-9a-fA-F-]{36})")


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- builders
def _seed_chunks(session, n: int = 1):
    """Persist a source + n chunks (committed); returns (source_id, [chunk_ids])."""

    source = KnowledgeSource(
        source_type="framework", framework="RICE", title="RICE Reference",
        corpus_version=CORPUS, embedding_model_id="hash-fake-v1", content_hash="0" * 64,
    )
    chunks = [
        KnowledgeChunk(
            corpus_version=CORPUS, embedding_model_id="hash-fake-v1", ordinal=i,
            content=f"framework passage {i}", content_hash=f"{i:064d}", token_count=5,
        )
        for i in range(n)
    ]
    source.chunks.extend(chunks)
    session.add(source)
    session.commit()
    return source.id, [chunk.id for chunk in chunks]


def _decision() -> Decision:
    return Decision(
        subject_type="feature", subject_id=uuid.uuid4(), recommendation="build_now",
        title="Build it", rationale="Grounded in the RICE framework.", priority_rank=1,
        confidence_score=0.8, confidence={"score": 0.8, "components": {}, "basis": "strong"},
        model_meta={"model_id": "m", "prompt_version": "v", "input_tokens": 1, "output_tokens": 1, "stop_reason": "end"},
    )


def _cit(chunk_id, score):
    return RetrievalCitation(
        chunk_id=chunk_id, source_id=uuid.uuid4(), score=score, content="passage", ordinal=0,
        framework=FrameworkName.RICE, corpus_version=CORPUS, source_title="RICE Ref",
        source_type=KnowledgeSourceType.FRAMEWORK,
    )


def _result(query, citations):
    n = len(citations)
    return RetrievalResult(
        query_text=query.text, corpus_version=query.corpus_version, citations=list(citations),
        meta=RetrievalMeta(
            embedding_model_id="hash-fake-v1", top_k=query.top_k, vector_match_count=n,
            resolved_count=n, dropped_stale_count=0, dropped_score_count=0,
        ),
    )


class FakeRetrievalService:
    def __init__(self, *, default=None):
        self.default = default or []

    def retrieve(self, query):
        return _result(query, self.default)


class FrameworkCitingClient:
    """Stage 4 fake that grounds every decision on the framework chunk_ids in the prompt."""

    def __init__(self, *, duplicate: bool = False):
        self._inner = Stage4FakeClient()
        self._duplicate = duplicate

    def complete(self, *, system, messages, tool):
        resp = self._inner.complete(system=system, messages=messages, tool=tool)
        chunk_ids = _CHUNK_ID_RE.findall(messages[0].content)
        if chunk_ids:
            cited = [chunk_ids[0], chunk_ids[0]] if self._duplicate else chunk_ids
            for decision in resp.tool_input["decisions"]:
                decision["framework_citation_ids"] = list(cited)
        return resp


def _service(session, retrieval, *, client=None) -> DecisionService:
    return DecisionService(
        session, DecisionRepository(session), ConflictRepository(session),
        FeatureRepository(session), ParsedSignalRepository(session),
        build_stage4_runner(client or Stage4FakeClient()),
        retrieval_service=retrieval,
    )


def _one_feature(session) -> uuid.UUID:
    signals = SignalService(
        session, SignalRepository(session), ParsedSignalRepository(session),
        build_stage1_runner(Stage1FakeClient(grounded=True)),
    )
    s = signals.create_signal(source_type=StakeholderType.SALES, raw_text="Enterprise SSO is critical").signal
    signals.analyze_signal(s.id)
    feature = FeatureService(
        session, FeatureRepository(session), FeatureSignalRepository(session),
        ParsedSignalRepository(session), SignalRepository(session),
        build_stage2_runner(Stage2FakeClient(mode="cluster_all")),
    ).extract_features([s.id]).features[0]
    return feature.id


# --------------------------------------------------------------------------- #
# Repository layer
# --------------------------------------------------------------------------- #
def test_repository_creates_framework_edge_with_provenance_defaults(session) -> None:
    _, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    DecisionRepository(session).create_with_links(decision, [], [], [(chunk_id, 0.91)])
    session.commit()

    session.expire_all()
    edge = session.get(Decision, decision.id).framework_citations[0]
    assert edge.chunk_id == chunk_id
    assert edge.retrieval_score == 0.91  # persisted from the pool
    assert edge.relationship_type is Relationship.GROUNDS
    assert edge.evidence_type is EvidenceType.FRAMEWORK_CITATION


def test_repository_persists_multiple_framework_citations(session) -> None:
    _, chunk_ids = _seed_chunks(session, 3)
    decision = _decision()
    DecisionRepository(session).create_with_links(
        decision, [], [], [(cid, 0.5) for cid in chunk_ids]
    )
    session.commit()

    session.expire_all()
    reloaded = session.get(Decision, decision.id)
    assert {e.chunk_id for e in reloaded.framework_citations} == set(chunk_ids)
    assert len(reloaded.framework_citations) == 3


def test_repository_without_framework_citations_is_unchanged(session) -> None:
    """The 3-arg call (no framework) persists a decision with zero framework edges."""

    decision = _decision()
    DecisionRepository(session).create_with_links(decision, [], [])  # backward-compatible call
    session.commit()

    session.expire_all()
    assert session.get(Decision, decision.id).framework_citations == []


def test_citing_a_nonexistent_chunk_is_rejected_by_fk(session) -> None:
    """An edge to a chunk_id with no knowledge_chunk row violates the FK (no orphan edge)."""

    decision = _decision()
    with pytest.raises(IntegrityError):
        # uuid4() is not a real chunk: the ON DELETE RESTRICT FK rejects it at flush.
        DecisionRepository(session).create_with_links(decision, [], [], [(uuid.uuid4(), 0.5)])


def test_decision_delete_cascades_to_framework_edges(session) -> None:
    _, [chunk_id] = _seed_chunks(session, 1)
    decision = _decision()
    DecisionRepository(session).create_with_links(decision, [], [], [(chunk_id, 0.7)])
    session.commit()
    decision_id = decision.id

    session.delete(decision)
    session.commit()

    assert session.get(Decision, decision_id) is None
    assert session.get(KnowledgeChunk, chunk_id) is not None  # cited chunk survives (RESTRICT)
    assert session.query(DecisionFrameworkCitation).filter_by(decision_id=decision_id).all() == []


# --------------------------------------------------------------------------- #
# Service resolution: dedupe / pool validation / score source
# --------------------------------------------------------------------------- #
def test_resolve_dedupes_citation_ids(session) -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    resolved = _service(session, None)._resolve_framework_citations([a, a, b, a], {a: 0.9, b: 0.5})
    assert resolved == [(a, 0.9), (b, 0.5)]  # deduped, order preserved, scores from pool


def test_resolve_rejects_id_not_in_pool(session) -> None:
    a, unknown = uuid.uuid4(), uuid.uuid4()
    with pytest.raises(FrameworkCitationPoolError):
        _service(session, None)._resolve_framework_citations([a, unknown], {a: 0.9})


def test_resolve_takes_score_from_pool(session) -> None:
    a = uuid.uuid4()
    assert _service(session, None)._resolve_framework_citations([a], {a: 0.42}) == [(a, 0.42)]


# --------------------------------------------------------------------------- #
# End-to-end through synthesize_decisions
# --------------------------------------------------------------------------- #
def test_synthesize_persists_grounded_framework_citation(session) -> None:
    feature_id = _one_feature(session)
    _, [chunk_id] = _seed_chunks(session, 1)
    fake = FakeRetrievalService(default=[_cit(chunk_id, 0.87)])

    result = _service(session, fake, client=FrameworkCitingClient()).synthesize_decisions(
        [feature_id], corpus_version=CORPUS
    )

    session.expire_all()
    reloaded = session.get(Decision, result.decisions[0].id)
    assert len(reloaded.framework_citations) == 1
    edge = reloaded.framework_citations[0]
    assert edge.chunk_id == chunk_id
    assert edge.retrieval_score == 0.87  # the pool's score, not recomputed
    assert edge.relationship_type is Relationship.GROUNDS
    assert edge.evidence_type is EvidenceType.FRAMEWORK_CITATION


def test_synthesize_dedupes_duplicate_model_citations(session) -> None:
    feature_id = _one_feature(session)
    _, [chunk_id] = _seed_chunks(session, 1)
    fake = FakeRetrievalService(default=[_cit(chunk_id, 0.5)])

    result = _service(
        session, fake, client=FrameworkCitingClient(duplicate=True)
    ).synthesize_decisions([feature_id], corpus_version=CORPUS)

    session.expire_all()
    reloaded = session.get(Decision, result.decisions[0].id)
    assert len(reloaded.framework_citations) == 1  # the duplicated id collapsed to one edge


def test_synthesize_creates_no_edges_when_model_cites_no_framework(session) -> None:
    """A populated pool with a model that grounds on nothing persists zero framework edges."""

    feature_id = _one_feature(session)
    _, [chunk_id] = _seed_chunks(session, 1)
    fake = FakeRetrievalService(default=[_cit(chunk_id, 0.9)])

    result = _service(session, fake).synthesize_decisions([feature_id], corpus_version=CORPUS)

    assert result.framework_pool  # the pool was populated...
    session.expire_all()
    reloaded = session.get(Decision, result.decisions[0].id)
    assert reloaded.framework_citations == []  # ...but the model cited none, so no edges
