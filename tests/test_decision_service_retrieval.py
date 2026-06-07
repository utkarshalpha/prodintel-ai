"""Phase 7C-ii -- DecisionService Phase 1.5 framework-retrieval integration.

Exercises the retrieval plumbing in isolation from the model: the deterministic,
max-score, order-independent framework pool; graceful degradation vs. fail-fast error
handling; the four lifecycle log events; and -- end to end through ``synthesize_decisions``
with a real feature -- that retrieval and the commit both happen before the Claude call
and that the opt-in/empty paths leave the legacy flow unchanged.

A configurable fake RetrievalService stands in for the real one (no vector stack, no
network), so the pool algebra and error semantics are tested directly and deterministically.
"""

from __future__ import annotations

import logging
import math
from unittest import mock
from uuid import uuid4

import pytest

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType, StakeholderType
from app.ai_contracts.retrieval import RetrievalCitation, RetrievalMeta, RetrievalResult
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_service import DecisionService
from app.services.errors import CorpusEmbeddingModelMismatchError, RetrievalFailedError
from app.services.feature_service import FeatureService
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage4.prompts import FeatureForDecision, FrameworkPassage
from app.stages.stage4.runner import build_stage4_runner
from app.vector.errors import EmbeddingDimensionMismatchError

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage4FakeClient,
    make_engine,
    make_session_factory,
)

CORPUS = "pm-corpus-v1"
C1, C2, C3 = uuid4(), uuid4(), uuid4()


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Fake RetrievalService + builders
# --------------------------------------------------------------------------- #
def _cit(chunk_id, score, *, framework=FrameworkName.RICE, content="passage", title="RICE Ref"):
    return RetrievalCitation(
        chunk_id=chunk_id,
        source_id=uuid4(),
        score=score,
        content=content,
        ordinal=0,
        framework=framework,
        corpus_version=CORPUS,
        source_title=title,
        source_type=KnowledgeSourceType.FRAMEWORK,
    )


def _result(query, citations):
    n = len(citations)
    return RetrievalResult(
        query_text=query.text,
        corpus_version=query.corpus_version,
        citations=list(citations),
        meta=RetrievalMeta(
            embedding_model_id="hash-fake-v1",
            top_k=query.top_k,
            vector_match_count=n,
            resolved_count=n,
            dropped_stale_count=0,
            dropped_score_count=0,
        ),
    )


class FakeRetrievalService:
    """Returns citations by feature-title substring; can raise per-title or globally."""

    def __init__(self, by_title=None, *, default=None, raises=None, raise_all=None):
        self.by_title = by_title or {}
        self.raises = raises or {}        # title substring -> exception
        self.raise_all = raise_all        # exception raised for every query
        self.default = default            # citations for any non-matching query
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        if self.raise_all is not None:
            raise self.raise_all
        for title, exc in self.raises.items():
            if title in query.text:
                raise exc
        for title, citations in self.by_title.items():
            if title in query.text:
                return _result(query, citations)
        return _result(query, self.default or [])


def _service(session, retrieval, *, runner=None, mode="recommend"):
    return DecisionService(
        session,
        DecisionRepository(session),
        ConflictRepository(session),
        FeatureRepository(session),
        ParsedSignalRepository(session),
        runner or build_stage4_runner(Stage4FakeClient(mode=mode)),
        retrieval_service=retrieval,
    )


def _view(title, jtbd="When..., I want..., so I can..."):
    return FeatureForDecision(feature_id=uuid4(), title=title, jtbd=jtbd)


def _one_feature(session):
    """Persist one feature backed by one analyzed signal (no conflict)."""

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
# 1-3, 8: pool construction (deterministic, max-score, order-independent, frozen)
# --------------------------------------------------------------------------- #
def test_deterministic_pool_construction(session) -> None:
    fake = FakeRetrievalService(by_title={"FeatA": [_cit(C1, 0.9), _cit(C2, 0.5)]})
    service = _service(session, fake)
    views = [_view("FeatA")]

    pool_a = service._retrieve_framework_pool(CORPUS, views)
    pool_b = service._retrieve_framework_pool(CORPUS, views)
    assert pool_a == pool_b  # same inputs -> byte-identical pool
    assert [p.chunk_id for p in pool_a] == [C1, C2]  # ordered by descending score
    assert pool_a[0].framework == "RICE"  # FrameworkName mapped to its value


def test_max_score_dedupe(session) -> None:
    """A chunk surfaced by two features keeps the higher score, once."""

    fake = FakeRetrievalService(
        by_title={"FeatA": [_cit(C2, 0.5)], "FeatB": [_cit(C2, 0.8)]}
    )
    service = _service(session, fake)
    pool = service._retrieve_framework_pool(CORPUS, [_view("FeatA"), _view("FeatB")])

    assert [p.chunk_id for p in pool] == [C2]  # deduped to a single passage
    assert pool[0].retrieval_score == 0.8  # the maximum, not 0.5


def test_feature_iteration_order_independence(session) -> None:
    """Pool contents, scores, and order do not depend on feature iteration order."""

    by_title = {"FeatA": [_cit(C1, 0.9), _cit(C2, 0.5)], "FeatB": [_cit(C2, 0.8), _cit(C3, 0.3)]}
    a, b = _view("FeatA"), _view("FeatB")

    forward = _service(session, FakeRetrievalService(by_title=by_title))._retrieve_framework_pool(CORPUS, [a, b])
    reverse = _service(session, FakeRetrievalService(by_title=by_title))._retrieve_framework_pool(CORPUS, [b, a])

    assert forward == reverse
    assert [(p.chunk_id, p.retrieval_score) for p in forward] == [(C1, 0.9), (C2, 0.8), (C3, 0.3)]


def test_equal_scores_break_ties_by_chunk_id(session) -> None:
    """A score tie is ordered deterministically by chunk_id (independent of feature order)."""

    lo, hi = sorted([uuid4(), uuid4()], key=str)
    by_title = {"FeatA": [_cit(hi, 0.7)], "FeatB": [_cit(lo, 0.7)]}  # hi seen first, but lo sorts first
    pool = _service(session, FakeRetrievalService(by_title=by_title))._retrieve_framework_pool(
        CORPUS, [_view("FeatA"), _view("FeatB")]
    )
    assert [p.chunk_id for p in pool] == [lo, hi]


def test_pool_is_frozen_tuple_of_frozen_passages(session) -> None:
    fake = FakeRetrievalService(by_title={"FeatA": [_cit(C1, 0.9)]})
    pool = _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA")])

    assert isinstance(pool, tuple)
    assert all(isinstance(p, FrameworkPassage) for p in pool)
    with pytest.raises(Exception):  # frozen passage -> cannot mutate the shared pool
        pool[0].retrieval_score = 0.0


def test_non_finite_score_never_survives_dedupe(session) -> None:
    """A NaN score seen first must not block a later valid max (adversarial: score drift)."""

    fake = FakeRetrievalService(
        by_title={"FeatA": [_cit(C1, float("nan"))], "FeatB": [_cit(C1, 0.95)]}
    )
    pool = _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA"), _view("FeatB")])
    assert [p.chunk_id for p in pool] == [C1]
    assert pool[0].retrieval_score == 0.95  # the real max retained; NaN dropped


def test_non_finite_scores_excluded_from_pool(session) -> None:
    """NaN/inf-scored chunks are dropped entirely, keeping the pool finite and ordered."""

    fake = FakeRetrievalService(
        by_title={"FeatA": [_cit(C1, float("nan")), _cit(C2, float("inf")), _cit(C3, 0.4)]}
    )
    pool = _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA")])
    assert [p.chunk_id for p in pool] == [C3]  # only the finite-scored chunk survives
    assert all(math.isfinite(p.retrieval_score) for p in pool)


def test_query_text_is_title_plus_jtbd(session) -> None:
    fake = FakeRetrievalService(default=[])
    _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("Enterprise SSO", "Login fast")])
    assert fake.calls[0].text == "Enterprise SSO\nLogin fast"
    assert fake.calls[0].corpus_version == CORPUS


# --------------------------------------------------------------------------- #
# 4: graceful degradation (RetrievalFailedError)
# --------------------------------------------------------------------------- #
def test_per_feature_degradation_skips_only_that_feature(session) -> None:
    fake = FakeRetrievalService(
        by_title={"FeatB": [_cit(C3, 0.3)]},
        raises={"FeatA": RetrievalFailedError("vector query failed")},
    )
    pool = _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA"), _view("FeatB")])
    assert [p.chunk_id for p in pool] == [C3]  # FeatA degraded out; FeatB still contributes


def test_total_degradation_yields_empty_pool(session) -> None:
    fake = FakeRetrievalService(raise_all=RetrievalFailedError("embedding failed"))
    pool = _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA"), _view("FeatB")])
    assert pool == ()  # nothing retrieved -> degrade to no framework knowledge


# --------------------------------------------------------------------------- #
# 5: fail-fast (corpus/dimension mismatch) -- propagate, roll back, persist nothing
# --------------------------------------------------------------------------- #
def test_corpus_model_mismatch_fails_fast(session) -> None:
    fake = FakeRetrievalService(raise_all=CorpusEmbeddingModelMismatchError(CORPUS, "model-a", "model-b"))
    service = _service(session, fake)
    with mock.patch.object(session, "rollback", wraps=session.rollback) as spy_rollback:
        with pytest.raises(CorpusEmbeddingModelMismatchError):
            service._retrieve_framework_pool(CORPUS, [_view("FeatA")])
    assert spy_rollback.called  # session rolled back before propagating


def test_embedding_dimension_mismatch_fails_fast(session) -> None:
    fake = FakeRetrievalService(raise_all=EmbeddingDimensionMismatchError("dim 16 != 8"))
    with pytest.raises(EmbeddingDimensionMismatchError):
        _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA")])


# --------------------------------------------------------------------------- #
# 9: logging coverage -- all four lifecycle events
# --------------------------------------------------------------------------- #
def _events(caplog):
    return [getattr(r, "event", None) for r in caplog.records]


def test_logs_started_and_succeeded(session, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.services.decision_service")
    fake = FakeRetrievalService(by_title={"FeatA": [_cit(C1, 0.9)]})
    _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA")])
    events = _events(caplog)
    assert "retrieval_started" in events
    assert "retrieval_succeeded" in events


def test_logs_degraded(session, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.services.decision_service")
    fake = FakeRetrievalService(raise_all=RetrievalFailedError("boom"))
    _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA")])
    assert "retrieval_degraded" in _events(caplog)


def test_logs_failed_on_fast_fail(session, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.services.decision_service")
    fake = FakeRetrievalService(raise_all=CorpusEmbeddingModelMismatchError(CORPUS, "a", "b"))
    with pytest.raises(CorpusEmbeddingModelMismatchError):
        _service(session, fake)._retrieve_framework_pool(CORPUS, [_view("FeatA")])
    assert "retrieval_failed" in _events(caplog)


# --------------------------------------------------------------------------- #
# 6, 7 + opt-in: end-to-end through synthesize_decisions (real feature)
# --------------------------------------------------------------------------- #
def test_commit_and_retrieval_happen_before_claude(session) -> None:
    feature_id = _one_feature(session)
    fake = FakeRetrievalService(default=[_cit(C1, 0.9)])

    captured: dict[str, int] = {}

    class SpyRunner:
        def __init__(self, inner):
            self._inner = inner

        def run(self, context):
            captured["commits_before_claude"] = spy_commit.call_count
            captured["pool_size_at_claude"] = len(context.framework_knowledge)
            return self._inner.run(context)

    with mock.patch.object(session, "commit", wraps=session.commit) as spy_commit:
        service = _service(session, fake, runner=SpyRunner(build_stage4_runner(Stage4FakeClient())))
        result = service.synthesize_decisions([feature_id], corpus_version=CORPUS)

    assert captured["commits_before_claude"] >= 1          # committed before the model call (ADR-010)
    assert captured["pool_size_at_claude"] == 1            # pool built before the model call
    assert result.framework_pool[0].chunk_id == C1
    assert len(result.decisions) == 1


def test_empty_retrieval_path_synthesizes_without_framework(session) -> None:
    feature_id = _one_feature(session)
    fake = FakeRetrievalService(default=[])  # corpus known but nothing matches
    result = _service(session, fake).synthesize_decisions([feature_id], corpus_version=CORPUS)

    assert result.framework_pool == ()
    assert len(result.decisions) == 1
    assert fake.calls, "retrieval was still attempted"


def test_opt_in_off_skips_retrieval_entirely(session) -> None:
    feature_id = _one_feature(session)
    fake = FakeRetrievalService(default=[_cit(C1, 0.9)])
    result = _service(session, fake).synthesize_decisions([feature_id])  # no corpus_version

    assert fake.calls == []  # retrieval never consulted
    assert result.framework_pool == ()
    assert len(result.decisions) == 1


def test_corpus_version_without_retrieval_service_is_config_error(session) -> None:
    feature_id = _one_feature(session)
    service = _service(session, None)  # retrieval_service omitted
    with pytest.raises(RuntimeError, match="no RetrievalService"):
        service.synthesize_decisions([feature_id], corpus_version=CORPUS)
