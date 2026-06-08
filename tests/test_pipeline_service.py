"""Phase L1 -- PipelineService orchestration (saga coordinator) tests.

Exercises the coordinator with the existing deterministic stage stubs (appropriate here:
we test wiring + saga semantics + the PipelineRunResult contract, not analysis quality)
over in-memory SQLite. Covers happy path, empty-conflict success, per-stage failures,
partial Stage 1, framework grounding pass-through, corpus fail-fast capture, StageResult
preservation, and the explanation/empty-input edges.
"""

from __future__ import annotations

import re
import uuid

import pytest

# Initialize the api package first: DecisionExplanationService transitively imports
# app.api, which deadlocks on a circular import if loaded before app.api.app is ready.
import app.api.app  # noqa: F401,E402

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType, StakeholderType
from app.ai_contracts.retrieval import RetrievalCitation, RetrievalMeta, RetrievalResult
from app.ai_runtime.interfaces import LLMToolResponse
from app.ai_runtime.retry_policy import RetryPolicy
from app.models.knowledge import KnowledgeChunk, KnowledgeSource
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.errors import CorpusEmbeddingModelMismatchError
from app.services.pipeline_service import FeedbackEntry, PipelineService, PipelineStage, StageStatus
from app.stages.stage1 import prompts as s1p
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage3.runner import build_stage3_runner
from app.stages.stage4.runner import build_stage4_runner

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage3FakeClient,
    Stage4FakeClient,
    make_engine,
    make_session_factory,
)

CORPUS = "pm-corpus-v1"


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _entries():
    return [
        FeedbackEntry(StakeholderType.SALES, "Enterprise SSO is critical to close the Acme deal."),
        FeedbackEntry(StakeholderType.ENGINEERING, "SSO is high risk and needs six to eight weeks of work."),
    ]


def _explanation(session):
    return DecisionExplanationService(
        DecisionRepository(session), FeatureRepository(session),
        ConflictRepository(session), SignalRepository(session),
    )


def _service(session, *, grounded=True, s2="cluster_all", s3="conflict", s4="recommend",
             retrieval=None, explain=True, s1_client=None, s4_client=None, max_attempts=2):
    rp = RetryPolicy(max_attempts=max_attempts)
    return PipelineService(
        session,
        stage1_runner=build_stage1_runner(s1_client or Stage1FakeClient(grounded=grounded), retry_policy=rp),
        stage2_runner=build_stage2_runner(Stage2FakeClient(mode=s2), retry_policy=rp),
        stage3_runner=build_stage3_runner(Stage3FakeClient(mode=s3), retry_policy=rp),
        stage4_runner=build_stage4_runner(s4_client or Stage4FakeClient(mode=s4), retry_policy=rp),
        retrieval_service=retrieval,
        explanation_service=_explanation(session) if explain else None,
    )


def _status(result, stage):
    return next(o for o in result.stage_outcomes if o.stage == stage).status


# --------------------------------------------------------------------------- #
# Happy path + empty conflicts
# --------------------------------------------------------------------------- #
def test_full_run_succeeds(session) -> None:
    r = _service(session).analyze(_entries())
    assert r.succeeded and r.failed_stage is None
    assert len(r.signals) == 2 and len(r.parsed_signals) == 2
    assert len(r.features) >= 1 and len(r.conflicts) == 1 and len(r.decisions) >= 1
    assert len(r.explanations) == 1
    assert _status(r, PipelineStage.SIGNAL_ANALYSIS) is StageStatus.SUCCEEDED
    assert _status(r, PipelineStage.DECISION_SYNTHESIS) is StageStatus.SUCCEEDED
    assert _status(r, PipelineStage.EXPLANATION) is StageStatus.SUCCEEDED
    assert PipelineStage.EXPLANATION in r.completed_stages
    assert isinstance(r.run_id, uuid.UUID) and r.completed_at >= r.started_at


def test_empty_conflicts_is_success(session) -> None:
    r = _service(session, s3="none").analyze(_entries())
    assert r.succeeded
    assert r.conflicts == () and len(r.decisions) >= 1
    assert _status(r, PipelineStage.CONFLICT_DETECTION) is StageStatus.SUCCEEDED


# --------------------------------------------------------------------------- #
# Stage 1 -- all fail, and partial
# --------------------------------------------------------------------------- #
def test_all_signals_fail_is_fatal(session) -> None:
    r = _service(session, grounded=False, max_attempts=1).analyze(_entries())
    assert r.failed_stage is PipelineStage.SIGNAL_ANALYSIS
    assert r.parsed_signals == () and r.features == () and r.decisions == ()
    assert _status(r, PipelineStage.SIGNAL_ANALYSIS) is StageStatus.FAILED
    assert _status(r, PipelineStage.FEATURE_EXTRACTION) is StageStatus.SKIPPED


class _PartialStage1Client:
    """Grounds normally, but returns an ungrounded claim for text containing 'FAILME'."""

    def complete(self, *, system, messages, tool):
        content = messages[0].content
        signal_id = re.search(r"signal_id:\s*([0-9a-fA-F-]{36})", content).group(1)
        raw = re.search(r"<<<SIGNAL>>>\n(.*)\n<<<END>>>", content, re.DOTALL).group(1)
        hint = re.search(r"Known source channel \(hint\):\s*(\w+)", content)
        stakeholder = hint.group(1) if hint else "customer"
        if "FAILME" in raw:
            claim = {"text": "zzz qqq www unrelated", "source_span": [0, min(3, len(raw))], "claim_confidence": 0.9}
        else:
            claim = {"text": raw, "source_span": [0, len(raw)], "claim_confidence": 0.95}
        payload = {"signal_id": signal_id, "intent": "intent", "stakeholder_type": stakeholder,
                   "urgency": 3, "sentiment": 0.0, "extracted_claims": [claim],
                   "confidence": {"score": 0.8, "components": {"span_grounding_ratio": 1.0}}}
        return LLMToolResponse(tool_name=s1p.STAGE1_TOOL_NAME, tool_input=payload, stop_reason="tool_use",
                               model_id="partial", input_tokens=10, output_tokens=5)


def test_partial_signal_analysis_is_non_fatal(session) -> None:
    entries = [
        FeedbackEntry(StakeholderType.SALES, "SSO is critical for the Acme deal."),
        FeedbackEntry(StakeholderType.ENGINEERING, "SSO is high risk to build."),
        FeedbackEntry(StakeholderType.CUSTOMER, "FAILME this claim will not ground."),
    ]
    r = _service(session, s1_client=_PartialStage1Client(), max_attempts=1).analyze(entries)
    assert r.failed_stage is None              # partial Stage 1 does not abort the run
    assert len(r.signals) == 2                 # the FAILME entry dropped
    assert _status(r, PipelineStage.SIGNAL_ANALYSIS) is StageStatus.PARTIAL
    assert PipelineStage.SIGNAL_ANALYSIS in r.completed_stages
    assert len(r.decisions) >= 1


# --------------------------------------------------------------------------- #
# Stage 2 / 3 / 4 fatal failures
# --------------------------------------------------------------------------- #
def test_feature_extraction_failure(session) -> None:
    r = _service(session, s2="fabricate", max_attempts=1).analyze(_entries())
    assert r.failed_stage is PipelineStage.FEATURE_EXTRACTION
    assert len(r.signals) == 2 and r.features == () and r.decisions == ()
    assert _status(r, PipelineStage.FEATURE_EXTRACTION) is StageStatus.FAILED
    assert _status(r, PipelineStage.DECISION_SYNTHESIS) is StageStatus.SKIPPED
    assert PipelineStage.SIGNAL_ANALYSIS in r.completed_stages
    assert PipelineStage.FEATURE_EXTRACTION not in r.completed_stages


def test_conflict_detection_failure(session) -> None:
    r = _service(session, s3="no_opposition", max_attempts=1).analyze(_entries())
    assert r.failed_stage is PipelineStage.CONFLICT_DETECTION
    assert len(r.features) >= 1 and r.decisions == ()


def test_decision_synthesis_failure(session) -> None:
    r = _service(session, s4="ignore_conflict", max_attempts=1).analyze(_entries())
    assert r.failed_stage is PipelineStage.DECISION_SYNTHESIS
    assert len(r.conflicts) == 1 and r.decisions == () and r.explanations == ()
    assert _status(r, PipelineStage.EXPLANATION) is StageStatus.SKIPPED


# --------------------------------------------------------------------------- #
# Framework grounding pass-through + corpus fail-fast capture
# --------------------------------------------------------------------------- #
def _seed_chunk(session):
    src = KnowledgeSource(source_type="framework", framework="RICE", title="RICE Ref",
                          corpus_version=CORPUS, embedding_model_id="hash-fake-v1", content_hash="a" * 64)
    chunk = KnowledgeChunk(corpus_version=CORPUS, embedding_model_id="hash-fake-v1", ordinal=0,
                           framework="RICE", content="RICE prioritizes reach x impact x confidence / effort.",
                           content_hash="b" * 64, token_count=10)
    src.chunks.append(chunk)
    session.add(src)
    session.commit()
    return chunk.id


def _cit(chunk_id, score):
    return RetrievalCitation(chunk_id=chunk_id, source_id=uuid.uuid4(), score=score, content="passage",
                             ordinal=0, framework=FrameworkName.RICE, corpus_version=CORPUS,
                             source_title="RICE Ref", source_type=KnowledgeSourceType.FRAMEWORK)


class _FakeRetrieval:
    def __init__(self, citations=None, *, raises=None):
        self._c = citations or []
        self._raises = raises

    def retrieve(self, query):
        if self._raises is not None:
            raise self._raises
        n = len(self._c)
        return RetrievalResult(query_text=query.text, corpus_version=query.corpus_version, citations=list(self._c),
                               meta=RetrievalMeta(embedding_model_id="hash-fake-v1", top_k=query.top_k,
                                                  vector_match_count=n, resolved_count=n,
                                                  dropped_stale_count=0, dropped_score_count=0))


_CHUNK_RE = re.compile(r"chunk_id:\s*([0-9a-fA-F-]{36})")


class _FrameworkCitingClient:
    def __init__(self):
        self._inner = Stage4FakeClient()

    def complete(self, *, system, messages, tool):
        resp = self._inner.complete(system=system, messages=messages, tool=tool)
        chunk_ids = _CHUNK_RE.findall(messages[0].content)
        if chunk_ids:
            for d in resp.tool_input["decisions"]:
                d["framework_citation_ids"] = list(chunk_ids)
        return resp


def test_framework_grounding_pass_through(session) -> None:
    chunk_id = _seed_chunk(session)
    r = _service(session, retrieval=_FakeRetrieval([_cit(chunk_id, 0.83)]),
                 s4_client=_FrameworkCitingClient()).analyze(_entries(), corpus_version=CORPUS)
    assert r.succeeded
    assert len(r.framework_pool) == 1 and r.framework_pool[0].chunk_id == chunk_id
    assert len(r.decisions) >= 1


def test_corpus_mismatch_is_captured_not_raised(session) -> None:
    _seed_chunk(session)
    retrieval = _FakeRetrieval(raises=CorpusEmbeddingModelMismatchError(CORPUS, "model-a", "model-b"))
    r = _service(session, retrieval=retrieval).analyze(_entries(), corpus_version=CORPUS)
    assert r.failed_stage is PipelineStage.DECISION_SYNTHESIS
    assert r.decisions == ()
    dec = next(o for o in r.stage_outcomes if o.stage is PipelineStage.DECISION_SYNTHESIS)
    assert dec.status is StageStatus.FAILED and dec.error_code == "CorpusEmbeddingModelMismatchError"
    assert len(r.conflicts) == 1  # earlier stages remain committed (partial-state saga)


# --------------------------------------------------------------------------- #
# Contract: StageResult preservation, empty input, explanation opt-out
# --------------------------------------------------------------------------- #
def test_stage_results_are_preserved(session) -> None:
    r = _service(session).analyze(_entries())
    s1 = next(o for o in r.stage_outcomes if o.stage is PipelineStage.SIGNAL_ANALYSIS)
    assert len(s1.stage_results) == 2                     # one StageResult per analyzed signal
    dec = next(o for o in r.stage_outcomes if o.stage is PipelineStage.DECISION_SYNTHESIS)
    assert len(dec.stage_results) == 1
    assert dec.stage_results[0].metrics.total_input_tokens >= 0   # real StageResult preserved


def test_empty_entries_raises(session) -> None:
    with pytest.raises(ValueError):
        _service(session).analyze([])


def test_explanation_skipped_when_not_configured(session) -> None:
    r = _service(session, explain=False).analyze(_entries())
    assert r.succeeded and r.explanations == ()
    assert _status(r, PipelineStage.EXPLANATION) is StageStatus.SKIPPED
