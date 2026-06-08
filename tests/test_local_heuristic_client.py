"""Phase L3 -- LocalHeuristicClient: input-derived, deterministic, gate-safe local engine.

Drives the client through the REAL runners + PipelineService (so the real validation
gates run) over varied inputs, plus determinism, gate-focused edge cases, and pathological
inputs. Also unit-tests the pure helpers.
"""

from __future__ import annotations

import uuid

import pytest

import app.api.app  # noqa: F401,E402  -- init api package (explanation service imports it)

from app.ai_clients.local_heuristic_client import (
    LocalHeuristicClient,
    _content_claims,
    _keywords,
    _stance,
    _union_find,
)
from app.ai_contracts.enums import StakeholderType
from app.ai_contracts.validation.grounding import token_overlap
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.pipeline_service import FeedbackEntry, PipelineService, PipelineStage
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage3.runner import build_stage3_runner
from app.stages.stage4.runner import build_stage4_runner
from showcase.lib.assemble import assemble_snapshot

from tests.app_helpers import make_engine, make_session_factory


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _service(session):
    client = LocalHeuristicClient()
    explanation = DecisionExplanationService(
        DecisionRepository(session), FeatureRepository(session),
        ConflictRepository(session), SignalRepository(session),
    )
    return PipelineService(
        session,
        stage1_runner=build_stage1_runner(client),
        stage2_runner=build_stage2_runner(client),
        stage3_runner=build_stage3_runner(client),
        stage4_runner=build_stage4_runner(client),
        explanation_service=explanation,
    )


_OPPOSING = [
    FeedbackEntry(StakeholderType.SALES, "We urgently need dark mode. Customers keep asking for it."),
    FeedbackEntry(StakeholderType.ENGINEERING, "Dark mode is risky and time-consuming to build across all screens."),
]
_AGREEING = [
    FeedbackEntry(StakeholderType.SALES, "We really need dark mode; it would add great value."),
    FeedbackEntry(StakeholderType.CUSTOMER, "Please add dark mode, it would be great and helpful."),
]


# --------------------------------------------------------------------------- #
# End-to-end through the real gates
# --------------------------------------------------------------------------- #
def test_end_to_end_succeeds_on_opposing_input(session) -> None:
    r = _service(session).analyze(_OPPOSING)
    assert r.succeeded and r.failed_stage is None
    assert len(r.signals) == 2 and len(r.features) >= 1
    assert len(r.conflicts) >= 1            # opposing stances -> a conflict
    assert len(r.decisions) >= 1 and len(r.explanations) == 1
    assert PipelineStage.EXPLANATION in r.completed_stages


def test_outputs_are_input_derived(session) -> None:
    r = _service(session).analyze(_OPPOSING)
    titles = " ".join(f.title.lower() for f in r.features)
    assert "dark" in titles or "mode" in titles   # title derived from the feedback text

    # A different topic yields a different feature title.
    other = _service(make_session_factory(make_engine())()).analyze([
        FeedbackEntry(StakeholderType.SUPPORT, "Customers want faster export of reports to csv."),
        FeedbackEntry(StakeholderType.SALES, "Report export speed is a deal driver."),
    ])
    other_titles = " ".join(f.title.lower() for f in other.features)
    assert ("export" in other_titles or "report" in other_titles or "csv" in other_titles)
    assert other_titles != titles


def test_determinism_same_input_same_content(session) -> None:
    r1 = _service(session).analyze(_OPPOSING)
    r2 = _service(make_session_factory(make_engine())()).analyze(_OPPOSING)
    # ids differ (fresh DBs) but derived content is identical.
    assert [f.title for f in r1.features] == [f.title for f in r2.features]
    assert [d.recommendation.value for d in r1.decisions] == [d.recommendation.value for d in r2.decisions]
    assert [c.conflict_type.value for c in r1.conflicts] == [c.conflict_type.value for c in r2.conflicts]
    assert [p.intent for p in r1.parsed_signals] == [p.intent for p in r2.parsed_signals]


# --------------------------------------------------------------------------- #
# Gate-focused checks (via the assembled snapshot)
# --------------------------------------------------------------------------- #
def test_grounding_claims_are_exact_substrings(session) -> None:
    snap = assemble_snapshot(_service(session).analyze(_OPPOSING))
    for a in snap["analyzed_signals"]:
        assert a["claims"]
        for c in a["claims"]:
            assert c["quoted_text"] == c["text"]               # exact-substring grounding
            assert token_overlap(c["text"], c["quoted_text"]) == 1.0


def test_provenance_is_a_full_partition(session) -> None:
    r = _service(session).analyze(_OPPOSING)
    assigned = [sid for f in r.features for fs in [f.feature_signals] for sid in [x.signal_id for x in fs]]
    all_ids = [s.id for s in r.signals]
    assert sorted(assigned) == sorted(all_ids)                 # every signal assigned once
    assert len(assigned) == len(set(assigned))                 # disjoint (no duplicates)
    assert r.unassigned_signal_ids == ()


def test_conflict_only_on_genuine_opposition(session) -> None:
    assert len(_service(session).analyze(_OPPOSING).conflicts) >= 1
    agree = _service(make_session_factory(make_engine())()).analyze(_AGREEING)
    assert agree.succeeded and agree.conflicts == ()           # all advocate -> no conflict
    assert len(agree.decisions) >= 1


def test_decision_acknowledges_conflicts_over_its_subject(session) -> None:
    r = _service(session).analyze(_OPPOSING)
    conflicts_by_subject: dict = {}
    for c in r.conflicts:
        conflicts_by_subject.setdefault(c.subject_id, set()).add(c.id)
    for d in r.decisions:
        expected = conflicts_by_subject.get(d.subject_id, set())
        acked = {a.conflict_id for a in d.acknowledged_conflicts}
        assert acked == expected                                # every (and only) same-subject conflict


def test_framework_free_yields_no_citations(session) -> None:
    r = _service(session).analyze(_OPPOSING)   # no corpus_version -> framework-free
    assert r.framework_pool == ()
    assert all(d.framework_citations == [] for d in r.decisions)


# --------------------------------------------------------------------------- #
# Pathological inputs -> graceful (no crash; success or recorded failure)
# --------------------------------------------------------------------------- #
def test_pathological_punctuation_signal_degrades_gracefully(session) -> None:
    r = _service(session).analyze([
        FeedbackEntry(StakeholderType.SALES, "!!! ??? ..."),                 # no content -> Stage 1 drops it
        FeedbackEntry(StakeholderType.ENGINEERING, "Add offline sync support."),
    ])
    assert r.failed_stage is None                              # the content-bearing signal survives
    assert len(r.signals) == 1 and len(r.decisions) >= 1


def test_all_signals_contentless_fails_without_crash(session) -> None:
    r = _service(session).analyze([
        FeedbackEntry(StakeholderType.SALES, "!!!"),
        FeedbackEntry(StakeholderType.SUPPORT, "..."),
    ])
    assert r.failed_stage is PipelineStage.SIGNAL_ANALYSIS      # graceful, recorded, no exception
    assert r.decisions == ()


# --------------------------------------------------------------------------- #
# Pure helper units
# --------------------------------------------------------------------------- #
def test_content_claims_spans_are_exact() -> None:
    raw = "SSO is critical. It blocks the Acme deal."
    for text, start, end in _content_claims(raw):
        assert raw[start:end] == text

def test_keywords_and_stance_are_input_sensitive() -> None:
    assert "dark" in _keywords("Dark mode is needed for dark interfaces")
    assert _stance("we really need this, please add it") == "advocate"
    assert _stance("this is risky and will break things") == "risk_flag"
    assert _stance("the sky is blue") == "neutral"

def test_union_find_partitions_all() -> None:
    comps = _union_find([{"a", "b"}, {"b", "c"}, {"x"}])  # 0-1 merge via "b"; 2 alone
    flat = sorted(i for comp in comps for i in comp)
    assert flat == [0, 1, 2]                                    # full coverage
    assert sorted(len(c) for c in comps) == [1, 2]              # {0,1} and {2}
