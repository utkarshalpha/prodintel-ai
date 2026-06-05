"""Phase 5 (A) -- pure ExplanationAssembler tests.

Exercises the assembler with **detached, in-memory ORM objects** (no database, no
session): determinism, signal deduplication, reached_via tagging, quoted_text
generation, and integrity failures.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.ai_contracts.enums import (
    ConflictStatus,
    ConflictType,
    DecisionRecommendation,
    DecisionStatus,
    FeatureStatus,
    Stance,
    StakeholderType,
    SubjectType,
)
from app.api.explanation_assembler import ExplanationAssembler
from app.models.conflict import Conflict, ConflictParty
from app.models.decision import Decision, DecisionEvidence
from app.models.feature import Feature, FeatureSignal
from app.models.signal import ParsedSignal, Signal

FIXED = datetime(2026, 6, 5, 12, 0, 0, tzinfo=timezone.utc)
CONF = {"score": 0.8, "basis": "strong", "components": {}}


def _signal(sid, raw="Enterprise SSO is critical", *, source="sales", with_analysis=True, span=None):
    span = span or (0, len(raw))
    sig = Signal(
        id=sid, workspace_id=None, source_type=StakeholderType(source),
        raw_text=raw, source_ref=None, content_hash=sid.hex, created_at=FIXED,
    )
    sig.parsed = (
        ParsedSignal(
            id=uuid4(), signal_id=sid, intent="wants the feature",
            stakeholder_type=StakeholderType(source), urgency=4, sentiment=0.0,
            extracted_claims=[{"text": raw[span[0]:span[1]], "source_span": [span[0], span[1]], "claim_confidence": 0.9}],
            confidence_score=0.8, confidence=CONF, model_meta={"source": "test"}, created_at=FIXED,
        )
        if with_analysis else None
    )
    return sig


def _decision(did, subject_id, evidence_signal_ids):
    d = Decision(
        id=did, workspace_id=None, subject_type=SubjectType.FEATURE, subject_id=subject_id,
        recommendation=DecisionRecommendation.BUILD_NOW, title="Build SSO",
        rationale="the evidence supports building it", priority_rank=1, status=DecisionStatus.PROPOSED,
        confidence_score=0.8, confidence=CONF, model_meta={"source": "test"}, created_at=FIXED,
    )
    d.evidence = [DecisionEvidence(decision_id=did, signal_id=sid, created_at=FIXED) for sid in evidence_signal_ids]
    return d


def _feature(fid, signal_ids):
    f = Feature(
        id=fid, workspace_id=None, title="Enterprise SSO", description="d",
        jtbd="When I evaluate vendors, I want SSO, so I can meet policy",
        status=FeatureStatus.CANDIDATE, confidence_score=0.8, confidence=CONF,
        model_meta={"source": "test"}, created_at=FIXED,
    )
    f.feature_signals = [FeatureSignal(feature_id=fid, signal_id=sid, created_at=FIXED) for sid in signal_ids]
    return f


def _conflict(cid, parties, *, severity=4):
    c = Conflict(
        id=cid, workspace_id=None, subject_type=SubjectType.FEATURE, subject_id=uuid4(),
        conflict_type=ConflictType.RISK, severity=severity, status=ConflictStatus.OPEN,
        confidence_score=0.8, confidence=CONF, model_meta={"source": "test"}, created_at=FIXED,
    )
    c.parties = [
        ConflictParty(
            id=uuid4(), conflict_id=cid, stakeholder_type=StakeholderType(st), stance=Stance(stance),
            summary=f"{st} position", evidence_signal_ids=[str(s) for s in sids], created_at=FIXED,
        )
        for st, stance, sids in parties
    ]
    return c


# --------------------------------------------------------------------------- #
# Structure, dedup, reached_via, quoted_text
# --------------------------------------------------------------------------- #
def test_assemble_produces_full_explanation() -> None:
    s1, s2 = uuid4(), uuid4()
    fid, cid, did = uuid4(), uuid4(), uuid4()
    sig1, sig2 = _signal(s1, source="sales"), _signal(s2, "SSO is high risk", source="engineering")
    decision = _decision(did, fid, [s1, s2])
    feature = _feature(fid, [s1, s2])
    conflict = _conflict(cid, [("sales", "advocate", [s1]), ("engineering", "risk_flag", [s2])])

    resp = ExplanationAssembler.assemble(decision, feature, [conflict], [sig1, sig2])

    assert resp.decision.id == did
    assert resp.subject_feature.id == fid
    assert {e.signal_id for e in resp.direct_evidence} == {s1, s2}
    assert len(resp.conflicts) == 1 and len(resp.conflicts[0].parties) == 2
    assert {sp.id for sp in resp.signals} == {s1, s2}  # dedup: each signal once
    assert resp.integrity.complete is True
    assert resp.meta.signal_count == 2 and resp.meta.conflict_count == 1


def test_reached_via_records_every_path() -> None:
    s1 = uuid4()
    fid, cid, did = uuid4(), uuid4(), uuid4()
    sig1 = _signal(s1)
    decision = _decision(did, fid, [s1])               # direct evidence
    feature = _feature(fid, [s1])                       # feature path
    conflict = _conflict(cid, [("sales", "advocate", [s1]), ("engineering", "risk_flag", [s1])])  # conflict path

    resp = ExplanationAssembler.assemble(decision, feature, [conflict], [sig1])
    sp = resp.signals[0]
    assert sp.reached_via == ["direct_evidence", f"feature:{fid}", f"conflict:{cid}"]


def test_quoted_text_matches_source_span() -> None:
    s1 = uuid4()
    raw = "Enterprise SSO is critical for closing the deal"
    sig = _signal(s1, raw, span=(0, 13))  # "Enterprise SS" ... use a real sub-span
    decision = _decision(uuid4(), uuid4(), [s1])
    resp = ExplanationAssembler.assemble(decision, None, [], [sig])
    claim = resp.signals[0].analysis.claims[0]
    assert claim.quoted_text == raw[claim.source_span[0]:claim.source_span[1]] == raw[0:13]


def test_signal_without_analysis_is_handled() -> None:
    s1 = uuid4()
    sig = _signal(s1, with_analysis=False)
    resp = ExplanationAssembler.assemble(_decision(uuid4(), uuid4(), [s1]), None, [], [sig])
    assert resp.signals[0].analysis is None


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_output_is_independent_of_input_order() -> None:
    s1, s2 = uuid4(), uuid4()
    fid, did = uuid4(), uuid4()
    c1, c2 = uuid4(), uuid4()
    sig1, sig2 = _signal(s1, source="sales"), _signal(s2, "SSO risk", source="engineering")
    decision = _decision(did, fid, [s1, s2])
    feature = _feature(fid, [s1, s2])
    conflict1 = _conflict(c1, [("sales", "advocate", [s1]), ("engineering", "risk_flag", [s2])], severity=3)
    conflict2 = _conflict(c2, [("sales", "advocate", [s1]), ("support", "oppose", [s2])], severity=5)

    a = ExplanationAssembler.assemble(decision, feature, [conflict1, conflict2], [sig1, sig2])
    # shuffle conflicts, signals, and the decision's evidence order
    decision.evidence = list(reversed(decision.evidence))
    b = ExplanationAssembler.assemble(decision, feature, [conflict2, conflict1], [sig2, sig1])

    assert a.model_dump() == b.model_dump()


# --------------------------------------------------------------------------- #
# Integrity (annotate, never silently drop)
# --------------------------------------------------------------------------- #
def test_unresolved_conflict_evidence_is_reported_not_dropped() -> None:
    s1, s_missing = uuid4(), uuid4()
    fid, cid, did = uuid4(), uuid4(), uuid4()
    sig1 = _signal(s1)
    decision = _decision(did, fid, [s1])
    feature = _feature(fid, [s1])
    conflict = _conflict(cid, [("sales", "advocate", [s1]), ("engineering", "risk_flag", [s_missing])])

    resp = ExplanationAssembler.assemble(decision, feature, [conflict], [sig1])  # s_missing not loaded

    assert resp.integrity.complete is False
    assert s_missing in resp.integrity.unresolved_signal_ids
    assert any("did not resolve" in note for note in resp.integrity.notes)
    # not silently dropped: still present on the conflict party...
    party_evidence = [sid for p in resp.conflicts[0].parties for sid in p.evidence_signal_ids]
    assert s_missing in party_evidence
    # ...and absent from signals (no row can be fabricated)
    assert s_missing not in {sp.id for sp in resp.signals}


def test_unresolved_subject_feature_is_reported() -> None:
    s1 = uuid4()
    did, fid = uuid4(), uuid4()
    resp = ExplanationAssembler.assemble(_decision(did, fid, [s1]), None, [], [_signal(s1)])
    assert resp.subject_feature is None
    assert resp.integrity.complete is False
    assert any("subject feature" in note for note in resp.integrity.notes)
