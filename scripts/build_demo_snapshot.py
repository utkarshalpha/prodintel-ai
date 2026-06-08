"""Phase 1 (Streamlit showcase) -- generate the deterministic demo snapshot.

Runs the TeamFlow scenario through the REAL pipeline -- the real services, validation
gates, provenance edges, framework grounding, persistence, and `/why` traversal -- with
deterministic, scenario-specific LLM stubs standing in for Claude (exactly as the test
suite stubs the LLM boundary). Only the LLM output is faked; every trust mechanism is the
production code path.

The run uses random UUIDs and wall-clock timestamps, so for a byte-deterministic artifact
the script remaps every entity id to a stable role label (sig-01, feat-sso, ...) and pins
all `created_at` values after the run. Re-running produces an identical file.

Output: showcase/data/demo_snapshot.json -- the single source of truth for the showcase.

Run:  python scripts/build_demo_snapshot.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import Session, sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import app.models  # noqa: E402,F401  -- registers tables on Base.metadata
import app.api.app  # noqa: E402,F401  -- initialize the api package first (avoids a circular import)
from app.ai_contracts.base import ModelMeta  # noqa: E402
from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType, StakeholderType  # noqa: E402
from app.ai_contracts.retrieval import RetrievalCitation, RetrievalMeta, RetrievalResult  # noqa: E402
from app.ai_contracts.stage4_decision import DecisionContract, DecisionSynthesisContract  # noqa: E402
from app.ai_runtime.interfaces import LLMMessage, LLMToolResponse, ToolSpec  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.models.knowledge import KnowledgeChunk, KnowledgeSource  # noqa: E402
from app.repositories.conflict_repository import ConflictRepository  # noqa: E402
from app.repositories.decision_repository import DecisionRepository  # noqa: E402
from app.repositories.feature_repository import FeatureRepository  # noqa: E402
from app.repositories.signal_repository import SignalRepository  # noqa: E402
from app.services.decision_explanation_service import DecisionExplanationService  # noqa: E402
from app.services.pipeline_service import FeedbackEntry, PipelineService  # noqa: E402
from app.stages.stage1.runner import build_stage1_runner  # noqa: E402
from app.stages.stage1 import prompts as s1p  # noqa: E402
from app.stages.stage2.runner import build_stage2_runner  # noqa: E402
from app.stages.stage2 import prompts as s2p  # noqa: E402
from app.stages.stage3.runner import build_stage3_runner  # noqa: E402
from app.stages.stage3 import prompts as s3p  # noqa: E402
from app.stages.stage4.runner import Stage4Context, build_stage4_runner  # noqa: E402
from app.stages.stage4 import prompts as s4p  # noqa: E402
from app.stages.stage4.prompts import ConflictForDecision, FeatureForDecision, SignalForDecision  # noqa: E402
from app.stages.stage4.validators import DecisionIntegrityValidator  # noqa: E402
from showcase.lib.assemble import assemble_snapshot  # noqa: E402

CORPUS = "pm-corpus-v1"
_FIXED_TS = "2026-01-01T00:00:00+00:00"
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_SIGNAL_BLOCK_RE = re.compile(r"signal_id:\s*([0-9a-fA-F-]{36})(.*?)(?=signal_id:|\Z)", re.DOTALL)
_FEATURE_ID_RE = re.compile(r"feature_id:\s*([0-9a-fA-F-]{36})")
_CONFLICT_SUBJECT_RE = re.compile(r"conflict_id:\s*([0-9a-fA-F-]{36})\s*\|\s*subject_id:\s*([0-9a-fA-F-]{36})")
_CHUNK_ID_RE = re.compile(r"chunk_id:\s*([0-9a-fA-F-]{36})")
_STAKEHOLDER_RE = re.compile(r"stakeholder:\s*(\w+)")
_RAW_RE = re.compile(r"<<<SIGNAL>>>\n(.*)\n<<<END>>>", re.DOTALL)
_SIGNAL_ID_RE = re.compile(r"signal_id:\s*([0-9a-fA-F-]{36})")


# --------------------------------------------------------------------------- #
# Scenario content (TeamFlow: a B2B SaaS tool moving upmarket to enterprise)
# --------------------------------------------------------------------------- #
SIGNALS = [
    {"role": "sig-01", "source_type": StakeholderType.SALES,
     "raw": "Enterprise SSO (SAML) is a hard blocker for the Acme Corp deal worth $240K ARR. "
            "Two other enterprise prospects asked for it this month.",
     "key": "Acme", "intent": "Unblock the Acme enterprise deal by shipping SSO", "urgency": 5, "sentiment": -0.4,
     "claims": ["Enterprise SSO (SAML) is a hard blocker", "Acme Corp deal worth $240K ARR"]},
    {"role": "sig-02", "source_type": StakeholderType.ENGINEERING,
     "raw": "Building SAML SSO properly is high-risk: it touches authentication across the whole "
            "platform and needs roughly 6 to 8 weeks plus a security review.",
     "key": "security review", "intent": "Flag SSO as a high-risk, multi-week effort needing a security review",
     "urgency": 4, "sentiment": -0.3,
     "claims": ["Building SAML SSO properly is high-risk", "needs roughly 6 to 8 weeks plus a security review"]},
    {"role": "sig-03", "source_type": StakeholderType.CUSTOMER,
     "raw": "Our security team will not approve TeamFlow for company-wide rollout without SSO and "
            "SCIM user provisioning.",
     "key": "SCIM", "intent": "Require SSO and SCIM before approving company-wide rollout", "urgency": 4,
     "sentiment": -0.2,
     "claims": ["security team will not approve TeamFlow for company-wide rollout",
                "without SSO and SCIM user provisioning"]},
    {"role": "sig-04", "source_type": StakeholderType.SUPPORT,
     "raw": "Admins send 12 to 15 tickets a week asking for directory sync and bulk user provisioning.",
     "key": "tickets", "intent": "Reduce manual user-management load via directory sync", "urgency": 3,
     "sentiment": -0.1,
     "claims": ["12 to 15 tickets a week asking for directory sync", "bulk user provisioning"]},
    {"role": "sig-05", "source_type": StakeholderType.LEADERSHIP,
     "raw": "Our goal this year is moving upmarket to enterprise, so anything that unblocks enterprise "
            "logos should be prioritized this quarter.",
     "key": "upmarket", "intent": "Prioritize enterprise-unblocking work this quarter", "urgency": 4,
     "sentiment": 0.2,
     "claims": ["moving upmarket to enterprise",
                "anything that unblocks enterprise logos should be prioritized this quarter"]},
    {"role": "sig-06", "source_type": StakeholderType.ENGINEERING,
     "raw": "We already committed to shipping mobile offline mode this quarter, and the team cannot "
            "deliver both SSO and offline well at once.",
     "key": "offline", "intent": "Surface the capacity conflict between SSO and mobile offline", "urgency": 3,
     "sentiment": -0.2,
     "claims": ["already committed to shipping mobile offline mode this quarter",
                "the team cannot deliver both SSO and offline well at once"]},
]


def _signal_by_key(raw: str) -> dict:
    for s in SIGNALS:
        if s["key"] in raw:
            return s
    raise RuntimeError(f"no scenario signal matched raw text: {raw[:60]!r}")


# --------------------------------------------------------------------------- #
# Scenario LLM stubs (deterministic; the only test doubles)
# --------------------------------------------------------------------------- #
def _resp(tool_name, payload, *, in_tok=150, out_tok=70):
    return LLMToolResponse(
        tool_name=tool_name, tool_input=payload, stop_reason="tool_use",
        model_id="claude-demo-stub", input_tokens=in_tok, output_tokens=out_tok,
    )


class ScenarioStage1Client:
    """Emits source-anchored claims for the matched scenario signal (real grounding)."""

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        content = messages[0].content
        signal_id = _SIGNAL_ID_RE.search(content).group(1)
        raw = _RAW_RE.search(content).group(1)
        hint = re.search(r"Known source channel \(hint\):\s*(\w+)", content)
        stakeholder = hint.group(1) if hint else "customer"
        spec = _signal_by_key(raw)
        claims = []
        for text in spec["claims"]:
            idx = raw.find(text)
            if idx < 0:
                raise RuntimeError(f"claim not a substring of raw ({spec['role']}): {text!r}")
            claims.append({"text": text, "source_span": [idx, idx + len(text)], "claim_confidence": 0.93})
        payload = {
            "signal_id": signal_id, "intent": spec["intent"], "stakeholder_type": stakeholder,
            "urgency": spec["urgency"], "sentiment": spec["sentiment"], "extracted_claims": claims,
            "confidence": {"score": 0.86, "components": {"span_grounding_ratio": 1.0}},
        }
        return _resp(s1p.STAGE1_TOOL_NAME, payload)


class ScenarioStage2Client:
    """Clusters signals into two features (SSO, Offline); leadership signal is unassigned."""

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        content = messages[0].content
        sso, offline, unassigned = [], [], []
        for sid, block in _SIGNAL_BLOCK_RE.findall(content):
            if "offline" in block:
                offline.append(sid)
            elif "upmarket" in block or "source: leadership" in block:
                unassigned.append(sid)
            else:
                sso.append(sid)
        features = [
            {"feature_id": "f-sso", "title": "Enterprise SSO & SCIM Provisioning",
             "description": "SAML single sign-on and directory-synced user provisioning for enterprise rollout.",
             "jtbd": "When I roll out TeamFlow company-wide, I want SAML SSO with directory-synced provisioning, "
                     "so I can satisfy my security policy",
             "source_signal_ids": sso, "confidence": {"score": 0.88, "components": {"cluster_cohesion": 0.9}}},
            {"feature_id": "f-offline", "title": "Mobile Offline Mode",
             "description": "View and edit tasks without connectivity, syncing when back online.",
             "jtbd": "When I am traveling without connectivity, I want to view and edit tasks offline, "
                     "so I can stay productive",
             "source_signal_ids": offline, "confidence": {"score": 0.79, "components": {"cluster_cohesion": 0.8}}},
        ]
        payload = {"features": features, "unassigned_signal_ids": unassigned}
        return _resp(s2p.STAGE2_TOOL_NAME, payload)


class ScenarioStage3Client:
    """Surfaces a risk conflict and a resource conflict, both over the SSO feature."""

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        content = messages[0].content
        feature_ids = _FEATURE_ID_RE.findall(content)
        subject = feature_ids[0]  # SSO feature is passed first
        sales = eng_risk = eng_cap = None
        for sid, block in _SIGNAL_BLOCK_RE.findall(content):
            st = _STAKEHOLDER_RE.search(block)
            st = st.group(1) if st else ""
            if st == "sales":
                sales = sid
            elif st == "engineering" and "offline" in block:
                eng_cap = sid
            elif st == "engineering":
                eng_risk = sid
        conflicts = [
            {"conflict_id": "c-risk", "conflict_type": "risk", "severity": 4, "subject_type": "feature",
             "subject_id": subject, "stakeholders": ["sales", "engineering"],
             "positions": [
                 {"stakeholder": "sales", "stance": "advocate",
                  "summary": "Enterprise SSO unblocks the Acme deal and two more prospects.",
                  "evidence_signal_ids": [sales]},
                 {"stakeholder": "engineering", "stance": "risk_flag",
                  "summary": "SSO is a high-risk, ~6-8 week auth change needing a security review.",
                  "evidence_signal_ids": [eng_risk]}],
             "evidence_signal_ids": [sales, eng_risk],
             "confidence": {"score": 0.82, "components": {"opposition_strength": 0.85}}},
            {"conflict_id": "c-resource", "conflict_type": "resource", "severity": 3, "subject_type": "feature",
             "subject_id": subject, "stakeholders": ["sales", "engineering"],
             "positions": [
                 {"stakeholder": "sales", "stance": "advocate",
                  "summary": "SSO must be the priority this quarter to unblock enterprise revenue.",
                  "evidence_signal_ids": [sales]},
                 {"stakeholder": "engineering", "stance": "risk_flag",
                  "summary": "The team cannot deliver both SSO and mobile offline well in one quarter.",
                  "evidence_signal_ids": [eng_cap]}],
             "evidence_signal_ids": [sales, eng_cap],
             "confidence": {"score": 0.78, "components": {"opposition_strength": 0.8}}},
        ]
        return _resp(s3p.STAGE3_TOOL_NAME, {"conflicts": conflicts})


class ScenarioStage4Client:
    """Builds the grounded SSO decision (acks both conflicts, cites RICE+Kano) + defers offline."""

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        content = messages[0].content
        feature_ids = _FEATURE_ID_RE.findall(content)
        conflict_pairs = _CONFLICT_SUBJECT_RE.findall(content)
        chunk_ids = _CHUNK_ID_RE.findall(content)
        blocks = _SIGNAL_BLOCK_RE.findall(content)
        non_offline = [sid for sid, block in blocks if "offline" not in block]
        offline = [sid for sid, block in blocks if "offline" in block]

        decisions = []
        for feature_id in feature_ids:
            acknowledged = [cid for cid, subj in conflict_pairs if subj == feature_id]
            if acknowledged:  # the SSO feature -> the build_now, grounded decision
                decisions.append({
                    "decision_id": "d-sso", "subject_type": "feature", "subject_id": feature_id,
                    "recommendation": "build_now", "priority_rank": 1,
                    "title": "Build Enterprise SSO & SCIM this quarter, behind a security review",
                    "rationale": (
                        "A $240K deal and two more enterprise prospects are blocked on SSO, and a customer "
                        "security team requires SCIM provisioning -- high reach and revenue-gating impact. "
                        "RICE ranks this highly even at high effort, and Kano classes SSO as a 'must-be' for "
                        "enterprise buyers. The engineering risk is acknowledged and addressed with a "
                        "mandatory security review and a phased rollout; the capacity conflict is acknowledged "
                        "by deferring mobile offline to next quarter."),
                    "acknowledged_conflict_ids": acknowledged,
                    "evidence_signal_ids": non_offline,
                    "framework_citation_ids": list(chunk_ids),
                    "confidence": {"score": 0.84, "components": {"evidence_coverage": 0.9, "framework_support": 0.82}},
                })
            else:  # the offline feature -> deferred
                decisions.append({
                    "decision_id": "d-offline", "subject_type": "feature", "subject_id": feature_id,
                    "recommendation": "build_later", "priority_rank": 2,
                    "title": "Defer Mobile Offline Mode to next quarter",
                    "rationale": (
                        "Valuable but not revenue-gating this quarter; deferred to resolve the capacity "
                        "conflict with the higher-priority SSO work."),
                    "acknowledged_conflict_ids": [],
                    "evidence_signal_ids": offline,
                    "framework_citation_ids": [],
                    "confidence": {"score": 0.70, "components": {"evidence_coverage": 0.75}},
                })
        return _resp(s4p.STAGE4_TOOL_NAME, {"decisions": decisions})


# --------------------------------------------------------------------------- #
# Fake retrieval (offline; returns the seeded RICE + Kano chunks)
# --------------------------------------------------------------------------- #
class FakeRetrievalService:
    def __init__(self, citations):
        self._citations = citations

    def retrieve(self, query):
        n = len(self._citations)
        return RetrievalResult(
            query_text=query.text, corpus_version=query.corpus_version, citations=list(self._citations),
            meta=RetrievalMeta(embedding_model_id="hash-fake-v1", top_k=query.top_k, vector_match_count=n,
                               resolved_count=n, dropped_stale_count=0, dropped_score_count=0),
        )


# --------------------------------------------------------------------------- #
# DB helpers (self-contained; mirrors the test in-memory SQLite + FK pragma)
# --------------------------------------------------------------------------- #
def _make_session() -> Session:
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool, future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _record):  # pragma: no cover
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session)()


def build_snapshot() -> dict:
    session = _make_session()

    # ---- Seed the framework corpus (RICE + Kano) so citations have real FKs. -
    src_rice = KnowledgeSource(source_type=KnowledgeSourceType.FRAMEWORK, framework=FrameworkName.RICE,
                               title="RICE Prioritization (Intercom)", corpus_version=CORPUS,
                               embedding_model_id="hash-fake-v1", content_hash="a" * 64)
    chunk_rice = KnowledgeChunk(corpus_version=CORPUS, embedding_model_id="hash-fake-v1", ordinal=0,
                                framework=FrameworkName.RICE,
                                content="RICE scores prioritize by Reach x Impact x Confidence / Effort; "
                                        "revenue-gating work with broad reach ranks highly even at high effort.",
                                content_hash="b" * 64, token_count=28)
    src_rice.chunks.append(chunk_rice)
    src_kano = KnowledgeSource(source_type=KnowledgeSourceType.FRAMEWORK, framework=FrameworkName.KANO,
                               title="Kano Model -- Feature Categorization", corpus_version=CORPUS,
                               embedding_model_id="hash-fake-v1", content_hash="c" * 64)
    chunk_kano = KnowledgeChunk(corpus_version=CORPUS, embedding_model_id="hash-fake-v1", ordinal=0,
                                framework=FrameworkName.KANO,
                                content="'Must-be' features such as SSO for enterprise buyers cause "
                                        "disproportionate dissatisfaction when absent; their presence is expected.",
                                content_hash="d" * 64, token_count=24)
    src_kano.chunks.append(chunk_kano)
    session.add_all([src_rice, src_kano])
    session.commit()

    def _cit(chunk, score, framework, title):
        return RetrievalCitation(chunk_id=chunk.id, source_id=chunk.source_id, score=score, content=chunk.content,
                                 ordinal=0, framework=framework, corpus_version=CORPUS, source_title=title,
                                 source_type=KnowledgeSourceType.FRAMEWORK)

    retrieval = FakeRetrievalService([
        _cit(chunk_rice, 0.83, FrameworkName.RICE, "RICE Prioritization (Intercom)"),
        _cit(chunk_kano, 0.79, FrameworkName.KANO, "Kano Model -- Feature Categorization"),
    ])

    # ---- Run the TeamFlow scenario through the REAL coordinator (PipelineService).
    explanation = DecisionExplanationService(DecisionRepository(session), FeatureRepository(session),
                                             ConflictRepository(session), SignalRepository(session))
    pipeline = PipelineService(
        session,
        stage1_runner=build_stage1_runner(ScenarioStage1Client()),
        stage2_runner=build_stage2_runner(ScenarioStage2Client()),
        stage3_runner=build_stage3_runner(ScenarioStage3Client()),
        stage4_runner=build_stage4_runner(ScenarioStage4Client()),
        retrieval_service=retrieval,
        explanation_service=explanation,
    )
    entries = [FeedbackEntry(stakeholder_type=spec["source_type"], text=spec["raw"]) for spec in SIGNALS]
    run = pipeline.analyze(entries, corpus_version=CORPUS)

    # ---- Derive the stable id -> label map from the run's entities. ----------
    role_by_raw = {spec["raw"]: spec["role"] for spec in SIGNALS}
    id_map: dict[str, str] = {}
    sig_by_role: dict[str, Any] = {}
    for s in run.signals:
        role = role_by_raw[s.raw_text]
        id_map[str(s.id)] = role
        sig_by_role[role] = s
    feat_by_title = {f.title: f for f in run.features}
    feat_sso = feat_by_title["Enterprise SSO & SCIM Provisioning"]
    feat_offline = feat_by_title["Mobile Offline Mode"]
    id_map[str(feat_sso.id)] = "feat-sso"
    id_map[str(feat_offline.id)] = "feat-offline"
    conf_by_type = {c.conflict_type.value: c for c in run.conflicts}
    conf_risk = conf_by_type["risk"]
    conf_resource = conf_by_type["resource"]
    id_map[str(conf_risk.id)] = "conf-risk"
    id_map[str(conf_resource.id)] = "conf-resource"
    for d in run.decisions:
        id_map[str(d.id)] = "dec-01" if d.subject_id == feat_sso.id else "dec-02"
    id_map[str(chunk_rice.id)] = "chunk-rice"
    id_map[str(chunk_kano.id)] = "chunk-kano"
    id_map[str(src_rice.id)] = "source-rice"
    id_map[str(src_kano.id)] = "source-kano"

    # ---- Build the snapshot via the single source of truth, then the demo-only bits.
    snapshot = assemble_snapshot(
        run,
        scenario="TeamFlow -- B2B SaaS moving upmarket to enterprise",
        generated_by="scripts/build_demo_snapshot.py",
        note="Generated from a real pipeline run (real services, validation gates, provenance edges, "
             "framework grounding, persistence, and /why). The LLM boundary is deterministically stubbed, "
             "exactly as in the test suite; entity ids are remapped to stable role labels and timestamps "
             "are pinned so the artifact is byte-deterministic.",
    )
    snapshot["invalid_example"] = _build_invalid_example(
        feat_sso, feat_offline, conf_risk, conf_resource, sig_by_role)

    session.close()
    return _normalize(snapshot, id_map)


def _build_invalid_example(feat_sso, feat_offline, conf_risk, conf_resource, sig_by_role) -> dict:
    """Demo-only artifact: a build_now SSO decision that ignores both conflicts (rejected).

    Generator-owned (NOT produced by assemble_snapshot): it is built straight from the real
    decision-integrity validator to show the gate rejecting an unacknowledged-conflict decision.
    """

    invalid_ctx = Stage4Context(
        features=[FeatureForDecision(feature_id=feat_sso.id, title=feat_sso.title, jtbd=feat_sso.jtbd),
                  FeatureForDecision(feature_id=feat_offline.id, title=feat_offline.title, jtbd=feat_offline.jtbd)],
        signals=[SignalForDecision(signal_id=sig_by_role[r].id, stakeholder_type=sig_by_role[r].source_type.value,
                                   intent="(input)", claims=["(input)"]) for r in ("sig-01", "sig-02")],
        conflicts=[ConflictForDecision(conflict_id=conf_risk.id, subject_id=feat_sso.id, conflict_type="risk",
                                       severity=4, summary="Sales advocate vs Engineering risk_flag over SSO"),
                   ConflictForDecision(conflict_id=conf_resource.id, subject_id=feat_sso.id, conflict_type="resource",
                                       severity=3, summary="SSO vs mobile offline capacity")],
    )
    invalid_decision = DecisionContract(
        decision_id="d-invalid", subject_type="feature", subject_id=feat_sso.id, recommendation="build_now",
        title="Build Enterprise SSO now", priority_rank=1,
        rationale="Sales says the Acme deal is blocked, so we should just build SSO immediately.",
        acknowledged_conflict_ids=[], evidence_signal_ids=[sig_by_role["sig-01"].id],
        confidence={"score": 0.8})
    invalid_contract = DecisionSynthesisContract(
        decisions=[invalid_decision],
        model_meta=ModelMeta(model_id="claude-demo-stub", prompt_version=s4p.STAGE4_PROMPT_VERSION,
                             input_tokens=180, output_tokens=60, stop_reason="tool_use"))
    invalid_result = DecisionIntegrityValidator().validate(invalid_contract, invalid_ctx)
    return {
        "label": "Rejected: a decision that ignores a known conflict over its subject",
        "model_output": json.loads(invalid_decision.model_dump_json()),
        "validation": {
            "ok": invalid_result.ok,
            "retry_feedback": invalid_result.retry_feedback(),
            "issues": [{"code": i.code, "message": i.message, "field": i.field, "hint": i.hint}
                       for i in invalid_result.errors],
        },
    }


def _normalize(snapshot: dict, id_map: dict[str, str]) -> dict:
    """Remap entity ids -> stable role labels and pin timestamps (determinism)."""

    text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    for raw_id, label in id_map.items():
        text = text.replace(raw_id, label)
    text = re.sub(r'("created_at":\s*)"[^"]*"', r'\g<1>"' + _FIXED_TS + '"', text)
    snap = json.loads(text)

    # Several lists are derived from sets or relationship loads ordered by the now-replaced
    # random UUIDs. Sort every such id-list by its stable label so the artifact is
    # byte-deterministic (semantic-order lists -- claims, signals, etc. -- are left alone).
    snap["unassigned_signal_ids"].sort()
    for feat in snap["features"]:
        feat["source_signal_ids"].sort()
    for conf in snap["conflicts"]:
        conf["parties"].sort(key=lambda p: p["stance"])
        for party in conf["parties"]:
            party["evidence_signal_ids"].sort()
    for dec in snap["decisions"]:
        dec["evidence_signal_ids"].sort()
        dec["acknowledged_conflict_ids"].sort()
        dec["framework_citation_ids"].sort()
    for citations in snap["framework_citations"].values():
        citations.sort(key=lambda c: (-c["retrieval_score"], c["chunk_id"]))
    snap["framework_pool"].sort(key=lambda p: (-p["retrieval_score"], p["chunk_id"]))

    why = snap.get("why", {})
    for sig in why.get("signals", []):
        sig["reached_via"].sort()
    why.get("signals", []).sort(key=lambda s: s["id"])
    why.get("direct_evidence", []).sort(key=lambda e: e["signal_id"])
    why.get("conflicts", []).sort(key=lambda c: c["id"])
    for conf in why.get("conflicts", []):
        conf["parties"].sort(key=lambda p: p["stance"])
        for party in conf["parties"]:
            party["evidence_signal_ids"].sort()
    if why.get("subject_feature"):
        why["subject_feature"]["source_signal_ids"].sort()
    if "integrity" in why:
        why["integrity"]["unresolved_signal_ids"].sort()

    # The decision-integrity gate emits unacknowledged conflicts in set order; sort the
    # issues by their (now label-bearing) message and rebuild the retry feedback in that
    # stable order, mirroring ValidationResult.retry_feedback's exact format.
    val = snap["invalid_example"]["validation"]
    val["issues"].sort(key=lambda i: (i["code"], i["message"]))
    lines = ["Your previous tool output was rejected. Fix every issue below and call the tool "
             "again with corrected input:"]
    for idx, issue in enumerate(val["issues"], start=1):
        loc = f" (field: {issue['field']})" if issue["field"] else ""
        hint = f" Hint: {issue['hint']}" if issue["hint"] else ""
        lines.append(f"{idx}. [{issue['code']}]{loc} {issue['message']}.{hint}")
    val["retry_feedback"] = "\n".join(lines)
    return snap


def _leftover_uuids(snapshot: dict) -> list[str]:
    return sorted(set(_UUID_RE.findall(json.dumps(snapshot))))


def main() -> None:
    snapshot = build_snapshot()
    leftovers = _leftover_uuids(snapshot)
    out_dir = _REPO_ROOT / "showcase" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_snapshot.json"
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"signals={len(snapshot['signals'])} features={len(snapshot['features'])} "
          f"conflicts={len(snapshot['conflicts'])} decisions={len(snapshot['decisions'])} "
          f"pool={len(snapshot['framework_pool'])}")
    print(f"dec-01 citations={snapshot['framework_citations'].get('dec-01')}")
    print(f"why.integrity.complete={snapshot['why']['integrity']['complete']}")
    print(f"invalid_example.ok={snapshot['invalid_example']['validation']['ok']} "
          f"issues={[i['code'] for i in snapshot['invalid_example']['validation']['issues']]}")
    if leftovers:
        print(f"WARNING: leftover raw UUIDs (non-deterministic): {leftovers}")
    else:
        print("OK: no leftover UUIDs -- snapshot is deterministic")


if __name__ == "__main__":
    main()
