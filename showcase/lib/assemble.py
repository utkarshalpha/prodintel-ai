"""assemble_snapshot -- the single source of truth for snapshot construction.

Transforms a :class:`~app.services.pipeline_service.PipelineRunResult` (domain objects)
into the exact snapshot schema the Streamlit showcase already renders. Both the demo
generator (``scripts/build_demo_snapshot.py``) and the future Live Analysis mode use this
function, so snapshot-building logic exists in exactly one place.

Strict separation of concerns:

* ``PipelineService`` returns domain objects; it performs no serialization.
* ``assemble_snapshot`` performs the presentation-oriented transformation only.
* ``invalid_example`` is a demo-only artifact (built from the integrity validator) and is
  **not** produced here -- the generator owns it.
* The schema is preserved exactly, including known inconsistencies (``framework_citations``
  is a map keyed by decision id while the other sections are lists; ``why`` carries the
  verbatim ``/why`` payload), to avoid any renderer churn.

Session-scope contract: ``run_result`` carries live ORM entities, so call this while the
producing session is still open (relationships are read lazily). The demo generator and an
in-request API/Streamlit run both satisfy this.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # typing only -- assemble imports nothing from app at runtime
    from app.services.pipeline_service import PipelineRunResult

__all__ = ["assemble_snapshot"]

SNAPSHOT_SCHEMA_VERSION = "1.0"


def _conf(d: dict) -> dict:
    return {"score": d.get("score"), "basis": d.get("basis"), "components": d.get("components", {})}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _claims(parsed, raw_text: str) -> list[dict]:
    out = []
    for c in parsed.extracted_claims:
        start, end = c["source_span"]
        out.append({"text": c["text"], "source_span": [start, end],
                    "quoted_text": raw_text[start:end], "claim_confidence": c["claim_confidence"]})
    return out


def assemble_snapshot(
    run_result: "PipelineRunResult",
    *,
    scenario: str | None = None,
    generated_by: str | None = None,
    note: str | None = None,
    include_why: bool = True,
) -> dict:
    """Build the showcase snapshot dict from a pipeline run (real ids/timestamps).

    Does not emit ``invalid_example`` (generator-owned) and does not remap ids or pin
    timestamps (deterministic normalization is the generator's concern). ``why`` is the
    pre-computed explanation for the rank-1 decision, or ``None`` if the run produced none.
    """

    raw_by_id = {s.id: s.raw_text for s in run_result.signals}

    snapshot: dict = {
        "meta": {
            "scenario": scenario,
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "generated_by": generated_by,
            "note": note,
        },
        "signals": [
            {"id": str(s.id), "source_type": s.source_type.value, "raw_text": s.raw_text,
             "source_ref": s.source_ref, "created_at": _iso(s.created_at), "content_hash": s.content_hash}
            for s in run_result.signals
        ],
        "analyzed_signals": [
            {"signal_id": str(p.signal_id), "intent": p.intent, "stakeholder_type": p.stakeholder_type.value,
             "urgency": p.urgency, "sentiment": p.sentiment, "confidence": _conf(p.confidence),
             "claims": _claims(p, raw_by_id.get(p.signal_id, ""))}
            for p in run_result.parsed_signals
        ],
        "features": [
            {"id": str(f.id), "title": f.title, "jtbd": f.jtbd, "description": f.description,
             "status": f.status.value, "confidence": _conf(f.confidence),
             "source_signal_ids": [str(fs.signal_id) for fs in f.feature_signals]}
            for f in run_result.features
        ],
        "unassigned_signal_ids": [str(sid) for sid in run_result.unassigned_signal_ids],
        "conflicts": [
            {"id": str(c.id), "conflict_type": c.conflict_type.value, "severity": c.severity,
             "status": c.status.value, "subject_type": c.subject_type.value, "subject_id": str(c.subject_id),
             "confidence": _conf(c.confidence),
             "parties": [{"stakeholder_type": party.stakeholder_type.value, "stance": party.stance.value,
                          "summary": party.summary,
                          "evidence_signal_ids": [str(s) for s in party.evidence_signal_ids]}
                         for party in c.parties]}
            for c in run_result.conflicts
        ],
        "decisions": [],
        "framework_pool": [
            {"chunk_id": str(p.chunk_id), "framework": p.framework, "source_title": p.source_title,
             "content": p.content, "retrieval_score": p.retrieval_score}
            for p in run_result.framework_pool
        ],
        "framework_citations": {},
        "why": (json.loads(run_result.explanations[0].model_dump_json())
                if include_why and run_result.explanations else None),
    }

    for dec in run_result.decisions:
        snapshot["decisions"].append({
            "id": str(dec.id), "subject_id": str(dec.subject_id), "recommendation": dec.recommendation.value,
            "title": dec.title, "rationale": dec.rationale, "priority_rank": dec.priority_rank,
            "status": dec.status.value, "confidence": _conf(dec.confidence),
            "evidence_signal_ids": [str(e.signal_id) for e in dec.evidence],
            "acknowledged_conflict_ids": [str(a.conflict_id) for a in dec.acknowledged_conflicts],
            "framework_citation_ids": [str(fc.chunk_id) for fc in dec.framework_citations]})
        snapshot["framework_citations"][str(dec.id)] = [
            {"chunk_id": str(fc.chunk_id), "retrieval_score": fc.retrieval_score,
             "relationship_type": fc.relationship_type.value, "evidence_type": fc.evidence_type.value}
            for fc in dec.framework_citations]

    return snapshot
