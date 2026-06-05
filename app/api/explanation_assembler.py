"""Pure assembler for Decision Explainability (Phase 5).

Transforms the *already-loaded* provenance rows of a decision into a
:class:`DecisionExplanationResponse`. It is the read-side analog of the project's
deterministic validation gates: a pure function over its inputs, with no database
access, no LLM, and no writes. Given identical inputs it produces an identical
output regardless of input ordering -- the basis of the determinism tests.

Responsibilities (per the approved design):

* **Signal deduplication** -- a signal reached via direct evidence *and* the feature
  *and* a conflict appears exactly once in ``signals``.
* **reached_via tagging** -- each signal records every path by which it backs the
  decision (``direct_evidence`` | ``feature:<id>`` | ``conflict:<id>``).
* **quoted_text generation** -- each claim carries ``raw_text[span.start:span.end]``.
* **Deterministic ordering** -- every collection is sorted by a stable key, so the
  output does not depend on the order rows were loaded in.
* **explanation_integrity validation** -- the subject feature and every referenced
  signal id must resolve; unresolved references are reported, never dropped.

The assembler assumes its inputs are fully loaded (the service guarantees this via
eager-loading repositories); it never triggers a lazy load.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from app.ai_contracts.enums import SubjectType
from app.api.schemas import ConfidenceResponse
from app.api.schemas_decision_explanation import (
    AnalysisSummary,
    ClaimProvenance,
    ConflictExplanation,
    ConflictPartyExplanation,
    DecisionExplanationResponse,
    DecisionSummary,
    EvidenceLink,
    ExplanationMeta,
    FeatureSummary,
    IntegritySummary,
    SignalProvenance,
)
from app.models.conflict import Conflict
from app.models.decision import Decision
from app.models.feature import Feature
from app.models.signal import Signal

__all__ = ["ExplanationAssembler"]


def _confidence(blob: dict) -> ConfidenceResponse:
    """Build a :class:`ConfidenceResponse` from a stored confidence JSON blob."""

    return ConfidenceResponse(
        score=blob["score"],
        basis=blob["basis"],
        components=blob.get("components", {}),
    )


def _conflict_signal_ids(conflict: Conflict) -> set[uuid.UUID]:
    """All evidence signal ids referenced by a conflict's parties (JSON -> UUID)."""

    ids: set[uuid.UUID] = set()
    for party in conflict.parties:
        for raw in party.evidence_signal_ids:
            ids.add(uuid.UUID(str(raw)))
    return ids


class ExplanationAssembler:
    """Pure transformer: provenance rows -> :class:`DecisionExplanationResponse`."""

    @classmethod
    def assemble(
        cls,
        decision: Decision,
        subject_feature: Feature | None,
        conflicts: Sequence[Conflict],
        signals: Sequence[Signal],
    ) -> DecisionExplanationResponse:
        """Assemble the explanation. Inputs must be fully loaded; ordering-independent."""

        signal_by_id: dict[uuid.UUID, Signal] = {s.id: s for s in signals}
        conflicts_sorted = sorted(conflicts, key=lambda c: (-c.severity, c.id))

        # ---- the three provenance paths to a signal -----------------------------
        direct_ids = {edge.signal_id for edge in decision.evidence}
        feature_ids: set[uuid.UUID] = (
            {fs.signal_id for fs in subject_feature.feature_signals} if subject_feature is not None else set()
        )
        per_conflict_ids: dict[uuid.UUID, set[uuid.UUID]] = {
            c.id: _conflict_signal_ids(c) for c in conflicts_sorted
        }
        conflict_ids: set[uuid.UUID] = set()
        for ids in per_conflict_ids.values():
            conflict_ids |= ids

        all_referenced = direct_ids | feature_ids | conflict_ids
        resolved = {sid for sid in all_referenced if sid in signal_by_id}

        # ---- reached_via (deterministic: direct, then feature, then conflicts) ---
        def reached_via(sid: uuid.UUID) -> list[str]:
            tags: list[str] = []
            if sid in direct_ids:
                tags.append("direct_evidence")
            if subject_feature is not None and sid in feature_ids:
                tags.append(f"feature:{subject_feature.id}")
            for conflict in conflicts_sorted:
                if sid in per_conflict_ids[conflict.id]:
                    tags.append(f"conflict:{conflict.id}")
            return tags

        # ---- signals (de-duplicated, sorted, with analysis + quoted_text) -------
        signal_views: list[SignalProvenance] = []
        for sid in sorted(resolved):
            signal = signal_by_id[sid]
            signal_views.append(
                SignalProvenance(
                    id=signal.id,
                    source_type=signal.source_type,
                    source_ref=signal.source_ref,
                    raw_text=signal.raw_text,
                    created_at=signal.created_at,
                    analysis=cls._analysis(signal),
                    reached_via=reached_via(sid),
                )
            )

        # ---- direct evidence links (sorted by signal id) ------------------------
        direct_evidence = [
            EvidenceLink(signal_id=edge.signal_id, created_at=edge.created_at)
            for edge in sorted(decision.evidence, key=lambda e: e.signal_id)
        ]

        # ---- conflicts (+ parties), sorted deterministically --------------------
        conflict_views = [cls._conflict(c) for c in conflicts_sorted]

        # ---- subject feature ----------------------------------------------------
        feature_view = cls._feature(subject_feature) if subject_feature is not None else None

        # ---- integrity ----------------------------------------------------------
        integrity = cls._integrity(decision, subject_feature, all_referenced, resolved)

        return DecisionExplanationResponse(
            decision=cls._decision(decision),
            subject_feature=feature_view,
            direct_evidence=direct_evidence,
            conflicts=conflict_views,
            signals=signal_views,
            integrity=integrity,
            meta=ExplanationMeta(
                decision_id=decision.id,
                signal_count=len(signal_views),
                conflict_count=len(conflict_views),
            ),
        )

    # ----------------------------------------------------------------- builders
    @staticmethod
    def _decision(decision: Decision) -> DecisionSummary:
        return DecisionSummary(
            id=decision.id,
            workspace_id=decision.workspace_id,
            subject_type=decision.subject_type,
            subject_id=decision.subject_id,
            recommendation=decision.recommendation,
            title=decision.title,
            rationale=decision.rationale,
            priority_rank=decision.priority_rank,
            status=decision.status,
            confidence=_confidence(decision.confidence),
            created_at=decision.created_at,
        )

    @staticmethod
    def _feature(feature: Feature) -> FeatureSummary:
        return FeatureSummary(
            id=feature.id,
            title=feature.title,
            jtbd=feature.jtbd,
            status=feature.status,
            confidence=_confidence(feature.confidence),
            source_signal_ids=sorted(fs.signal_id for fs in feature.feature_signals),
        )

    @staticmethod
    def _conflict(conflict: Conflict) -> ConflictExplanation:
        parties = [
            ConflictPartyExplanation(
                stakeholder_type=party.stakeholder_type,
                stance=party.stance,
                summary=party.summary,
                evidence_signal_ids=[uuid.UUID(str(s)) for s in party.evidence_signal_ids],
            )
            for party in sorted(conflict.parties, key=lambda p: (p.stakeholder_type.value, p.id))
        ]
        return ConflictExplanation(
            id=conflict.id,
            conflict_type=conflict.conflict_type,
            severity=conflict.severity,
            status=conflict.status,
            confidence=_confidence(conflict.confidence),
            parties=parties,
        )

    @staticmethod
    def _analysis(signal: Signal) -> AnalysisSummary | None:
        parsed = signal.parsed
        if parsed is None:
            return None
        claims = [
            ClaimProvenance(
                text=claim["text"],
                source_span=(claim["source_span"][0], claim["source_span"][1]),
                quoted_text=signal.raw_text[claim["source_span"][0]:claim["source_span"][1]],
                claim_confidence=claim["claim_confidence"],
            )
            for claim in sorted(parsed.extracted_claims, key=lambda c: (c["source_span"][0], c["source_span"][1]))
        ]
        return AnalysisSummary(
            intent=parsed.intent,
            stakeholder_type=parsed.stakeholder_type,
            urgency=parsed.urgency,
            sentiment=parsed.sentiment,
            confidence=_confidence(parsed.confidence),
            claims=claims,
        )

    @staticmethod
    def _integrity(
        decision: Decision,
        subject_feature: Feature | None,
        all_referenced: set[uuid.UUID],
        resolved: set[uuid.UUID],
    ) -> IntegritySummary:
        notes: list[str] = []
        subject_unresolved = decision.subject_type == SubjectType.FEATURE and subject_feature is None
        if subject_unresolved:
            notes.append(f"subject feature {decision.subject_id} did not resolve")

        unresolved = sorted(all_referenced - resolved)
        for sid in unresolved:
            notes.append(f"referenced signal {sid} did not resolve")

        complete = not subject_unresolved and not unresolved
        return IntegritySummary(complete=complete, unresolved_signal_ids=list(unresolved), notes=notes)
