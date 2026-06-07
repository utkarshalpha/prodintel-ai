"""DecisionService -- the application service for Stage 4 (Decision Synthesis).

Coordinates synthesizing evidence-backed decisions over a set of features, drawing
on the stakeholder signals behind them and the conflicts already detected over them,
plus retrieval and listing. Owns the transaction boundary and follows the hardened
three-phase pattern (Read -> AI -> Write) so **no database transaction is held
during the Claude call**.

Use cases (PART 5)
------------------
* :meth:`synthesize_decisions` -- run Stage 4 over features, their signals, and their
  conflicts; persist evidence-traceable decisions.
* :meth:`get_decision`         -- fetch a decision (with evidence + conflicts) or raise.
* :meth:`list_decisions`       -- list decisions, optionally by subject feature/workspace.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.ai_contracts.retrieval import RetrievalQuery
from app.models.conflict import Conflict
from app.models.decision import Decision
from app.observability.logging import get_logger, log_context
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.services.errors import (
    CorpusEmbeddingModelMismatchError,
    DecisionNotFoundError,
    DecisionSynthesisFailedError,
    FeatureNotFoundError,
    RetrievalFailedError,
    SignalsNotAnalyzedError,
)
from app.stages.stage4.prompts import (
    ConflictForDecision,
    FeatureForDecision,
    FrameworkPassage,
    SignalForDecision,
)
from app.stages.stage4.runner import Stage4Context, Stage4DecisionRunner
from app.vector.errors import EmbeddingDimensionMismatchError

if TYPE_CHECKING:  # typing only -- avoids importing the vector stack at module load
    from app.services.retrieval_service import RetrievalService

__all__ = ["DecisionService", "DecisionSynthesisResult", "FrameworkCitationPoolError"]

_logger = get_logger(__name__)
_STAGE = "stage4"


class FrameworkCitationPoolError(RuntimeError):
    """A validated framework citation id was absent from the frozen pool at persist time.

    The Stage 4 integrity gate guarantees every ``framework_citation_id`` is a member of
    the injected pool, so reaching this is an invariant breach (validator/pool divergence):
    fail loudly rather than persist a framework edge that is not grounded in what was
    actually retrieved. Internal error -- not a client-facing condition.
    """

    def __init__(self, chunk_id: uuid.UUID) -> None:
        self.chunk_id = chunk_id
        super().__init__(f"framework citation {chunk_id} is not in the retrieval pool")


@dataclass(frozen=True)
class DecisionSynthesisResult:
    """Outcome of a synthesis run: created decisions + the run's StageResult.

    ``framework_pool`` is the frozen, deterministically ordered set of framework
    passages the run grounded against (empty when no ``corpus_version`` was requested
    or retrieval degraded to nothing). It is the single source of truth shared by the
    prompt and the integrity validator; persisting the cited edges is Phase 7C-iii.
    """

    decisions: list[Decision]
    stage_result: object  # StageResult[DecisionSynthesisContract]
    framework_pool: tuple[FrameworkPassage, ...] = field(default_factory=tuple)


class DecisionService:
    """Application service for decision synthesis and management."""

    def __init__(
        self,
        session: Session,
        decision_repo: DecisionRepository,
        conflict_repo: ConflictRepository,
        feature_repo: FeatureRepository,
        parsed_repo: ParsedSignalRepository,
        runner: Stage4DecisionRunner,
        retrieval_service: "RetrievalService | None" = None,
    ) -> None:
        self._session = session
        self._decisions = decision_repo
        self._conflicts = conflict_repo
        self._features = feature_repo
        self._parsed = parsed_repo
        self._runner = runner
        # Optional: only consulted when a caller opts into framework grounding by
        # passing a corpus_version. Absent it, synthesis behaves exactly as before.
        self._retrieval = retrieval_service

    # -------------------------------------------------------------- synthesize
    def synthesize_decisions(
        self,
        feature_ids: Sequence[uuid.UUID],
        *,
        workspace_id: uuid.UUID | None = None,
        corpus_version: str | None = None,
    ) -> DecisionSynthesisResult:
        """Synthesize evidence-backed decisions over the given features.

        When ``corpus_version`` is given (opt-in framework grounding), Phase 1.5 runs
        framework-aware retrieval per feature, builds one frozen, deterministically
        ordered pool, and injects it into the Stage 4 context so the model may ground
        decisions on it and the integrity gate can reject any fabricated citation.
        Absent ``corpus_version`` the flow is identical to the framework-free Stage 4.

        Raises :class:`FeatureNotFoundError` if a feature is missing,
        :class:`SignalsNotAnalyzedError` if a backing signal lacks analysis,
        :class:`DecisionSynthesisFailedError` if Stage 4 produced no valid,
        evidence-traceable result (nothing is persisted in that case), and -- fail-fast,
        before the model call -- :class:`CorpusEmbeddingModelMismatchError` /
        :class:`EmbeddingDimensionMismatchError` if the corpus is misconfigured. A
        transient :class:`RetrievalFailedError` degrades gracefully (that feature
        contributes no framework knowledge) rather than failing the run.
        """

        # ---- Phase 1: READ -- gather features, their analyses, and conflicts. ---
        features = []
        signal_ids: list[uuid.UUID] = []
        seen_signals: set[uuid.UUID] = set()
        for feature_id in feature_ids:
            feature = self._features.get(feature_id)
            if feature is None:
                self._session.rollback()
                raise FeatureNotFoundError(feature_id)
            features.append(feature)
            for link in feature.feature_signals:
                if link.signal_id not in seen_signals:
                    seen_signals.add(link.signal_id)
                    signal_ids.append(link.signal_id)

        parsed_rows = {sid: self._parsed.get_by_signal_id(sid) for sid in signal_ids}
        missing = [sid for sid, row in parsed_rows.items() if row is None]
        if missing:
            self._session.rollback()
            raise SignalsNotAnalyzedError(missing)

        # ALL conflicts already detected over any of the subject features. This must
        # not truncate: the decision-integrity gate requires every conflict over a
        # subject to be acknowledged (ADR-012), so a bounded listing would let a
        # decision silently omit real conflicts. list_by_subjects() is unbounded.
        conflicts: list[Conflict] = list(
            self._conflicts.list_by_subjects([feature.id for feature in features])
        )

        feature_views = [
            FeatureForDecision(feature_id=feature.id, title=feature.title, jtbd=feature.jtbd)
            for feature in features
        ]
        signal_views = [
            SignalForDecision(
                signal_id=row.signal_id,
                stakeholder_type=row.stakeholder_type.value,
                intent=row.intent,
                claims=[claim["text"] for claim in row.extracted_claims],
            )
            for row in parsed_rows.values()
        ]
        conflict_views = [
            ConflictForDecision(
                conflict_id=conflict.id,
                subject_id=conflict.subject_id,
                conflict_type=conflict.conflict_type.value,
                severity=conflict.severity,
                summary=_summarize_conflict(conflict),
            )
            for conflict in conflicts
        ]

        # ---- Phase 1.5: RETRIEVAL -- build the frozen framework pool (opt-in). ---
        # Runs after the view DTOs are built (so ORM rows expiring on a retrieve-internal
        # commit is harmless) and before the commit-before-Claude below. RetrievalService
        # itself never holds a DB transaction across its network calls (ADR-010); a
        # fail-fast corpus error rolls back and propagates from within the helper.
        framework_pool: tuple[FrameworkPassage, ...] = ()
        if corpus_version is not None:
            if self._retrieval is None:
                self._session.rollback()
                raise RuntimeError(
                    "corpus_version was requested but DecisionService has no RetrievalService configured"
                )
            framework_pool = self._retrieve_framework_pool(corpus_version, feature_views)

        self._session.commit()  # release the connection before the AI call (and after retrieval)

        # ---- Phase 2: AI EXECUTION -- no DB transaction/connection held. --------
        with log_context(feature_count=len(feature_views), conflict_count=len(conflict_views)):
            _logger.info(
                "stage_started",
                extra={
                    "event": "stage_started",
                    "stage": _STAGE,
                    "feature_count": len(feature_views),
                    "conflict_count": len(conflict_views),
                    "framework_pool_size": len(framework_pool),
                },
            )
            context = Stage4Context(
                features=feature_views,
                signals=signal_views,
                conflicts=conflict_views,
                framework_knowledge=list(framework_pool),
                workspace_id=workspace_id,
            )
            result = self._runner.run(context)
            self._log_stage_attempts(result)

            if not result.succeeded or result.output is None:
                _logger.warning(
                    "stage_failed",
                    extra={
                        "event": "stage_failed",
                        "stage": _STAGE,
                        "status": result.status.value,
                        "error_code": result.error.code.value if result.error else None,
                        "attempts": result.attempts_used,
                    },
                )
                raise DecisionSynthesisFailedError(result)

            # ---- Phase 3: WRITE -- fresh transaction. ---------------------------
            model_meta = result.output.model_meta.model_dump(mode="json")
            # The frozen pool (7C-ii) is the single source of truth for framework
            # grounding: resolve every decision's cited ids against it up-front (dedupe +
            # score lookup) so a divergence fails before any row is written, never leaving
            # a partial set persisted.
            pool_scores = {passage.chunk_id: passage.retrieval_score for passage in framework_pool}
            resolved_citations = [
                self._resolve_framework_citations(item.framework_citation_ids, pool_scores)
                for item in result.output.decisions
            ]
            created: list[Decision] = []
            for item, framework_citations in zip(result.output.decisions, resolved_citations):
                decision = Decision.from_contract(item, model_meta, workspace_id=workspace_id)
                self._decisions.create_with_links(
                    decision,
                    item.evidence_signal_ids,
                    item.acknowledged_conflict_ids,
                    framework_citations,
                )
                created.append(decision)
            self._session.commit()
            _logger.info(
                "db_write",
                extra={"event": "db_write", "entity": "decision", "count": len(created)},
            )
            _logger.info(
                "stage_succeeded",
                extra={
                    "event": "stage_succeeded",
                    "stage": _STAGE,
                    "decision_count": len(created),
                    "attempts": result.attempts_used,
                    "input_tokens": result.metrics.total_input_tokens,
                    "output_tokens": result.metrics.total_output_tokens,
                },
            )
        return DecisionSynthesisResult(
            decisions=created, stage_result=result, framework_pool=framework_pool
        )

    # ----------------------------------------------------------- Phase 1.5 pool
    def _retrieve_framework_pool(
        self,
        corpus_version: str,
        feature_views: Sequence[FeatureForDecision],
    ) -> tuple[FrameworkPassage, ...]:
        """Retrieve per feature and reduce to one frozen, deterministic framework pool.

        For each feature the query text is ``title`` + ``jtbd``. Citations are unioned
        across features and deduplicated by ``chunk_id`` keeping the **highest**
        ``retrieval_score`` (a max over a set -- order-independent), then ordered by
        ``(-score, chunk_id)`` -- a total order on the deduped set. The result is a
        pure function of the deduped (chunk_id, max_score) set, so feature iteration
        order changes neither the contents nor the persisted scores.

        Degradation vs. fail-fast: a per-feature :class:`RetrievalFailedError` (transient
        embed/vector failure) is logged and skipped -- that feature contributes nothing.
        A :class:`CorpusEmbeddingModelMismatchError` / :class:`EmbeddingDimensionMismatchError`
        is a corpus misconfiguration: roll back and propagate so the whole run fails fast
        before the model call (nothing is persisted).
        """

        _logger.info(
            "retrieval_started",
            extra={
                "event": "retrieval_started",
                "stage": _STAGE,
                "corpus_version": corpus_version,
                "feature_count": len(feature_views),
            },
        )

        best: dict[uuid.UUID, "RetrievalCitation"] = {}  # noqa: F821 -- forward ref in annotation
        degraded = False
        try:
            for feature in feature_views:
                query = RetrievalQuery(
                    text=f"{feature.title}\n{feature.jtbd}", corpus_version=corpus_version
                )
                try:
                    result = self._retrieval.retrieve(query)  # type: ignore[union-attr]
                except RetrievalFailedError as exc:
                    degraded = True
                    _logger.warning(
                        "retrieval_degraded",
                        extra={
                            "event": "retrieval_degraded",
                            "stage": _STAGE,
                            "corpus_version": corpus_version,
                            "feature_id": str(feature.feature_id),
                            "reason": str(exc),
                        },
                    )
                    continue
                for citation in result.citations:
                    # Guard against a non-finite score (NaN/inf) from a degenerate
                    # embedding: NaN defeats the max comparison ("NaN > x" is always
                    # False) and makes the final (-score, ...) sort non-deterministic,
                    # breaking both pool invariants. Drop it so corruption never enters
                    # the pool, the ordering, or the rendered prompt.
                    if not math.isfinite(citation.score):
                        degraded = True
                        _logger.warning(
                            "retrieval_score_dropped",
                            extra={
                                "event": "retrieval_score_dropped",
                                "stage": _STAGE,
                                "corpus_version": corpus_version,
                                "chunk_id": str(citation.chunk_id),
                            },
                        )
                        continue
                    current = best.get(citation.chunk_id)
                    if current is None or citation.score > current.score:
                        best[citation.chunk_id] = citation
        except (CorpusEmbeddingModelMismatchError, EmbeddingDimensionMismatchError) as exc:
            self._session.rollback()
            _logger.error(
                "retrieval_failed",
                extra={
                    "event": "retrieval_failed",
                    "stage": _STAGE,
                    "corpus_version": corpus_version,
                    "error": type(exc).__name__,
                },
            )
            raise

        pool = tuple(
            FrameworkPassage(
                chunk_id=citation.chunk_id,
                content=citation.content,
                framework=citation.framework.value if citation.framework else None,
                source_title=citation.source_title,
                retrieval_score=citation.score,
            )
            for citation in sorted(best.values(), key=lambda c: (-c.score, str(c.chunk_id)))
        )
        _logger.info(
            "retrieval_succeeded",
            extra={
                "event": "retrieval_succeeded",
                "stage": _STAGE,
                "corpus_version": corpus_version,
                "pool_size": len(pool),
                "degraded": degraded,
            },
        )
        return pool

    def _resolve_framework_citations(
        self,
        citation_ids: Sequence[uuid.UUID],
        pool_scores: Mapping[uuid.UUID, float | None],
    ) -> list[tuple[uuid.UUID, float | None]]:
        """Dedupe cited chunk ids (``dict.fromkeys``) and resolve each to its pool score.

        The frozen pool is the source of truth: every cited id must be a pool member, and
        the persisted ``retrieval_score`` is taken from the pool (never recomputed). A miss
        is an invariant breach -- the integrity gate should have rejected it -- so raise
        :class:`FrameworkCitationPoolError` before any write rather than persist an
        ungrounded edge. Returns ``[(chunk_id, retrieval_score), ...]`` with unique ids.
        """

        resolved: list[tuple[uuid.UUID, float | None]] = []
        for chunk_id in dict.fromkeys(citation_ids):
            if chunk_id not in pool_scores:
                raise FrameworkCitationPoolError(chunk_id)
            resolved.append((chunk_id, pool_scores[chunk_id]))
        return resolved

    # --------------------------------------------------------------------- get
    def get_decision(self, decision_id: uuid.UUID) -> Decision:
        """Return a decision or raise :class:`DecisionNotFoundError`."""

        decision = self._decisions.get(decision_id)
        if decision is None:
            raise DecisionNotFoundError(decision_id)
        return decision

    def list_decisions(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        subject_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Decision]:
        """List decisions, optionally filtered by subject feature or workspace."""

        return self._decisions.list(workspace_id=workspace_id, subject_id=subject_id, limit=limit)

    # ----------------------------------------------------------------- logging
    @staticmethod
    def _log_stage_attempts(result: object) -> None:
        """Emit one structured log per failed attempt (validation/retry events)."""

        for attempt in result.metrics.attempts:  # type: ignore[attr-defined]
            if attempt.error_code is not None:
                _logger.warning(
                    "stage_retry",
                    extra={
                        "event": "retry_event",
                        "stage": _STAGE,
                        "attempt": attempt.attempt,
                        "error_code": attempt.error_code.value,
                    },
                )


def _summarize_conflict(conflict: Conflict) -> str:
    """Render a one-line, human-readable summary of a conflict from its parties."""

    parties = "; ".join(
        f"{party.stakeholder_type.value} ({party.stance.value}): {party.summary}"
        for party in conflict.parties
    )
    return f"{conflict.conflict_type.value} conflict (severity {conflict.severity}) -- {parties}"
