"""Streamlit-free execution layer for Live Analysis mode.

Wires the existing architecture into one call -- it constructs nothing new of substance:

    FeedbackEntry[]  (from L4 ingestion)
      -> PipelineService.analyze   (the real saga over the five stage services)
         with LocalHeuristicClient behind every stage runner
      -> assemble_snapshot         (the single snapshot constructor)
      -> LiveRunResult             (a detached, ORM-free result the UI can render)

Why this layer exists (and is Streamlit-free):

* **No ORM leaks.** ``assemble_snapshot`` reads live ORM entities, so it is called *inside*
  the session scope here; only the resulting plain ``dict`` (plus plain-data run notes)
  escapes. The UI never holds a Session-bound object -- it cannot trip the session-lifetime
  contract.
* **Fresh in-memory DB per run.** Each analysis gets its own ``:memory:`` SQLite engine, so
  re-running identical feedback is never deduplicated against a previous run's signals
  (``create_signal`` dedupes by content hash) and nothing persists between runs.
* **Unit-testable.** No ``streamlit`` import; pure ``FeedbackEntry[] -> LiveRunResult``.

The ``import app.api.app`` below is the established mitigation for the
``DecisionExplanationService -> app.api`` import cycle (see PipelineService docstring): the
explanation service is injected, never built by the coordinator, and the API package must be
initialised before it is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.api.app  # noqa: F401  -- initialise the api package before the explanation service
import app.models  # noqa: F401  -- register tables on Base.metadata
from app.ai_clients.local_heuristic_client import LocalHeuristicClient
from app.db.base import Base
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.pipeline_service import FeedbackEntry, PipelineService  # noqa: F401  (FeedbackEntry re-exported for callers)
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage3.runner import build_stage3_runner
from app.stages.stage4.runner import build_stage4_runner

from showcase.lib.assemble import assemble_snapshot

__all__ = ["LiveRunResult", "StageNote", "run_local_analysis", "LIVE_SCENARIO"]

LIVE_SCENARIO = "Live analysis (local heuristic engine)"
_GENERATED_BY = "local-heuristic-engine"


@dataclass(frozen=True)
class StageNote:
    """One stage's outcome, flattened to plain data (no StageResult/ORM references)."""

    stage: str
    status: str
    detail: str | None
    error_code: str | None


@dataclass(frozen=True)
class LiveRunResult:
    """Detached, renderable result of one local analysis run.

    ``snapshot`` is exactly what the existing renderers consume (built by
    ``assemble_snapshot``). Everything here is plain data -- safe to stash in
    ``st.session_state`` and render across reruns.
    """

    snapshot: dict
    succeeded: bool
    failed_stage: str | None
    stage_notes: tuple[StageNote, ...]
    counts: dict


def _build_engine():
    """A fresh in-memory SQLite engine with FK enforcement and all tables created."""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _record):  # pragma: no cover - trivial
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


def run_local_analysis(
    entries: list[FeedbackEntry],
    *,
    scenario: str = LIVE_SCENARIO,
) -> LiveRunResult:
    """Run the full pipeline over ``entries`` with the local engine; return a detached result.

    Raises ``ValueError`` on empty ``entries`` (the same contract as
    ``PipelineService.analyze``). All known stage failures are captured in the result's
    ``stage_notes`` / ``failed_stage`` rather than raised.
    """

    engine = _build_engine()
    session: Session = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, class_=Session
    )()
    try:
        client = LocalHeuristicClient()
        explanation = DecisionExplanationService(
            DecisionRepository(session), FeatureRepository(session),
            ConflictRepository(session), SignalRepository(session),
        )
        service = PipelineService(
            session,
            stage1_runner=build_stage1_runner(client),
            stage2_runner=build_stage2_runner(client),
            stage3_runner=build_stage3_runner(client),
            stage4_runner=build_stage4_runner(client),
            explanation_service=explanation,
        )
        run = service.analyze(list(entries))

        # Serialize WHILE the session is open (assemble_snapshot reads live ORM entities).
        snapshot = assemble_snapshot(run, scenario=scenario, generated_by=_GENERATED_BY)
        notes = tuple(
            StageNote(o.stage.value, o.status.value, o.detail, o.error_code)
            for o in run.stage_outcomes
        )
        counts = {
            "signals": len(run.signals),
            "features": len(run.features),
            "conflicts": len(run.conflicts),
            "decisions": len(run.decisions),
        }
        return LiveRunResult(
            snapshot=snapshot,
            succeeded=run.succeeded,
            failed_stage=(run.failed_stage.value if run.failed_stage else None),
            stage_notes=notes,
            counts=counts,
        )
    finally:
        session.close()
        engine.dispose()
