"""FastAPI dependency wiring.

Builds a :class:`SignalService` per request from a session and the configured Stage
1 runner. The runner is read from ``app.state.stage1_runner`` so the Claude client
can be wired at startup (out of scope here) and overridden in tests -- the
application layer never constructs the client itself.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.signal_service import SignalService
from app.stages.stage1.runner import Stage1SignalRunner

__all__ = ["get_db", "get_stage1_runner", "get_signal_service"]


def get_stage1_runner(request: Request) -> Stage1SignalRunner:
    """Return the Stage 1 runner configured on the app, or 503 if unavailable."""

    runner = getattr(request.app.state, "stage1_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Signal analysis is not configured.",
        )
    return runner


def get_signal_service(
    session: Session = Depends(get_db),
    runner: Stage1SignalRunner = Depends(get_stage1_runner),
) -> SignalService:
    """Construct a request-scoped :class:`SignalService`."""

    return SignalService(
        session=session,
        signal_repo=SignalRepository(session),
        parsed_repo=ParsedSignalRepository(session),
        runner=runner,
    )
