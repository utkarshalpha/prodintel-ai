"""Decision API routes (PART 6).

POST /decisions/synthesize  -- synthesize evidence-backed decisions over features
GET  /decisions/{id}        -- retrieve a decision
GET  /decisions             -- list decisions (optional ?subject_id / ?workspace_id)

Dependencies are defined here (self-contained); the Stage 4 runner is read from
``app.state.stage4_runner`` (wired at startup; 503 if absent).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import RunMeta
from app.api.schemas_decision import (
    DecisionResponse,
    DecisionSynthesizeRequest,
    DecisionSynthesizeResponse,
)
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.services.decision_service import DecisionService
from app.stages.stage4.runner import Stage4DecisionRunner

router = APIRouter(prefix="/decisions", tags=["decisions"])


def get_stage4_runner(request: Request) -> Stage4DecisionRunner:
    runner = getattr(request.app.state, "stage4_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Decision synthesis is not configured.",
        )
    return runner


def get_decision_service(
    session: Session = Depends(get_db),
    runner: Stage4DecisionRunner = Depends(get_stage4_runner),
) -> DecisionService:
    return DecisionService(
        session=session,
        decision_repo=DecisionRepository(session),
        conflict_repo=ConflictRepository(session),
        feature_repo=FeatureRepository(session),
        parsed_repo=ParsedSignalRepository(session),
        runner=runner,
    )


@router.post("/synthesize", response_model=DecisionSynthesizeResponse)
def synthesize_decisions(
    body: DecisionSynthesizeRequest,
    service: DecisionService = Depends(get_decision_service),
) -> DecisionSynthesizeResponse:
    """Synthesize evidence-backed decisions over the given features."""

    result = service.synthesize_decisions(body.feature_ids, workspace_id=body.workspace_id)
    return DecisionSynthesizeResponse(
        decisions=[DecisionResponse.from_row(d) for d in result.decisions],
        run=RunMeta.from_result(result.stage_result),
    )


@router.get("/{decision_id}", response_model=DecisionResponse)
def get_decision(
    decision_id: uuid.UUID,
    service: DecisionService = Depends(get_decision_service),
) -> DecisionResponse:
    """Retrieve a decision by id."""

    return DecisionResponse.from_row(service.get_decision(decision_id))


@router.get("", response_model=list[DecisionResponse])
def list_decisions(
    subject_id: uuid.UUID | None = Query(default=None),
    workspace_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    service: DecisionService = Depends(get_decision_service),
) -> list[DecisionResponse]:
    """List decisions, optionally filtered by subject feature or workspace."""

    decisions = service.list_decisions(workspace_id=workspace_id, subject_id=subject_id, limit=limit)
    return [DecisionResponse.from_row(d) for d in decisions]
