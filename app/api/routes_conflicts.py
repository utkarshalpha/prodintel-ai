"""Conflict API routes (PART 6).

POST /conflicts/detect  -- detect conflicts among the stakeholders behind features
GET  /conflicts/{id}    -- retrieve a conflict
GET  /conflicts         -- list conflicts (optional ?subject_id / ?workspace_id)

Dependencies are defined here (self-contained); the Stage 3 runner is read from
``app.state.stage3_runner`` (wired at startup; 503 if absent).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import RunMeta
from app.api.schemas_conflict import (
    ConflictDetectRequest,
    ConflictDetectResponse,
    ConflictResponse,
)
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.services.conflict_service import ConflictService
from app.stages.stage3.runner import Stage3ConflictRunner

router = APIRouter(prefix="/conflicts", tags=["conflicts"])


def get_stage3_runner(request: Request) -> Stage3ConflictRunner:
    runner = getattr(request.app.state, "stage3_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conflict detection is not configured.",
        )
    return runner


def get_conflict_service(
    session: Session = Depends(get_db),
    runner: Stage3ConflictRunner = Depends(get_stage3_runner),
) -> ConflictService:
    return ConflictService(
        session=session,
        conflict_repo=ConflictRepository(session),
        feature_repo=FeatureRepository(session),
        parsed_repo=ParsedSignalRepository(session),
        runner=runner,
    )


@router.post("/detect", response_model=ConflictDetectResponse)
def detect_conflicts(
    body: ConflictDetectRequest,
    service: ConflictService = Depends(get_conflict_service),
) -> ConflictDetectResponse:
    """Detect conflicts among the stakeholders behind the given features."""

    result = service.detect_conflicts(body.feature_ids, workspace_id=body.workspace_id)
    return ConflictDetectResponse(
        conflicts=[ConflictResponse.from_row(c) for c in result.conflicts],
        run=RunMeta.from_result(result.stage_result),
    )


@router.get("/{conflict_id}", response_model=ConflictResponse)
def get_conflict(
    conflict_id: uuid.UUID,
    service: ConflictService = Depends(get_conflict_service),
) -> ConflictResponse:
    """Retrieve a conflict by id."""

    return ConflictResponse.from_row(service.get_conflict(conflict_id))


@router.get("", response_model=list[ConflictResponse])
def list_conflicts(
    subject_id: uuid.UUID | None = Query(default=None),
    workspace_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    service: ConflictService = Depends(get_conflict_service),
) -> list[ConflictResponse]:
    """List conflicts, optionally filtered by subject feature or workspace."""

    conflicts = service.list_conflicts(workspace_id=workspace_id, subject_id=subject_id, limit=limit)
    return [ConflictResponse.from_row(c) for c in conflicts]
