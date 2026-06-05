"""Feature API routes (PART 6).

POST /features/extract  -- cluster analyzed signals into features
GET  /features/{id}     -- retrieve a feature
GET  /features          -- list features (optional ?signal_id / ?workspace_id)

Dependencies are defined here (self-contained) so the existing signal deps module is
untouched. The Stage 2 runner is read from ``app.state.stage2_runner`` (wired at
startup; 503 if absent).
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.api.schemas import RunMeta
from app.api.schemas_feature import (
    FeatureExtractRequest,
    FeatureExtractResponse,
    FeatureResponse,
)
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.feature_service import FeatureService
from app.stages.stage2.runner import Stage2FeatureRunner

router = APIRouter(prefix="/features", tags=["features"])


def get_stage2_runner(request: Request) -> Stage2FeatureRunner:
    runner = getattr(request.app.state, "stage2_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feature extraction is not configured.",
        )
    return runner


def get_feature_service(
    session: Session = Depends(get_db),
    runner: Stage2FeatureRunner = Depends(get_stage2_runner),
) -> FeatureService:
    return FeatureService(
        session=session,
        feature_repo=FeatureRepository(session),
        feature_signal_repo=FeatureSignalRepository(session),
        parsed_repo=ParsedSignalRepository(session),
        signal_repo=SignalRepository(session),
        runner=runner,
    )


@router.post("/extract", response_model=FeatureExtractResponse)
def extract_features(
    body: FeatureExtractRequest,
    service: FeatureService = Depends(get_feature_service),
) -> FeatureExtractResponse:
    """Cluster the given analyzed signals into normalized features."""

    result = service.extract_features(body.signal_ids, workspace_id=body.workspace_id)
    return FeatureExtractResponse(
        features=[FeatureResponse.from_row(f) for f in result.features],
        unassigned_signal_ids=result.unassigned_signal_ids,
        run=RunMeta.from_result(result.stage_result),
    )


@router.get("/{feature_id}", response_model=FeatureResponse)
def get_feature(
    feature_id: uuid.UUID,
    service: FeatureService = Depends(get_feature_service),
) -> FeatureResponse:
    """Retrieve a feature by id."""

    return FeatureResponse.from_row(service.get_feature(feature_id))


@router.get("", response_model=list[FeatureResponse])
def list_features(
    signal_id: uuid.UUID | None = Query(default=None),
    workspace_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    service: FeatureService = Depends(get_feature_service),
) -> list[FeatureResponse]:
    """List features, optionally filtered by linked signal or workspace."""

    features = service.list_features(workspace_id=workspace_id, signal_id=signal_id, limit=limit)
    return [FeatureResponse.from_row(f) for f in features]
