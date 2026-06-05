"""Signal API routes (PART 3).

POST /signals               -- create (idempotent; 201 new, 200 existing)
GET  /signals/{id}          -- retrieve a signal
POST /signals/{id}/analyze  -- run Stage 1 and persist a grounded analysis
GET  /signals/{id}/analysis -- retrieve the persisted analysis

Service errors are translated to HTTP status codes by exception handlers registered
in :mod:`app.api.app`, so route bodies stay focused on the happy path.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Response, status

from app.api.deps import get_signal_service
from app.api.schemas import (
    AnalysisResponse,
    AnalyzeResponse,
    RunMeta,
    SignalCreateRequest,
    SignalResponse,
)
from app.services.signal_service import SignalService

router = APIRouter(prefix="/signals", tags=["signals"])


@router.post("", response_model=SignalResponse, status_code=status.HTTP_201_CREATED)
def create_signal(
    body: SignalCreateRequest,
    response: Response,
    service: SignalService = Depends(get_signal_service),
) -> SignalResponse:
    """Create a signal, or return the existing identical one (200)."""

    result = service.create_signal(
        source_type=body.source_type,
        raw_text=body.raw_text,
        source_ref=body.source_ref,
        workspace_id=body.workspace_id,
    )
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return SignalResponse.from_row(result.signal)


@router.get("/{signal_id}", response_model=SignalResponse)
def get_signal(
    signal_id: uuid.UUID,
    service: SignalService = Depends(get_signal_service),
) -> SignalResponse:
    """Retrieve a signal by id."""

    return SignalResponse.from_row(service.get_signal(signal_id))


@router.post("/{signal_id}/analyze", response_model=AnalyzeResponse)
def analyze_signal(
    signal_id: uuid.UUID,
    service: SignalService = Depends(get_signal_service),
) -> AnalyzeResponse:
    """Analyze a signal with Stage 1 and persist the grounded result."""

    result = service.analyze_signal(signal_id)
    return AnalyzeResponse(
        analysis=AnalysisResponse.from_row(result.parsed_signal),
        run=RunMeta.from_result(result.stage_result),
    )


@router.get("/{signal_id}/analysis", response_model=AnalysisResponse)
def get_analysis(
    signal_id: uuid.UUID,
    service: SignalService = Depends(get_signal_service),
) -> AnalysisResponse:
    """Retrieve the persisted Stage 1 analysis for a signal."""

    return AnalysisResponse.from_row(service.get_analysis(signal_id))
