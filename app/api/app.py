"""FastAPI application factory and error-to-HTTP mapping.

``create_app`` wires the signal router and registers exception handlers that turn
service errors into clean HTTP responses. The Stage 1 runner is expected on
``app.state.stage1_runner`` (configured at startup with a real Claude client, which
is out of scope for this layer; tests inject a fake).
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.middleware import CorrelationIdMiddleware
from app.api.routes_conflicts import router as conflicts_router
from app.api.routes_features import router as features_router
from app.api.routes_signals import router as signals_router
from app.observability.logging import get_logger
from app.services.errors import (
    AnalysisFailedError,
    AnalysisNotFoundError,
    ConflictDetectionFailedError,
    ConflictNotFoundError,
    FeatureExtractionFailedError,
    FeatureNotFoundError,
    SignalNotFoundError,
    SignalsNotAnalyzedError,
)

__all__ = ["create_app"]

_logger = get_logger("app.api.errors")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""

    app = FastAPI(title="ProdIntel AI - Signal Service", version="0.1.0")
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(signals_router)
    app.include_router(features_router)
    app.include_router(conflicts_router)
    _register_exception_handlers(app)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SignalNotFoundError)
    async def _signal_not_found(_request: Request, exc: SignalNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(AnalysisNotFoundError)
    async def _analysis_not_found(_request: Request, exc: AnalysisNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(AnalysisFailedError)
    async def _analysis_failed(_request: Request, exc: AnalysisFailedError) -> JSONResponse:
        result = exc.stage_result
        detail = {
            "message": str(exc),
            "signal_id": str(exc.signal_id),
            "status": result.status.value,
            "attempts_used": result.attempts_used,
            "error_code": result.error.code.value if result.error else None,
        }
        # 422: the stage ran but produced no valid, grounded result. Integer literal
        # avoids the renamed-constant deprecation warning across Starlette versions.
        _logger.warning(
            "api_analysis_failed",
            extra={
                "event": "validation_failed",
                "signal_id": detail["signal_id"],
                "status": detail["status"],
                "attempts_used": detail["attempts_used"],
                "error_code": detail["error_code"],
            },
        )
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(FeatureNotFoundError)
    async def _feature_not_found(_request: Request, exc: FeatureNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(SignalsNotAnalyzedError)
    async def _signals_not_analyzed(_request: Request, exc: SignalsNotAnalyzedError) -> JSONResponse:
        detail = {"message": str(exc), "signal_ids": [str(s) for s in exc.signal_ids]}
        _logger.warning("api_signals_not_analyzed", extra={"event": "validation_failed", "count": len(exc.signal_ids)})
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(FeatureExtractionFailedError)
    async def _extraction_failed(_request: Request, exc: FeatureExtractionFailedError) -> JSONResponse:
        result = exc.stage_result
        detail = {
            "message": str(exc),
            "status": result.status.value,
            "attempts_used": result.attempts_used,
            "error_code": result.error.code.value if result.error else None,
        }
        _logger.warning(
            "api_extraction_failed",
            extra={
                "event": "validation_failed",
                "status": detail["status"],
                "attempts_used": detail["attempts_used"],
                "error_code": detail["error_code"],
            },
        )
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(ConflictNotFoundError)
    async def _conflict_not_found(_request: Request, exc: ConflictNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(ConflictDetectionFailedError)
    async def _detection_failed(_request: Request, exc: ConflictDetectionFailedError) -> JSONResponse:
        result = exc.stage_result
        detail = {
            "message": str(exc),
            "status": result.status.value,
            "attempts_used": result.attempts_used,
            "error_code": result.error.code.value if result.error else None,
        }
        _logger.warning(
            "api_detection_failed",
            extra={
                "event": "validation_failed",
                "status": detail["status"],
                "attempts_used": detail["attempts_used"],
                "error_code": detail["error_code"],
            },
        )
        return JSONResponse(status_code=422, content={"detail": detail})
