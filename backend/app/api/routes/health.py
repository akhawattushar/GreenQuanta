"""Health and service-capability endpoints (public)."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import get_db
from app.core.config import get_settings
from app.schemas.models import HealthResponse
from app.services import model_registry, reporting

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency check")
def health() -> HealthResponse:
    settings = get_settings()
    model_status = model_registry.status()
    try:
        db_ok = get_db().healthy()
    except Exception:  # noqa: BLE001 - health must never raise
        db_ok = False
    healthy = db_ok and model_status["loaded"]
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        environment=settings.app_env,
        database=db_ok,
        model_loaded=bool(model_status["loaded"]),
        model_error=(model_registry.MODEL_HEALTH_FAILED_MESSAGE if not model_status["loaded"] else None),
        models={
            "legacy": {"loaded": bool(model_status["loaded"]),
                       "error": (model_registry.MODEL_HEALTH_FAILED_MESSAGE
                                 if not model_status["loaded"] else None)},
            "fuelcast_xgboost": model_registry.fuelcast_status(),
            "fuelcast_vqr": model_registry.vqr_status(),
        },
        pdf_export=reporting.pdf_available(),
    )
