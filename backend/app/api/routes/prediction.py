"""Fuel prediction endpoints — backed by the real trained artifacts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbDep, WriteUser
from app.core.config import get_settings
from app.db.database import RunRepository
from app.schemas.models import ModelInfoResponse, PredictionRequest, PredictionResponse
from app.services.evaluator import Environment, VesselState, VoyagePlan, feature_row
from app.services.model_registry import (
    FeatureValidationError,
    ModelUnavailableError,
    evaluate_holdout,
    get_bundle,
    status as model_status,
)

router = APIRouter(prefix="/prediction", tags=["prediction"])


@router.get("/model", response_model=ModelInfoResponse, summary="Loaded model metadata and measured metrics")
def model_info(_: CurrentUser) -> ModelInfoResponse:
    info = model_status()
    if not info["loaded"]:
        return ModelInfoResponse(loaded=False, error=info["error"])
    metadata = {k: v for k, v in info.items() if k not in {"loaded", "error"}}
    try:
        metrics = evaluate_holdout()
    except ModelUnavailableError as exc:
        metrics = {"available": False, "reason": str(exc)}
    return ModelInfoResponse(loaded=True, metadata=metadata, metrics=metrics)


@router.post("/fuel", response_model=PredictionResponse, summary="Predict fuel consumption rate")
def predict_fuel(payload: PredictionRequest, user: WriteUser, db: DbDep) -> PredictionResponse:
    settings = get_settings()

    try:
        bundle = get_bundle()
    except ModelUnavailableError as exc:
        # Explicit failure. The API never substitutes a placeholder number here.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The trained model is unavailable, so no prediction can be returned. {exc}",
        ) from exc

    plan = VoyagePlan(
        distance_nm=payload.distance_nm or 1.0,
        speed_knots=payload.sailing_speed,
        vessel=VesselState(**payload.vessel.model_dump()),
        environment=Environment(**payload.environment.model_dump()),
    )
    features = feature_row(plan)

    try:
        rate = bundle.predict([features])[0]
    except FeatureValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    voyage_fuel = None
    duration = None
    if payload.distance_nm:
        duration = payload.distance_nm / payload.sailing_speed
        voyage_fuel = round(rate * duration / 1000.0, 4)

    response = PredictionResponse(
        fuel_rate=round(rate, 4),
        fuel_rate_unit=settings.target_unit,
        unit_verified=settings.target_unit_verified,
        voyage_fuel_tonnes=voyage_fuel,
        duration_hours=round(duration, 3) if duration else None,
        model_metadata=bundle.metadata(),
        features_used=features,
        note=(
            f"Produced by {bundle.model_class} via the shipped preprocessor. "
            "The training artifacts do not record a confidence interval, so none is reported. "
            "The target unit is taken from configuration and is not verified by the artifacts."
        ),
    )
    run_id = RunRepository(db, "prediction").save(
        user_id=user["id"],
        request=payload.model_dump(),
        response=response.model_dump(),
    )
    response.prediction_id = run_id
    return response


@router.get("/history", summary="Recent predictions for the current user")
def history(user: CurrentUser, db: DbDep, limit: int = 20) -> dict:
    return {"items": RunRepository(db, "prediction").history(user["id"], limit=min(max(limit, 1), 100))}
