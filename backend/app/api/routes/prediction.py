"""Fuel prediction endpoints — backed by the real trained artifacts."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbDep, WriteUser
from app.core.config import get_settings
from app.db.database import RunRepository
from app.schemas.models import ModelInfoResponse, PredictionRequest, PredictionResponse
from app.services.evaluator import Environment, VesselState, VoyagePlan, feature_row
from app.services.model_registry import (
    MODEL_UNAVAILABLE_MESSAGE,
    PREDICTION_FAILED_MESSAGE,
    FeatureValidationError,
    ModelUnavailableError,
    UnknownModelError,
    evaluate_holdout,
    fuelcast_status,
    vqr_status,
    get_bundle,
    get_predictor,
    status as model_status,
)

router = APIRouter(prefix="/prediction", tags=["prediction"])
logger = logging.getLogger(__name__)


@router.get("/model", response_model=ModelInfoResponse, summary="Loaded model metadata and measured metrics")
def model_info(_: CurrentUser) -> ModelInfoResponse:
    info = model_status()
    models = {"legacy": info, "fuelcast_xgboost": fuelcast_status(), "fuelcast_vqr": vqr_status()}
    if not info["loaded"]:
        return ModelInfoResponse(loaded=False, error=info["error"], models=models)
    metadata = {k: v for k, v in info.items() if k not in {"loaded", "error"}}
    try:
        metrics = evaluate_holdout()
    except ModelUnavailableError:
        logger.exception("Model holdout evaluation failed")
        metrics = {"available": False, "reason": MODEL_UNAVAILABLE_MESSAGE}
    except Exception:
        logger.exception("Model holdout evaluation failed")
        metrics = {"available": False, "reason": "Model evaluation could not be completed."}
    return ModelInfoResponse(loaded=True, metadata=metadata, metrics=metrics, models=models)


@router.post("/fuel", response_model=PredictionResponse, summary="Predict fuel consumption rate")
def predict_fuel(payload: PredictionRequest, user: WriteUser, db: DbDep) -> PredictionResponse:
    settings = get_settings()

    if payload.model_id not in {"legacy", "fuelcast_xgboost", "fuelcast_vqr"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(UnknownModelError(
                                f"Unknown model ID {payload.model_id!r}. "
                                "Available IDs: legacy, fuelcast_xgboost, fuelcast_vqr."
                            )))
    if payload.model_id == "fuelcast_vqr":
        features = payload.fuelcast_inputs.model_dump()
        try:
            predictor = get_predictor("fuelcast_vqr")
        except ModelUnavailableError as exc:
            logger.exception("FuelCast VQR model unavailable")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=MODEL_UNAVAILABLE_MESSAGE) from exc
        try:
            prediction = predictor.predict([features])
        except FeatureValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            logger.exception("FuelCast VQR prediction failed")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=PREDICTION_FAILED_MESSAGE) from exc
        response = PredictionResponse(
            model_id=prediction["model_id"],
            model_run_id=prediction["run_id"],
            attempt_id=prediction["attempt_id"],
            execution_type="exact quantum simulator",
            fuel_rate=prediction["fuel_rates"][0],
            fuel_rate_unit=prediction["fuel_rate_unit"],
            unit_verified=prediction["unit_verified"],
            model_metadata=predictor.metadata(),
            features_used=features,
            note=("Exact simulator prediction from explicit FuelCast inputs; no quantum hardware was used. "
                  "Wind-direction convention remains unverified. No voyage total, cost, or emissions is calculated."),
        )
        response.prediction_id = RunRepository(db, "prediction").save(
            user_id=user["id"], request=payload.model_dump(), response=response.model_dump(),
        )
        return response
    if payload.model_id == "fuelcast_xgboost":
        features = payload.fuelcast_inputs.model_dump()
        try:
            predictor = get_predictor("fuelcast_xgboost")
            prediction = predictor.predict([features])
        except FeatureValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
        except ModelUnavailableError as exc:
            logger.exception("FuelCast prediction failed")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail=PREDICTION_FAILED_MESSAGE) from exc
        response = PredictionResponse(
            model_id=prediction["model_id"],
            fuel_rate=round(prediction["fuel_rates"][0], 4),
            fuel_rate_unit=prediction["fuel_rate_unit"],
            unit_verified=prediction["unit_verified"],
            model_metadata=predictor.metadata(),
            features_used=features,
            note=("Prediction uses only explicit FuelCast inputs. Wind-direction reference convention "
                  "still requires confirmation; no voyage fuel total or confidence interval is reported."),
        )
        response.prediction_id = RunRepository(db, "prediction").save(
            user_id=user["id"], request=payload.model_dump(), response=response.model_dump(),
        )
        return response

    try:
        bundle = get_bundle()
    except ModelUnavailableError as exc:
        logger.exception("Legacy prediction model unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_UNAVAILABLE_MESSAGE,
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
        logger.exception("Legacy prediction failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=PREDICTION_FAILED_MESSAGE) from exc

    voyage_fuel = None
    duration = None
    if payload.distance_nm:
        duration = payload.distance_nm / payload.sailing_speed
        voyage_fuel = round(rate * duration / 1000.0, 4)

    response = PredictionResponse(
        model_id="legacy",
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
