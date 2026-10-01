"""Scenario analysis endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbDep, WriteUser
from app.db.database import RunRepository
from app.schemas.models import ScenarioRequest, ScenarioResponse
from app.services.evaluator import Environment, EvaluationError, VesselState
from app.services.model_registry import ModelUnavailableError
from app.services.scenario import list_scenarios, run_scenarios

router = APIRouter(prefix="/scenario", tags=["scenario"])


@router.get("/catalog", summary="Scenarios this service can evaluate")
def catalog(_: CurrentUser) -> dict:
    return {"scenarios": list_scenarios()}


@router.post("/run", response_model=ScenarioResponse, summary="Evaluate the voyage under each scenario")
def run(payload: ScenarioRequest, user: WriteUser, db: DbDep) -> ScenarioResponse:
    try:
        result = run_scenarios(
            scenario_keys=list(payload.scenarios),
            distance_nm=payload.distance_nm,
            vessel=VesselState(**payload.vessel.model_dump()),
            environment=(Environment(**payload.environment.model_dump()) if payload.environment is not None else None),
            available_fuels=list(payload.available_fuels),
            model_id=payload.model_id,
            fuelcast_inputs=(payload.fuelcast_inputs.model_dump() if payload.fuelcast_inputs is not None else None),
            fuelcast_scenario_inputs={key: value.model_dump() for key, value in
                                      (payload.fuelcast_scenario_inputs or {}).items()},
            fuelcast_speed_bounds_m_s=payload.fuelcast_speed_bounds_m_s,
            speed_knots=payload.speed_knots,
            max_eta_hours=payload.max_eta_hours,
            optimize=payload.optimize,
            population_size=payload.population_size,
            generations=payload.generations,
            seed=payload.seed,
        )
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The selected prediction model is unavailable or failed during scenario analysis.",
        ) from exc
    except EvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    run_id = RunRepository(db, "scenario").save(user_id=user["id"], request=payload.model_dump(), response=result)
    return ScenarioResponse(run_id=run_id, result=result)


@router.get("/runs/latest", summary="Most recent scenario run")
def latest(user: CurrentUser, db: DbDep) -> dict:
    record = RunRepository(db, "scenario").latest(user["id"])
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No scenario run yet.")
    return record


@router.get("/runs/{run_id}", summary="Fetch one scenario run")
def get_run(run_id: str, user: CurrentUser, db: DbDep) -> dict:
    record = RunRepository(db, "scenario").get(run_id, user["id"])
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scenario run not found.")
    return record


@router.get("/runs", summary="Scenario run history for the current user")
def history(user: CurrentUser, db: DbDep, limit: int = 20) -> dict:
    return {"items": RunRepository(db, "scenario").history(user["id"], limit=min(max(limit, 1), 100))}
