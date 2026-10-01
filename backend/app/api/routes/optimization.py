"""Fleet optimisation endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbDep, WriteUser
from app.db.database import RunRepository
from app.schemas.models import OptimizationRequest, OptimizationResponse
from app.services.evaluator import Environment, EvaluationError, VesselState, assumptions_block
from app.services.model_registry import MODEL_UNAVAILABLE_MESSAGE, ModelUnavailableError
from app.services.optimization import (
    ALGORITHM_LABELS,
    DEFAULT_SPEED_BOUNDS,
    OptimizationProblem,
    run_optimization,
)

router = APIRouter(prefix="/optimization", tags=["optimization"])
logger = logging.getLogger(__name__)


@router.get("/algorithms", summary="Available solvers and their status")
def algorithms(_: CurrentUser) -> dict:
    return {
        "algorithms": [
            {
                "key": "nsga2",
                "label": ALGORITHM_LABELS["nsga2"],
                "status": "implemented",
                "description": "Fast non-dominated sort, crowding distance, SBX crossover, polynomial mutation.",
            },
            {
                "key": "quantum",
                "label": ALGORITHM_LABELS["quantum"],
                "status": "implemented (classical simulation)",
                "description": (
                    "Quantum-inspired evolutionary algorithm: qubit probability registers collapsed "
                    "each generation and updated with a rotation gate. Runs on a CPU; no quantum "
                    "hardware is used and no quantum advantage is claimed."
                ),
            },
        ],
        "speed_bounds": list(DEFAULT_SPEED_BOUNDS),
        "speed_bounds_note": (
            "Legacy sailing-speed bounds in knots, clamped to its training data. "
            "FuelCast requires explicit speed-over-ground bounds in m/s."
        ),
        "assumptions": assumptions_block(),
    }


@router.post("/run", response_model=OptimizationResponse, summary="Run the optimiser(s)")
def run(payload: OptimizationRequest, user: WriteUser, db: DbDep) -> OptimizationResponse:
    selected = ("nsga2", "quantum") if payload.algorithm == "both" else (payload.algorithm,)
    problem = OptimizationProblem(
        distance_nm=payload.distance_nm,
        vessel=VesselState(**payload.vessel.model_dump()),
        environment=(Environment(**payload.environment.model_dump()) if payload.environment is not None else None),
        available_fuels=list(payload.available_fuels),
        model_id=payload.model_id,
        fuelcast_inputs=(payload.fuelcast_inputs.model_dump() if payload.fuelcast_inputs is not None else None),
        speed_bounds=(payload.fuelcast_speed_bounds_m_s if payload.model_id == "fuelcast_xgboost"
                      else DEFAULT_SPEED_BOUNDS),
        allow_shore_power=payload.allow_shore_power,
        max_eta_hours=payload.max_eta_hours,
        max_ghg_tonnes=payload.max_ghg_tonnes,
        cost_weight=payload.cost_weight,
        population_size=payload.population_size,
        generations=payload.generations,
        seed=payload.seed,
    )
    try:
        result = run_optimization(problem, algorithms=selected)
    except ModelUnavailableError as exc:
        logger.exception("Optimization prediction model unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_UNAVAILABLE_MESSAGE,
        ) from exc
    except EvaluationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    request_dump = payload.model_dump()
    request_dump["origin"] = payload.origin
    request_dump["destination"] = payload.destination
    run_id = RunRepository(db, "optimization").save(user_id=user["id"], request=request_dump, response=result)
    return OptimizationResponse(run_id=run_id, request=request_dump, result=result)


@router.get("/runs/latest", summary="Most recent optimisation run for the current user")
def latest(user: CurrentUser, db: DbDep) -> dict:
    record = RunRepository(db, "optimization").latest(user["id"])
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No optimisation has been run on this account yet.",
        )
    return record


@router.get("/runs/{run_id}", summary="Fetch one optimisation run")
def get_run(run_id: str, user: CurrentUser, db: DbDep) -> dict:
    record = RunRepository(db, "optimization").get(run_id, user["id"])
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Optimisation run not found.")
    return record


@router.get("/runs", summary="Optimisation history for the current user")
def history(user: CurrentUser, db: DbDep, limit: int = 20) -> dict:
    return {"items": RunRepository(db, "optimization").history(user["id"], limit=min(max(limit, 1), 100))}
