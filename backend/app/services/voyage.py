"""Voyage monitoring.

No AIS or telemetry feed is connected to this project. Voyages are created by
users and stored in MongoDB; progress is advanced deterministically from the
departure timestamp, so every payload carries ``data_source: "user_entered"``
and ``live_telemetry: false``. Swap `describe` for a real feed and the API
contract does not change.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.db.database import Database, VoyageRepository, new_id
from app.services.evaluator import Environment, VesselState, VoyagePlan, evaluate
from app.services.model_registry import ModelUnavailableError

DATA_SOURCE = "user_entered"

#: Nominal sea state used to score voyages that have no observation attached.
NOMINAL_ENVIRONMENT = Environment(
    wind_speed=14.0,
    wind_direction_relative=97.5,
    combined_wave_height=3.25,
    combined_wave_period=5.75,
    sea_current_speed=0.55,
    sea_current_direction_relative=90.1,
    sea_water_temperature=17.0,
)


def create_voyage(db: Database, *, user_id: str, user_name: str, payload: dict) -> dict:
    """Persist a real, user-entered voyage record."""
    now = datetime.now(timezone.utc)
    record = {
        "id": new_id("VY"),
        "user_id": user_id,
        "created_by_name": user_name,
        "vessel": payload["vessel"],
        "vessel_type": payload["vessel_type"],
        "origin": payload["origin"],
        "destination": payload["destination"],
        "distance_nm": payload["distance_nm"],
        "speed_knots": payload["speed_knots"],
        "fuel_loaded_tonnes": payload["fuel_loaded_tonnes"],
        "departed_at": payload.get("departed_at") or now.isoformat(timespec="seconds"),
        "data_source": DATA_SOURCE,
    }
    VoyageRepository(db).create(record)
    return describe(record)


def delete_voyage(db: Database, voyage_id: str, *, user_id: str, is_admin: bool) -> bool:
    repo = VoyageRepository(db)
    voyage = repo.get(voyage_id)
    if voyage is None:
        return False
    if not is_admin and voyage.get("user_id") != user_id:
        raise PermissionError("You can only delete voyages you created.")
    return repo.delete(voyage_id)


def _status(progress_pct: float, elapsed: float, planned: float) -> str:
    if progress_pct >= 100.0:
        return "Completed"
    if elapsed > planned * 1.05:
        return "Delayed"
    if elapsed > planned * 0.95:
        return "At Risk"
    return "On Track"


def describe(voyage: dict, *, now: datetime | None = None) -> dict:
    """Turn a stored voyage row into a monitoring payload."""
    now = now or datetime.now(timezone.utc)
    departed = datetime.fromisoformat(voyage["departed_at"])
    if departed.tzinfo is None:
        departed = departed.replace(tzinfo=timezone.utc)

    planned_hours = voyage["distance_nm"] / max(voyage["speed_knots"], 0.1)
    elapsed_hours = max((now - departed).total_seconds() / 3600.0, 0.0)
    progress = min(elapsed_hours / planned_hours, 1.0) if planned_hours > 0 else 0.0

    payload = {
        "id": voyage["id"],
        "user_id": voyage.get("user_id"),
        "created_by_name": voyage.get("created_by_name"),
        "vessel": voyage["vessel"],
        "vessel_type": voyage["vessel_type"],
        "origin": voyage["origin"],
        "destination": voyage["destination"],
        "distance_nm": voyage["distance_nm"],
        "speed_knots": voyage["speed_knots"],
        "departed_at": departed.isoformat(timespec="seconds"),
        "planned_duration_hours": round(planned_hours, 2),
        "elapsed_hours": round(elapsed_hours, 2),
        "progress_pct": round(progress * 100, 1),
        "distance_covered_nm": round(voyage["distance_nm"] * progress, 1),
        "distance_remaining_nm": round(voyage["distance_nm"] * (1 - progress), 1),
        "eta": (departed + timedelta(hours=planned_hours)).isoformat(timespec="seconds"),
        "fuel_loaded_tonnes": voyage["fuel_loaded_tonnes"],
        "status": _status(progress * 100, elapsed_hours, planned_hours),
        "data_source": voyage.get("data_source", DATA_SOURCE),
        "live_telemetry": False,
    }

    # Fuel burn is computed by the trained model for this vessel and speed,
    # not stored as a fixed number.
    try:
        plan = VoyagePlan(
            distance_nm=voyage["distance_nm"],
            speed_knots=voyage["speed_knots"],
            vessel=VesselState(
                vessel_type=voyage["vessel_type"],
                displacement=12.0,
                trim=0.0,
            ),
            environment=NOMINAL_ENVIRONMENT,
        )
        evaluation = evaluate(plan)
        consumed = evaluation.main_engine_fuel_tonnes * progress
        payload.update(
            {
                "fuel_model_available": True,
                "modelled_fuel_rate": evaluation.fuel_rate,
                "fuel_rate_unit": evaluation.fuel_rate_unit,
                "planned_fuel_tonnes": evaluation.fuel_tonnes,
                "fuel_consumed_tonnes": round(consumed, 2),
                "fuel_remaining_tonnes": round(voyage["fuel_loaded_tonnes"] - consumed, 2),
                "fuel_note": (
                    "Consumption is the model's rate for this vessel/speed under a nominal sea "
                    "state, scaled by elapsed progress. It is not a metered reading."
                ),
            }
        )
    except ModelUnavailableError as exc:
        payload.update(
            {
                "fuel_model_available": False,
                "fuel_error": str(exc),
                "fuel_consumed_tonnes": None,
                "fuel_remaining_tonnes": None,
            }
        )
    return payload


def list_voyages(db: Database, user_id: str | None = None) -> list:
    return [describe(v) for v in VoyageRepository(db).list(user_id=user_id)]


def get_voyage(db: Database, voyage_id: str) -> dict | None:
    row = VoyageRepository(db).get(voyage_id)
    return describe(row) if row else None


def fleet_summary(db: Database, user_id: str | None = None) -> dict:
    """Aggregate numbers for the dashboard, derived from stored voyages."""
    voyages = list_voyages(db, user_id=user_id)
    active = [v for v in voyages if v["status"] != "Completed"]
    modelled = [v for v in voyages if v.get("fuel_model_available")]
    planned_fuel = sum(v["planned_fuel_tonnes"] for v in modelled)
    consumed = sum(v["fuel_consumed_tonnes"] or 0.0 for v in modelled)
    return {
        "vessels": len({v["vessel"] for v in voyages}),
        "active_voyages": len(active),
        "total_voyages": len(voyages),
        "planned_fuel_tonnes": round(planned_fuel, 2),
        "fuel_consumed_tonnes": round(consumed, 2),
        "fuel_model_available": len(modelled) == len(voyages) and bool(voyages),
        "data_source": DATA_SOURCE,
        "live_telemetry": False,
        "note": (
            "Aggregated from stored voyage records with model-derived fuel figures. "
            "Cumulative 'fuel saved' and 'GHG avoided' are not reported because this project "
            "has no measured historical baseline to compare against."
        ),
    }
