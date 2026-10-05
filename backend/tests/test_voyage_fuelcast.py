"""Voyage use of an explicit FuelCast snapshot and stored model provenance."""

import copy
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import deps
from app.api.routes import voyage as voyage_route
from app.core.config import get_settings
from app.services import voyage as voyage_service
from app.services.evaluator import CO2E_TONNES_PER_TONNE_FUEL
from app.services.model_registry import ModelUnavailableError


INPUTS = {
    "speed_over_ground": 8.0, "wind_speed": 5.5, "wind_direction": 120.0,
    "wave_height": 1.2, "wave_period": 7.0, "current_speed": 0.8,
}
BASE = {
    "vessel": "MV Test", "vessel_type": "Tanker Ship", "origin": "Mumbai",
    "destination": "Singapore", "distance_nm": 120.0, "speed_knots": 12.0,
    "fuel_loaded_tonnes": 300.0,
}
FUELCAST = {**BASE, "speed_knots": 15.55,
            "model_id": "fuelcast_xgboost", "fuelcast_inputs": INPUTS}


class MemoryVoyages:
    rows = {}

    def __init__(self, db):
        pass

    def create(self, row):
        self.rows[row["id"]] = copy.deepcopy(row)

    def get(self, voyage_id):
        return copy.deepcopy(self.rows.get(voyage_id))

    def list(self, user_id=None):
        return [copy.deepcopy(row) for row in self.rows.values()
                if user_id is None or row["user_id"] == user_id]


@pytest.fixture()
def voyage_client(monkeypatch):
    MemoryVoyages.rows = {}
    monkeypatch.setattr(voyage_service, "VoyageRepository", MemoryVoyages)
    monkeypatch.setattr(voyage_route, "AuditRepository", lambda db: type(
        "Audit", (), {"log": lambda self, **kwargs: None})())
    app = FastAPI()
    app.include_router(voyage_route.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db] = lambda: object()
    app.dependency_overrides[deps.get_current_user] = lambda: {
        "id": "test-user", "name": "Test User", "role": "operator",
    }
    return TestClient(app)


def _predictor(monkeypatch, *, rate=0.5, error=None):
    calls = []

    class Predictor:
        def predict(self, rows):
            calls.extend(rows)
            if error:
                raise error
            return {"fuel_rates": [rate], "fuel_rate_unit": "kg/s", "run_id": "test-run"}

        def metadata(self):
            return {"run_id": "test-run"}

    monkeypatch.setattr(voyage_service, "get_predictor", lambda model_id: Predictor())
    return calls


def test_legacy_default_and_explicit_legacy_match_existing_result(voyage_client):
    default = voyage_client.post("/api/v1/voyage", json=BASE)
    explicit = voyage_client.post("/api/v1/voyage", json={**BASE, "model_id": "legacy"})
    assert default.status_code == explicit.status_code == 201
    for field in ("modelled_fuel_rate", "fuel_rate_unit", "planned_fuel_tonnes",
                  "fuel_consumed_tonnes", "fuel_remaining_tonnes", "fuel_note"):
        assert default.json()[field] == explicit.json()[field]
    assert "model_id" not in default.json()
    assert "normalized_voyage_fuel_tonnes" not in default.json()


def test_fuelcast_conversion_cost_emissions_and_persistence(voyage_client, monkeypatch):
    calls = _predictor(monkeypatch)
    response = voyage_client.post("/api/v1/voyage", json=FUELCAST)
    assert response.status_code == 201, response.text
    body = response.json()
    assert calls == [INPUTS]
    expected_tonnes = 0.5 * (120.0 / 15.55) * 3.6
    assert body["planned_fuel_tonnes"] == pytest.approx(expected_tonnes)
    assert body["normalized_voyage_fuel_tonnes"] == pytest.approx(expected_tonnes)
    assert body["conversion_duration_hours"] == pytest.approx(120.0 / 15.55)
    assert body["model_id"] == "fuelcast_xgboost"
    assert body["model_run_id"] == "test-run"
    assert body["raw_prediction"] == 0.5
    assert body["raw_prediction_unit"] == body["fuel_rate_unit"] == "kg/s"
    assert body["normalized_voyage_fuel_unit"] == "tonnes"
    assert body["fuelcast_inputs"] == INPUTS
    settings = get_settings()
    assert body["cost_usd"] == round(expected_tonnes * settings.fuel_price_usd_per_tonne, 2)
    assert body["cost_inr"] == round(expected_tonnes * settings.fuel_price_usd_per_tonne * settings.usd_to_inr, 2)
    assert body["ghg_tonnes_co2e"] == round(expected_tonnes * CO2E_TONNES_PER_TONNE_FUEL["Marine Diesel"], 4)
    stored = next(iter(MemoryVoyages.rows.values()))
    assert stored["model_id"] == "fuelcast_xgboost"
    assert stored["normalized_voyage_fuel_tonnes"] == pytest.approx(expected_tonnes)
    assert stored["raw_prediction"] == 0.5
    fetched = voyage_client.get(f"/api/v1/voyage/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["planned_fuel_tonnes"] == pytest.approx(expected_tonnes)
    assert len(calls) == 1  # Reading a stored result does not run inference again.


def test_older_record_without_provenance_remains_readable(voyage_client):
    response = voyage_client.post("/api/v1/voyage", json=BASE)
    assert response.status_code == 201
    fetched = voyage_client.get(f"/api/v1/voyage/{response.json()['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["planned_fuel_tonnes"] == response.json()["planned_fuel_tonnes"]


@pytest.mark.parametrize("speed_knots", [8.0 / 0.514444, 15.55])
def test_matching_and_rounded_fuelcast_speeds_succeed(voyage_client, monkeypatch, speed_knots):
    calls = _predictor(monkeypatch)
    response = voyage_client.post("/api/v1/voyage", json={**FUELCAST, "speed_knots": speed_knots})
    assert response.status_code == 201, response.text
    assert calls == [INPUTS]


def test_mismatched_fuelcast_speeds_fail_before_model_or_persistence(voyage_client, monkeypatch):
    calls = _predictor(monkeypatch)
    response = voyage_client.post("/api/v1/voyage", json={**FUELCAST, "speed_knots": 12.0})
    assert response.status_code == 422
    assert "same voyage speed" in response.json()["detail"]
    assert calls == []
    assert MemoryVoyages.rows == {}


@pytest.mark.parametrize("departed_at", ["2026-10-01T10:20:30", "2026-10-01T10:20:30+05:30"])
def test_valid_departure_timestamp_is_persisted_and_readable(voyage_client, departed_at):
    response = voyage_client.post("/api/v1/voyage", json={**BASE, "departed_at": departed_at})
    assert response.status_code == 201
    stored = next(iter(MemoryVoyages.rows.values()))
    assert stored["departed_at"] == departed_at
    assert voyage_client.get(f"/api/v1/voyage/{response.json()['id']}").status_code == 200


def test_invalid_departure_timestamp_is_422_before_model_or_persistence(voyage_client, monkeypatch):
    calls = _predictor(monkeypatch)
    response = voyage_client.post("/api/v1/voyage", json={**FUELCAST, "departed_at": "yesterday"})
    assert response.status_code == 422
    assert calls == []
    assert MemoryVoyages.rows == {}


@pytest.mark.parametrize("body", [
    {**BASE, "model_id": "fuelcast_xgboost"},
    {**BASE, "model_id": "unknown"},
    {**BASE, "fuelcast_inputs": INPUTS},
    {**FUELCAST, "fuelcast_inputs": {**INPUTS, "wind_direction": 360.0}},
    {**FUELCAST, "fuelcast_inputs": {**INPUTS, "wave_period": 0}},
    {**FUELCAST, "fuelcast_inputs": {**INPUTS, "wind_speed": -1}},
    {**FUELCAST, "fuelcast_inputs": {key: value for key, value in INPUTS.items()
                                       if key != "current_speed"}},
])
def test_invalid_model_or_inputs_return_422(voyage_client, body):
    assert voyage_client.post("/api/v1/voyage", json=body).status_code == 422


@pytest.mark.parametrize("distance,speed", [(0, 12), (120, 0), (math.inf, 12), (120, math.nan)])
def test_invalid_duration_is_rejected_before_inference(monkeypatch, distance, speed):
    calls = _predictor(monkeypatch)
    payload = {**FUELCAST, "distance_nm": distance, "speed_knots": speed}
    with pytest.raises(voyage_service.VoyageCalculationError, match="duration"):
        voyage_service._fuelcast_result(payload)
    assert calls == []


@pytest.mark.parametrize("rate", [math.nan, math.inf, -1.0, 1e308])
def test_invalid_prediction_or_total_has_safe_http_error(voyage_client, monkeypatch, rate):
    _predictor(monkeypatch, rate=rate)
    response = voyage_client.post("/api/v1/voyage", json=FUELCAST)
    assert response.status_code in (422, 503)
    assert "Traceback" not in response.text
    assert MemoryVoyages.rows == {}


def test_artifact_failure_hides_internal_path(voyage_client, monkeypatch):
    _predictor(monkeypatch, error=ModelUnavailableError("/private/model.json missing"))
    response = voyage_client.post("/api/v1/voyage", json=FUELCAST)
    assert response.status_code == 503
    assert "/private/" not in response.text
    assert MemoryVoyages.rows == {}


def test_legacy_voyage_fuel_error_hides_internal_path(voyage_client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ModelUnavailableError("/private/legacy.pkl missing")

    monkeypatch.setattr(voyage_service, "evaluate", unavailable)
    response = voyage_client.post("/api/v1/voyage", json=BASE)
    assert response.status_code == 201
    assert response.json()["fuel_model_available"] is False
    assert response.json()["fuel_error"] == "Selected prediction model is unavailable."
    assert "/private/" not in response.text
