"""FuelCast scenario snapshots and run-local assumptions."""

import copy
import json
import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import deps
from app.api.routes import scenario as scenario_route
from app.core.config import get_settings
from app.services import scenario as scenario_service
from app.services.evaluator import Environment, VesselState
from app.services.model_registry import ModelUnavailableError


INPUTS = {
    "speed_over_ground": 8.0, "wind_speed": 5.5, "wind_direction": 120.0,
    "wave_height": 1.2, "wave_period": 7.0, "current_speed": 0.8,
}
WEATHER = {**INPUTS, "wind_speed": 9.0, "wave_height": 2.0, "wave_period": 8.0}
LEGACY_ENV = {
    "wind_speed": 14, "wind_direction_relative": 90,
    "combined_wave_height": 3, "combined_wave_period": 5.5,
    "sea_current_speed": 0.5, "sea_current_direction_relative": 90,
    "sea_water_temperature": 17,
}
BASE = {
    "scenarios": ["base", "high_fuel_price"], "distance_nm": 120.0,
    "vessel": {"vessel_type": "Tanker Ship", "displacement": 12.0, "trim": 0.0},
    "available_fuels": ["Marine Diesel"], "optimize": False,
}
FUELCAST = {**BASE, "model_id": "fuelcast_xgboost", "fuelcast_inputs": INPUTS}


class Predictor:
    def __init__(self, rate=0.5, error=None):
        self.rate = rate
        self.error = error
        self.rows = []

    def predict(self, rows):
        self.rows.extend(copy.deepcopy(rows))
        if self.error is not None:
            raise self.error
        return {"fuel_rates": [self.rate], "fuel_rate_unit": "kg/s", "run_id": "test-run"}

    def metadata(self):
        return {"run_id": "test-run"}


@pytest.fixture()
def client(monkeypatch):
    class Runs:
        saved = None

        def __init__(self, db, kind):
            assert kind == "scenario"

        def save(self, *, user_id, request, response):
            Runs.saved = {"id": "scn-test", "request": request, "response": response}
            return "scn-test"

        def latest(self, user_id):
            return self.saved

        def get(self, run_id, user_id):
            return self.saved

    Runs.saved = None
    monkeypatch.setattr(scenario_route, "RunRepository", Runs)
    app = FastAPI()
    app.include_router(scenario_route.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db] = lambda: object()
    app.dependency_overrides[deps.get_current_user] = lambda: {
        "id": "test-user", "name": "Test User", "role": "operator",
    }
    return TestClient(app), Runs


def _rows(result):
    return {row["key"]: row for row in result["rows"]}


def test_fixed_fuelcast_price_changes_cost_without_changing_prediction(client, monkeypatch):
    test_client, runs = client
    predictor = Predictor()
    loads = []

    def load(model_id):
        loads.append(model_id)
        return predictor

    monkeypatch.setattr(scenario_service, "get_predictor", load)
    response = test_client.post("/api/v1/scenario/run", json=FUELCAST)
    assert response.status_code == 200, response.text
    rows = _rows(response.json()["result"])
    base, expensive = rows["base"], rows["high_fuel_price"]
    duration = 120 / (8 / 0.514444)
    tonnes = 0.5 * duration * 3.6
    assert loads == ["fuelcast_xgboost"]
    assert predictor.rows == [INPUTS, INPUTS]
    assert base["raw_prediction"] == expensive["raw_prediction"] == 0.5
    assert base["raw_prediction_unit"] == "kg/s"
    assert base["normalized_fuel_tonnes"] == pytest.approx(tonnes)
    assert expensive["normalized_fuel_tonnes"] == pytest.approx(tonnes)
    assert base["duration_hours"] == pytest.approx(duration)
    assert expensive["cost_usd"] > base["cost_usd"]
    assert expensive["cost_usd"] == pytest.approx(base["cost_usd"] * 1.4, abs=0.02)
    assert expensive["ghg_tonnes_co2e"] == base["ghg_tonnes_co2e"]
    assert expensive["applied_overrides"]["fuel_price_multiplier"] == 1.4
    assert base["model_id"] == "fuelcast_xgboost"
    assert base["model_run_id"] == "test-run"
    assert base["normalized_fuel_unit"] == "tonnes"
    assert base["fuelcast_input_snapshot"] == INPUTS
    assert runs.saved["response"]["rows"][0]["model_id"] == "fuelcast_xgboost"
    assert runs.saved["request"]["fuelcast_inputs"] == INPUTS
    assert test_client.get("/api/v1/scenario/runs/latest").json()["response"]["model_run_id"] == "test-run"


def test_weather_and_speed_scenarios_use_only_explicit_snapshots(client, monkeypatch):
    test_client, _ = client
    predictor = Predictor()
    monkeypatch.setattr(scenario_service, "get_predictor", lambda model_id: predictor)
    changed_speed = {**INPUTS, "speed_over_ground": 9.0}
    body = {
        **FUELCAST,
        "scenarios": ["base", "severe_weather", "fuelcast_speed_change"],
        "fuelcast_scenario_inputs": {
            "severe_weather": WEATHER,
            "fuelcast_speed_change": changed_speed,
        },
    }
    response = test_client.post("/api/v1/scenario/run", json=body)
    assert response.status_code == 200, response.text
    rows = _rows(response.json()["result"])
    assert predictor.rows == [INPUTS, WEATHER, changed_speed]
    assert rows["severe_weather"]["applied_overrides"]["fuelcast_inputs"] == {
        "wind_speed": 9.0, "wave_height": 2.0, "wave_period": 8.0,
    }
    assert rows["fuelcast_speed_change"]["speed_over_ground_m_s"] == 9.0
    assert rows["fuelcast_speed_change"]["duration_hours"] < rows["base"]["duration_hours"]


def test_optimized_scenarios_reuse_predictor_and_local_price(client, monkeypatch):
    test_client, _ = client
    predictor = Predictor()
    loads = []

    def load(model_id):
        loads.append(model_id)
        return predictor

    monkeypatch.setattr(scenario_service, "get_predictor", load)
    body = {**FUELCAST, "optimize": True, "fuelcast_speed_bounds_m_s": [5, 10],
            "population_size": 8, "generations": 5}
    response = test_client.post("/api/v1/scenario/run", json=body)
    assert response.status_code == 200, response.text
    rows = _rows(response.json()["result"])
    assert loads == ["fuelcast_xgboost"]
    assert len(predictor.rows) > 8
    assert len({row["speed_over_ground"] for row in predictor.rows}) > 1
    assert rows["base"]["optimization_algorithm_id"] in {"nsga2", "baseline"}
    assert rows["base"]["model_id"] == "fuelcast_xgboost"
    assert rows["base"]["raw_prediction_unit"] == "kg/s"
    assert rows["high_fuel_price"]["post_prediction_assumptions"]["fuel_price_usd_per_tonne"] > (
        rows["base"]["post_prediction_assumptions"]["fuel_price_usd_per_tonne"])


def test_legacy_default_and_explicit_legacy_match(client):
    test_client, _ = client
    body = {**BASE, "environment": LEGACY_ENV, "speed_knots": 15.5}
    default = test_client.post("/api/v1/scenario/run", json=body)
    explicit = test_client.post("/api/v1/scenario/run", json={**body, "model_id": "legacy"})
    assert default.status_code == explicit.status_code == 200
    for key in ("base", "high_fuel_price"):
        a, b = _rows(default.json()["result"])[key], _rows(explicit.json()["result"])[key]
        for field in ("fuel_rate", "fuel_tonnes", "cost_usd", "ghg_tonnes_co2e"):
            assert a[field] == b[field]
    assert "model_id" not in default.json()["result"]


def test_scenario_price_never_mutates_shared_settings(monkeypatch):
    settings = get_settings()
    original_price = settings.fuel_price_usd_per_tonne
    original_evaluate = scenario_service.evaluate
    seen = []

    def observe(plan, **kwargs):
        seen.append((get_settings().fuel_price_usd_per_tonne, kwargs["fuel_price_usd_per_tonne"]))
        return original_evaluate(plan, **kwargs)

    monkeypatch.setattr(scenario_service, "evaluate", observe)
    scenario_service.run_scenarios(
        scenario_keys=["base", "high_fuel_price"], distance_nm=120,
        vessel=VesselState("Tanker Ship", 12, 0), environment=Environment(**LEGACY_ENV),
        available_fuels=["Marine Diesel"], speed_knots=15.5, optimize=False,
    )
    assert seen == [(original_price, original_price), (original_price, original_price * 1.4)]
    assert get_settings().fuel_price_usd_per_tonne == original_price


@pytest.mark.parametrize("patch", [
    {"model_id": "unknown"}, {"fuelcast_inputs": None},
    {"fuelcast_inputs": {**INPUTS, "wave_period": 0}},
    {"fuelcast_inputs": {**INPUTS, "wind_direction": 360}},
    {"fuelcast_inputs": {key: value for key, value in INPUTS.items() if key != "wind_speed"}},
    {"environment": LEGACY_ENV}, {"speed_knots": 15},
    {"available_fuels": ["LNG"]}, {"scenarios": ["high_cargo_demand"]},
    {"shore_power": True},
    {"scenarios": ["severe_weather"]},
    {"scenarios": ["fuelcast_speed_change"]},
    {"optimize": True},
])
def test_invalid_fuelcast_scenario_contract_is_422(client, patch):
    test_client, _ = client
    response = test_client.post("/api/v1/scenario/run", json={**FUELCAST, **patch})
    assert response.status_code == 422


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_fuelcast_input_is_422(client, value):
    test_client, _ = client
    body = {**FUELCAST, "fuelcast_inputs": {**INPUTS, "wind_speed": value}}
    response = test_client.post(
        "/api/v1/scenario/run", content=json.dumps(body), headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_unavailable_and_invalid_predictor_are_safe(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(scenario_service, "get_predictor", lambda model_id: Predictor(
        error=ModelUnavailableError("/private/artifacts/model.json missing")))
    unavailable = test_client.post("/api/v1/scenario/run", json=FUELCAST)
    assert unavailable.status_code == 503
    assert "/private/" not in unavailable.text
    monkeypatch.setattr(scenario_service, "get_predictor", lambda model_id: Predictor(rate=math.nan))
    invalid = test_client.post("/api/v1/scenario/run", json=FUELCAST)
    assert invalid.status_code == 503
    assert "Traceback" not in invalid.text


def test_older_stored_scenario_document_remains_readable(client):
    test_client, runs = client
    runs.saved = {"id": "old-run", "request": {"scenarios": ["base"]},
                  "response": {"rows": [{"key": "base"}]}}
    response = test_client.get("/api/v1/scenario/runs/old-run")
    assert response.status_code == 200
    assert response.json()["response"]["rows"] == [{"key": "base"}]
