"""Explicit FuelCast model selection in the existing optimization solvers."""

import math
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import deps
from app.api.routes import optimization as optimization_route
from app.core.config import get_settings
from app.services import optimization as optimization_service
from app.services.evaluator import CO2E_TONNES_PER_TONNE_FUEL, EvaluationError, VesselState
from app.services.model_registry import ModelUnavailableError


INPUTS = {
    "speed_over_ground": 8.0, "wind_speed": 5.5, "wind_direction": 120.0,
    "wave_height": 1.2, "wave_period": 7.0, "current_speed": 0.8,
}
BODY = {
    "origin": "Mumbai", "destination": "Singapore", "distance_nm": 120.0,
    "vessel": {"vessel_type": "Tanker Ship", "displacement": 12.0, "trim": 0.0},
    "available_fuels": ["Marine Diesel"], "allow_shore_power": False,
    "model_id": "fuelcast_xgboost", "fuelcast_inputs": INPUTS,
    "fuelcast_speed_bounds_m_s": [5.0, 10.0], "algorithm": "nsga2",
    "population_size": 8, "generations": 5, "seed": 7,
}


def _problem():
    return optimization_service.OptimizationProblem(
        distance_nm=120.0, vessel=VesselState("Tanker Ship", 12.0, 0.0),
        environment=None, available_fuels=["Marine Diesel"], allow_shore_power=False,
        model_id="fuelcast_xgboost", fuelcast_inputs=dict(INPUTS), speed_bounds=(5.0, 10.0),
        population_size=8, generations=5, seed=7,
    )


class Predictor:
    def __init__(self, *, rate=0.5, bad_speeds=(), error=None):
        self.rows = []
        self.rate = rate
        self.bad_speeds = bad_speeds
        self.error = error

    def predict(self, rows):
        self.rows.extend(rows)
        if self.error:
            raise self.error
        speed = rows[0]["speed_over_ground"]
        rate = math.nan if speed in self.bad_speeds else self.rate
        return {"fuel_rates": [rate], "fuel_rate_unit": "kg/s", "run_id": "test-run"}

    def metadata(self):
        return {"run_id": "test-run"}


@pytest.fixture()
def client(monkeypatch):
    class Runs:
        saved = None

        def __init__(self, db, kind):
            assert kind == "optimization"

        def save(self, *, user_id, request, response):
            Runs.saved = {"id": "opt-test", "request": request, "response": response}
            return "opt-test"

        def latest(self, user_id):
            return self.saved

        def get(self, run_id, user_id):
            return self.saved

    Runs.saved = None
    monkeypatch.setattr(optimization_route, "RunRepository", Runs)
    app = FastAPI()
    app.include_router(optimization_route.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db] = lambda: object()
    app.dependency_overrides[deps.get_current_user] = lambda: {
        "id": "test-user", "name": "Test User", "role": "operator",
    }
    return TestClient(app), Runs


def test_candidate_speed_duration_conversion_and_objectives():
    problem = _problem()
    predictor = Predictor()
    problem.predictor = predictor
    problem.model_run_id = "test-run"
    candidates = [optimization_service.Candidate(speed=s, fuel_index=0, shore_power=False)
                  for s in (5.0, 10.0)]
    optimization_service._score(problem, candidates)
    assert [row["speed_over_ground"] for row in predictor.rows] == [5.0, 10.0]
    assert all({key: value for key, value in row.items() if key != "speed_over_ground"}
               == {key: value for key, value in INPUTS.items() if key != "speed_over_ground"}
               for row in predictor.rows)
    settings = get_settings()
    for candidate in candidates:
        ev = candidate.evaluation
        duration = 120.0 / (candidate.speed / optimization_service.KNOT_IN_M_S)
        tonnes = 0.5 * duration * 3.6
        assert ev.duration_hours == pytest.approx(duration)
        assert ev.fuel_tonnes == pytest.approx(tonnes)
        assert ev.main_engine_fuel_tonnes == pytest.approx(tonnes)
        assert ev.auxiliary_fuel_tonnes == 0
        assert ev.cost_usd == pytest.approx(tonnes * settings.fuel_price_usd_per_tonne)
        assert ev.ghg_tonnes_co2e == pytest.approx(
            tonnes * CO2E_TONNES_PER_TONNE_FUEL["Marine Diesel"])
        assert candidate.provenance["speed_over_ground_m_s"] == candidate.speed
    assert candidates[0].evaluation.duration_hours > candidates[1].evaluation.duration_hours
    assert candidates[0].evaluation.fuel_tonnes > candidates[1].evaluation.fuel_tonnes
    assert optimization_service.KNOT_IN_M_S == 0.514444


def test_run_reuses_predictor_and_separates_algorithm_from_model(monkeypatch):
    predictor = Predictor()
    loads = []

    def load(model_id):
        loads.append(model_id)
        return predictor

    monkeypatch.setattr(optimization_service, "get_predictor", load)
    result = optimization_service.run_optimization(_problem(), algorithms=("nsga2", "quantum"))
    assert loads == ["fuelcast_xgboost"]
    assert len(predictor.rows) > 8
    assert len({row["speed_over_ground"] for row in predictor.rows}) > 1
    assert result["model_id"] == "fuelcast_xgboost"
    assert result["model_run_id"] == "test-run"
    assert result["algorithms_run"] == ["nsga2", "quantum"]
    assert result["best_plan"]["algorithm_id"] in {"nsga2", "quantum", "baseline"}
    assert result["best_plan"]["model_id"] == "fuelcast_xgboost"
    assert result["best_plan"]["raw_prediction_unit"] == "kg/s"
    assert result["best_plan"]["normalized_voyage_fuel_unit"] == "tonnes"
    assert result["fixed_environment_snapshot"] == {
        key: value for key, value in INPUTS.items() if key != "speed_over_ground"}


def test_invalid_candidate_does_not_poison_valid_candidate():
    problem = _problem()
    problem.predictor = Predictor(bad_speeds=(5.0,))
    candidates = [optimization_service.Candidate(speed=s, fuel_index=0, shore_power=False)
                  for s in (5.0, 10.0)]
    optimization_service._score(problem, candidates)
    assert candidates[0].evaluation is None
    assert candidates[1].evaluation is not None


def test_no_valid_candidates_fails_deterministically():
    problem = _problem()
    problem.predictor = Predictor(rate=math.nan)
    candidates = [optimization_service.Candidate(speed=5.0, fuel_index=0, shore_power=False)]
    with pytest.raises(EvaluationError, match="No valid optimization candidates"):
        optimization_service._score(problem, candidates)


def test_api_persists_provenance_and_old_records_still_read(client, monkeypatch):
    test_client, runs = client
    predictor = Predictor()
    monkeypatch.setattr(optimization_service, "get_predictor", lambda model_id: predictor)
    response = test_client.post("/api/v1/optimization/run", json=BODY)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert runs.saved["request"]["model_id"] == "fuelcast_xgboost"
    assert runs.saved["response"]["model_id"] == "fuelcast_xgboost"
    assert runs.saved["response"]["best_plan"]["algorithm_id"] == "nsga2"
    assert test_client.get("/api/v1/optimization/runs/latest").json()["response"]["model_run_id"] == "test-run"
    runs.saved = {"id": "old", "request": {"origin": "Mumbai"}, "response": {"plans": []}}
    assert test_client.get("/api/v1/optimization/runs/old").json()["id"] == "old"
    assert result["best_plan"]["speed_over_ground_m_s"] > 0


def test_legacy_api_default_matches_explicit_legacy(client):
    test_client, _ = client
    legacy = {
        "origin": "Mumbai", "destination": "Singapore", "distance_nm": 120,
        "vessel": BODY["vessel"], "available_fuels": ["Marine Diesel"],
        "environment": {
            "wind_speed": 14, "wind_direction_relative": 90,
            "combined_wave_height": 3, "combined_wave_period": 5.5,
            "sea_current_speed": 0.5, "sea_current_direction_relative": 90,
            "sea_water_temperature": 17,
        },
        "algorithm": "nsga2", "population_size": 8, "generations": 5, "seed": 7,
    }
    default = test_client.post("/api/v1/optimization/run", json=legacy)
    explicit = test_client.post("/api/v1/optimization/run", json={**legacy, "model_id": "legacy"})
    assert default.status_code == explicit.status_code == 200
    for field in ("speed_knots", "fuel_rate", "fuel_tonnes", "cost_usd", "ghg_tonnes_co2e", "objective"):
        assert default.json()["result"]["best_plan"][field] == explicit.json()["result"]["best_plan"][field]
    assert "model_id" not in default.json()["result"]


@pytest.mark.parametrize("patch", [
    {"model_id": "unknown"}, {"fuelcast_inputs": None},
    {"fuelcast_speed_bounds_m_s": None}, {"fuelcast_speed_bounds_m_s": [0, 10]},
    {"fuelcast_speed_bounds_m_s": [10, 5]},
    {"fuelcast_inputs": {**INPUTS, "wave_period": 0}},
    {"fuelcast_inputs": {key: value for key, value in INPUTS.items() if key != "current_speed"}},
    {"environment": {"wind_speed": 5}}, {"available_fuels": ["LNG"]},
    {"allow_shore_power": True},
    {"fuelcast_inputs": {**INPUTS, "speed_over_ground": 12}},
])
def test_invalid_xgboost_contract_returns_422(client, patch):
    test_client, _ = client
    assert test_client.post("/api/v1/optimization/run", json={**BODY, **patch}).status_code == 422


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_fuelcast_input_is_422(client, value):
    test_client, _ = client
    body = {**BODY, "fuelcast_inputs": {**INPUTS, "wind_speed": value}}
    response = test_client.post(
        "/api/v1/optimization/run", content=json.dumps(body), headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_artifact_error_is_safe(client, monkeypatch):
    test_client, _ = client

    def unavailable(model_id):
        raise ModelUnavailableError("/private/artifacts/model.json missing")

    monkeypatch.setattr(optimization_service, "get_predictor", unavailable)
    response = test_client.post("/api/v1/optimization/run", json=BODY)
    assert response.status_code == 503
    assert "/private/" not in response.text


def test_no_valid_candidates_is_422(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(optimization_service, "get_predictor", lambda model_id: Predictor(rate=math.nan))
    response = test_client.post("/api/v1/optimization/run", json=BODY)
    assert response.status_code == 422
    assert "No valid optimization candidates" in response.json()["detail"]
