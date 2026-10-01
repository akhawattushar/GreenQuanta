"""HTTP contract for explicit FuelCast inputs and the unchanged legacy default."""

import copy
import json

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import deps
from app.api.routes import prediction as prediction_route
from app.api.routes import health as health_route
from app.api.routes import admin as admin_route
from app.core.config import Settings
from app.schemas.models import FuelCastInputs
from app.services import model_registry
from app.services.model_registry import FeatureValidationError, ModelUnavailableError

PATH = "/api/v1/prediction/fuel"
FUELCAST_INPUTS = {
    "speed_over_ground": 8.0,
    "wind_speed": 5.5,
    "wind_direction": 120.0,
    "wave_height": 1.2,
    "wave_period": 7.0,
    "current_speed": 0.8,
}
FUELCAST_BODY = {"model_id": "fuelcast_xgboost", "fuelcast_inputs": FUELCAST_INPUTS}
LEGACY_BODY = {
    "sailing_speed": 15.5,
    "vessel": {"vessel_type": "Tanker Ship", "displacement": 12.0, "trim": 0.0},
    "environment": {
        "wind_speed": 14.0, "wind_direction_relative": 90.0,
        "combined_wave_height": 3.0, "combined_wave_period": 5.5,
        "sea_current_speed": 0.5, "sea_current_direction_relative": 90.0,
        "sea_water_temperature": 17.0,
    },
}


@pytest.fixture()
def client(monkeypatch):
    app = FastAPI()
    app.include_router(prediction_route.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db] = lambda: object()
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": "test-user", "role": "operator"}

    class FakeRunRepository:
        def __init__(self, db, kind):
            assert kind == "prediction"

        def save(self, *, user_id, request, response):
            assert user_id == "test-user"
            return "prediction-test-id"

    monkeypatch.setattr(prediction_route, "RunRepository", FakeRunRepository)
    return TestClient(app)


def test_legacy_request_without_model_id_still_predicts(client):
    response = client.post(PATH, json=LEGACY_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "legacy"
    assert body["fuel_rate_unit"] != "kg/s"
    assert body["features_used"]["wind_direction_relative"] == 90.0
    assert body["prediction_id"] == "prediction-test-id"
    assert all("/" not in path for path in body["model_metadata"]["artifact_paths"].values())


def test_explicit_fuelcast_request_predicts_only_raw_inputs(client, monkeypatch):
    calls = []

    class Predictor:
        def predict(self, rows):
            calls.extend(rows)
            return {"model_id": "fuelcast_xgboost", "fuel_rates": [0.25],
                    "fuel_rate_unit": "kg/s", "unit_verified": True}

        def metadata(self):
            return {"model_id": "fuelcast_xgboost", "target_unit": "kg/s"}

    monkeypatch.setattr(prediction_route, "get_predictor", lambda model_id: Predictor())
    response = client.post(PATH, json=FUELCAST_BODY)
    assert response.status_code == 200
    body = response.json()
    assert calls == [FUELCAST_INPUTS]
    assert body["model_id"] == "fuelcast_xgboost"
    assert body["fuel_rate"] == 0.25
    assert body["fuel_rate_unit"] == "kg/s"
    assert body["unit_verified"] is True
    assert body["features_used"] == FUELCAST_INPUTS
    assert body["voyage_fuel_tonnes"] is None
    assert body["duration_hours"] is None


def test_fuelcast_inputs_are_required(client):
    response = client.post(PATH, json={"model_id": "fuelcast_xgboost"})
    assert response.status_code == 422
    assert "fuelcast_inputs" in str(response.json())


@pytest.mark.parametrize("missing", FUELCAST_INPUTS)
def test_each_fuelcast_field_is_required(client, missing):
    body = copy.deepcopy(FUELCAST_BODY)
    body["fuelcast_inputs"].pop(missing)
    response = client.post(PATH, json=body)
    assert response.status_code == 422
    assert missing in str(response.json())


@pytest.mark.parametrize("field,value", [
    ("speed_over_ground", -0.1), ("wind_speed", -0.1),
    ("wave_height", -0.1), ("current_speed", -0.1),
    ("wave_period", 0), ("wind_direction", -0.1),
    ("wind_direction", 360.0), ("wind_direction", 720.0),
])
def test_fuelcast_ranges_are_enforced(client, field, value):
    body = copy.deepcopy(FUELCAST_BODY)
    body["fuelcast_inputs"][field] = value
    response = client.post(PATH, json=body)
    assert response.status_code == 422
    assert field in str(response.json())


@pytest.mark.parametrize("field", FUELCAST_INPUTS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_fuelcast_values_are_rejected(client, field, value):
    body = copy.deepcopy(FUELCAST_BODY)
    body["fuelcast_inputs"][field] = value
    response = client.post(PATH, content=json.dumps(body), headers={"content-type": "application/json"})
    assert response.status_code == 422


def test_legacy_cannot_silently_use_fuelcast_inputs(client):
    assert client.post(PATH, json={**LEGACY_BODY, "fuelcast_inputs": FUELCAST_INPUTS}).status_code == 422


def test_fuelcast_cannot_silently_use_legacy_fields(client):
    assert client.post(PATH, json={**FUELCAST_BODY, "sailing_speed": 15.5}).status_code == 422


def test_unknown_model_id_remains_422(client):
    response = client.post(PATH, json={"model_id": "unknown"})
    assert response.status_code == 422
    assert "Unknown model ID" in response.json()["detail"]


@pytest.mark.parametrize("error,expected_status", [
    (ModelUnavailableError("runtime unavailable"), 503),
    (FeatureValidationError("bad raw row"), 422),
])
def test_predictor_errors_keep_safe_http_status(client, monkeypatch, error, expected_status):
    class Predictor:
        def predict(self, rows):
            raise error

    monkeypatch.setattr(prediction_route, "get_predictor", lambda model_id: Predictor())
    response = client.post(PATH, json=FUELCAST_BODY)
    assert response.status_code == expected_status
    if isinstance(error, ModelUnavailableError):
        assert response.json()["detail"] == "Prediction could not be completed."
    else:
        assert str(error) in response.json()["detail"]


def test_unavailable_model_load_is_503(client, monkeypatch):
    def unavailable(model_id):
        raise ModelUnavailableError("/private/artifacts/model.json unavailable")

    monkeypatch.setattr(prediction_route, "get_predictor", unavailable)
    response = client.post(PATH, json=FUELCAST_BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction could not be completed."
    assert "/private/" not in response.text


def test_legacy_model_error_and_public_health_hide_paths(client, monkeypatch):
    def unavailable():
        raise ModelUnavailableError("/private/artifacts/best_model.pkl missing")

    monkeypatch.setattr(prediction_route, "get_bundle", unavailable)
    response = client.post(PATH, json=LEGACY_BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == "Selected prediction model is unavailable."
    assert "/private/" not in response.text

    monkeypatch.setattr(health_route.model_registry, "status", lambda: {
        "loaded": False, "error": "/private/artifacts/best_model.pkl missing"})
    monkeypatch.setattr(health_route, "get_db", lambda: type("DB", (), {"healthy": lambda self: True})())
    health = health_route.health()
    assert health.model_error == "Model health check failed."


def test_model_status_and_missing_evaluation_data_hide_paths(monkeypatch, tmp_path):
    def unavailable():
        raise ModelUnavailableError("/private/artifacts/model.pkl missing")

    monkeypatch.setattr(model_registry, "get_bundle", unavailable)
    assert model_registry.status() == {"loaded": False, "error": "Model health check failed."}
    monkeypatch.setattr(model_registry, "get_predictor", lambda model_id: unavailable())
    assert model_registry.fuelcast_status()["error"] == "Model health check failed."
    monkeypatch.setattr(model_registry, "get_settings", lambda: Settings(artifacts_dir=tmp_path))
    metrics = model_registry.evaluate_holdout()
    assert metrics["available"] is False
    assert str(tmp_path) not in metrics["reason"]


def test_admin_model_reload_error_hides_artifact_path(monkeypatch):
    def unavailable(*, force_reload):
        raise ModelUnavailableError("/private/artifacts/model.pkl missing")

    monkeypatch.setattr(admin_route.model_registry, "get_bundle", unavailable)
    with pytest.raises(HTTPException) as exc:
        admin_route.reload_model({"id": "admin"})
    assert exc.value.status_code == 503
    assert exc.value.detail == "Selected prediction model is unavailable."


def test_openapi_describes_all_fuelcast_units_and_convention():
    properties = FuelCastInputs.model_json_schema()["properties"]
    for field, unit in (
        ("speed_over_ground", "m/s"), ("wind_speed", "m/s"),
        ("wind_direction", "degrees"), ("wave_height", "metres"),
        ("wave_period", "seconds"), ("current_speed", "m/s"),
    ):
        assert unit in properties[field]["description"]
    assert "not yet verified" in properties["wind_direction"]["description"]
