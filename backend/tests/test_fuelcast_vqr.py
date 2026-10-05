"""VQR adapter, registry, and direct HTTP contract without training the model."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import deps
from app.api.routes import health as health_route
from app.api.routes import prediction as prediction_route
from app.core.config import Settings
from app.services import fuelcast_vqr, model_registry
from app.services.model_registry import (
    FeatureValidationError, ModelUnavailableError, UnknownModelError,
)

ROOT = Path(__file__).resolve().parents[1] / "artifacts" / fuelcast_vqr.RELATIVE_DIR
ROW = {
    "speed_over_ground": 8.0, "wind_speed": 5.5, "wind_direction": 120.0,
    "wave_height": 1.2, "wave_period": 7.0, "current_speed": 0.8,
}
BODY = {"model_id": "fuelcast_vqr", "fuelcast_inputs": ROW}


@pytest.fixture(autouse=True)
def isolated_vqr_cache():
    previous = model_registry._vqr_bundle, model_registry._vqr_error
    model_registry._vqr_bundle = model_registry._vqr_error = None
    yield
    model_registry._vqr_bundle, model_registry._vqr_error = previous


@pytest.fixture
def bundle_copy(tmp_path):
    destination = tmp_path / fuelcast_vqr.RELATIVE_DIR
    shutil.copytree(ROOT, destination)
    return destination


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _rehash(bundle: Path, name: str) -> None:
    """Keep test manifests coherent when testing post-integrity validation."""
    candidate_path = bundle / "candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["artifact_sha256"][name] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    _write_json(candidate_path, candidate)
    manifest_path = bundle / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename in (name, "candidate_manifest.json"):
        path = bundle / filename
        manifest["packaged_artifacts"][filename]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["packaged_artifacts"][filename]["bytes"] = path.stat().st_size
    manifest["source_candidate_manifest"]["sha256"] = manifest["packaged_artifacts"]["candidate_manifest.json"]["sha256"]
    _write_json(manifest_path, manifest)


class FakeQNN:
    def __init__(self, result=None):
        self.result = np.array([[-0.34391220391691607]]) if result is None else result
        self.calls = []

    def forward(self, angles, weights):
        self.calls.append((np.array(angles), np.array(weights)))
        return self.result


def _loaded(monkeypatch, qnn=None):
    qnn = qnn or FakeQNN()
    monkeypatch.setattr(fuelcast_vqr, "_construct_qnn", lambda: qnn)
    return fuelcast_vqr.load_fuelcast_vqr(Settings())


def test_valid_bundle_initializes_and_predicts_explicit_kg_s(monkeypatch):
    qnn = FakeQNN()
    constructions = []
    monkeypatch.setattr(fuelcast_vqr, "_construct_qnn",
                        lambda: constructions.append(True) or qnn)
    predictor = fuelcast_vqr.load_fuelcast_vqr(Settings())
    assert predictor.metadata()["attempt_id"] == fuelcast_vqr.ATTEMPT_ID
    assert predictor.metadata()["hardware_execution"] is False
    first = predictor.predict([ROW])
    second = predictor.predict([ROW])
    assert first == second
    assert first["model_id"] == "fuelcast_vqr"
    assert first["run_id"] == fuelcast_vqr.RUN_ID
    assert first["attempt_id"] == fuelcast_vqr.ATTEMPT_ID
    assert first["fuel_rate_unit"] == "kg/s"
    assert first["fuel_rates"] == pytest.approx([0.7845485260467616], abs=1e-12)
    assert len(qnn.calls) == 2
    assert len(constructions) == 1
    assert qnn.calls[0][0].shape == (1, 6)
    assert qnn.calls[0][1].shape == (12,)


def test_corrupt_pickle_rejected_before_joblib_load(bundle_copy, monkeypatch):
    with (bundle_copy / "vqr_preprocessor.pkl").open("ab") as stream:
        stream.write(b"tampered")
    called = []
    monkeypatch.setattr("joblib.load", lambda *args: called.append(args))
    with pytest.raises(ModelUnavailableError, match="integrity"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))
    assert called == []


def test_manifest_path_traversal_rejected_before_pickle(bundle_copy, monkeypatch):
    manifest_path = bundle_copy / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["packaged_artifacts"]["weights.npz"]["filename"] = "../weights.npz"
    _write_json(manifest_path, manifest)
    called = []
    monkeypatch.setattr("joblib.load", lambda *args: called.append(args))
    with pytest.raises(ModelUnavailableError, match="filename"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))
    assert called == []


@pytest.mark.parametrize("field,value", [
    ("run_id", "wrong"), ("mode", "quick"), ("attempt_id", "wrong"),
    ("hardware_execution", True),
])
def test_wrong_identity_or_hardware_metadata_rejected(bundle_copy, field, value):
    path = bundle_copy / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    _write_json(path, manifest)
    with pytest.raises(ModelUnavailableError, match="contract"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))


def test_feature_order_mismatch_rejected(bundle_copy):
    path = bundle_copy / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["raw_features"].reverse()
    _write_json(path, manifest)
    with pytest.raises(ModelUnavailableError, match="contract"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))


@pytest.mark.parametrize("weights", [np.zeros(11), np.array([float("nan")] + [0.0] * 11)])
def test_bad_weights_rejected_after_hash_verification(bundle_copy, weights):
    np.savez_compressed(bundle_copy / "weights.npz", weights=weights)
    _rehash(bundle_copy, "weights.npz")
    with pytest.raises(ModelUnavailableError, match="twelve finite"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))


def test_incompatible_circuit_configuration_rejected(bundle_copy):
    path = bundle_copy / "circuit_config.json"
    config = json.loads(path.read_text())
    config["ansatz"]["repetitions"] = 2
    _write_json(path, config)
    _rehash(bundle_copy, "circuit_config.json")
    with pytest.raises(ModelUnavailableError, match="circuit"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))


def test_inference_manifest_circuit_disagreement_rejected(bundle_copy):
    path = bundle_copy / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["circuit"]["observable"] = "different observable"
    _write_json(path, manifest)
    with pytest.raises(ModelUnavailableError, match="circuit"):
        fuelcast_vqr.load_fuelcast_vqr(Settings(artifacts_dir=bundle_copy.parents[4]))


@pytest.mark.parametrize("value", [
    np.ones((1, 5)), np.array([[float("nan")] * 6]),
])
def test_invalid_transformed_values_fail_safely(monkeypatch, value):
    predictor = _loaded(monkeypatch)
    predictor.preprocessor = SimpleNamespace(transform=lambda raw: value)
    with pytest.raises(ModelUnavailableError, match="preprocessor"):
        predictor.predict([ROW])


@pytest.mark.parametrize("value", [
    np.zeros((2, 1)), np.array([[float("nan")]]),
    np.array([[float("inf")]]), np.array([[1.1]]),
])
def test_invalid_expectation_fails_safely(monkeypatch, value):
    predictor = _loaded(monkeypatch, FakeQNN(value))
    with pytest.raises(ModelUnavailableError, match="simulator"):
        predictor.predict([ROW])


def test_negative_inverse_scaled_rate_is_rejected(monkeypatch):
    predictor = _loaded(monkeypatch)
    predictor.target_scaler = SimpleNamespace(inverse_transform=lambda value: np.array([[-0.1]]))
    with pytest.raises(ModelUnavailableError, match="inverse scaling"):
        predictor.predict([ROW])


@pytest.mark.parametrize("change", [
    {"wind_direction": 360.0}, {"wave_period": 0.0},
    {"speed_over_ground": -1.0}, {"extra": 1.0},
    {"wind_direction_relative": 120.0}, {"sailing_speed": 15.5},
])
def test_adapter_rejects_invalid_or_unrelated_fields(monkeypatch, change):
    predictor = _loaded(monkeypatch)
    with pytest.raises(FeatureValidationError):
        predictor.predict([{**ROW, **change}])


def test_registry_keeps_legacy_default_and_caches_vqr(monkeypatch):
    selected = object()
    calls = []
    monkeypatch.setattr(fuelcast_vqr, "load_fuelcast_vqr", lambda settings: calls.append(settings) or selected)
    assert model_registry.get_predictor("fuelcast_vqr", force_reload=True) is selected
    assert model_registry.get_predictor("fuelcast_vqr") is selected
    assert len(calls) == 1
    assert model_registry.get_predictor() is model_registry.get_bundle()
    with pytest.raises(UnknownModelError):
        model_registry.get_predictor("quick")


def test_failed_initialization_leaves_no_partial_adapter(monkeypatch):
    def fail(settings):
        raise ModelUnavailableError("missing /private/model.pkl")

    monkeypatch.setattr(fuelcast_vqr, "load_fuelcast_vqr", fail)
    with pytest.raises(ModelUnavailableError):
        model_registry.get_predictor("fuelcast_vqr", force_reload=True)
    assert model_registry._vqr_bundle is None
    assert model_registry.vqr_status()["error"] == "Model health check failed."


def test_missing_simulator_dependency_is_an_unavailable_model(monkeypatch):
    def unavailable():
        raise ModelUnavailableError("No module named qiskit at /private/runtime")

    monkeypatch.setattr(fuelcast_vqr, "_construct_qnn", unavailable)
    with pytest.raises(ModelUnavailableError, match="qiskit"):
        model_registry.get_predictor("fuelcast_vqr", force_reload=True)
    assert model_registry._vqr_bundle is None
    assert model_registry.vqr_status()["error"] == "Model health check failed."


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(prediction_route.router, prefix="/api/v1")
    app.dependency_overrides[deps.get_db] = lambda: object()
    app.dependency_overrides[deps.get_current_user] = lambda: {"id": "user", "role": "operator"}

    class FakeRunRepository:
        def __init__(self, db, kind):
            assert kind == "prediction"

        def save(self, *, user_id, request, response):
            assert response["voyage_fuel_tonnes"] is None
            return "vqr-prediction-id"

    monkeypatch.setattr(prediction_route, "RunRepository", FakeRunRepository)
    return TestClient(app)


def test_direct_vqr_api_uses_only_explicit_inputs(client, monkeypatch):
    seen = []

    class Predictor:
        def predict(self, rows):
            seen.extend(rows)
            return {"model_id": "fuelcast_vqr", "run_id": fuelcast_vqr.RUN_ID,
                    "attempt_id": fuelcast_vqr.ATTEMPT_ID,
                    "fuel_rates": [0.25], "fuel_rate_unit": "kg/s", "unit_verified": True}

        def metadata(self):
            return {"model_id": "fuelcast_vqr", "execution_type": "exact quantum simulator"}

    monkeypatch.setattr(prediction_route, "get_predictor", lambda model_id: Predictor())
    response = client.post("/api/v1/prediction/fuel", json=BODY)
    assert response.status_code == 200
    data = response.json()
    assert seen == [ROW]
    assert data["model_id"] == "fuelcast_vqr"
    assert data["model_run_id"] == fuelcast_vqr.RUN_ID
    assert data["attempt_id"] == fuelcast_vqr.ATTEMPT_ID
    assert data["execution_type"] == "exact quantum simulator"
    assert data["fuel_rate"] == 0.25 and data["fuel_rate_unit"] == "kg/s"
    assert data["voyage_fuel_tonnes"] is None and data["duration_hours"] is None
    assert data["prediction_id"] == "vqr-prediction-id"
    assert "hardware" in data["note"]


@pytest.mark.parametrize("body", [
    {"model_id": "fuelcast_vqr"},
    {"model_id": "fuelcast_vqr", "fuelcast_inputs": {**ROW, "wind_direction": 360.0}},
    {"model_id": "fuelcast_vqr", "fuelcast_inputs": {**ROW, "wind_direction_relative": 90.0}},
    {"model_id": "fuelcast_vqr", "fuelcast_inputs": {**ROW, "sailing_speed": 15.5}},
    {"model_id": "fuelcast_vqr", "fuelcast_inputs": ROW, "sailing_speed": 15.5},
    {"model_id": "unknown"},
])
def test_vqr_api_rejects_missing_invalid_and_legacy_mappings(client, body):
    assert client.post("/api/v1/prediction/fuel", json=body).status_code == 422


def test_vqr_api_errors_hide_paths(client, monkeypatch):
    def missing(model_id):
        raise ModelUnavailableError("/private/secret/artifact.pkl")

    monkeypatch.setattr(prediction_route, "get_predictor", missing)
    response = client.post("/api/v1/prediction/fuel", json=BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == "Selected prediction model is unavailable."
    assert "/private/" not in response.text

    class Predictor:
        def predict(self, rows):
            raise ModelUnavailableError("/private/secret/prediction")

    monkeypatch.setattr(prediction_route, "get_predictor", lambda model_id: Predictor())
    response = client.post("/api/v1/prediction/fuel", json=BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == "Prediction could not be completed."
    assert "/private/" not in response.text


def test_metadata_and_health_describe_vqr_without_quick(monkeypatch):
    predictor = SimpleNamespace(metadata=lambda: {
        "model_id": "fuelcast_vqr", "run_id": fuelcast_vqr.RUN_ID,
        "attempt_id": fuelcast_vqr.ATTEMPT_ID, "target_unit": "kg/s",
        "execution_type": "exact quantum simulator", "hardware_execution": False,
    })
    monkeypatch.setattr(model_registry, "get_predictor", lambda model_id: predictor)
    status = model_registry.vqr_status()
    assert status["loaded"] is True and status["api_available"] is True
    assert status["mode"] == "normal"
    assert "quick" not in status
    monkeypatch.setattr(health_route, "get_db", lambda: SimpleNamespace(healthy=lambda: True))
    health = health_route.health()
    assert health.models["fuelcast_vqr"]["execution_type"] == "exact quantum simulator"
    assert health.models["fuelcast_vqr"]["hardware_execution"] is False


def test_real_packaged_normal_candidate_matches_runtime_reference():
    predictor = fuelcast_vqr.load_fuelcast_vqr(Settings())
    first = predictor.predict([ROW])["fuel_rates"][0]
    second = predictor.predict([ROW])["fuel_rates"][0]
    assert first == second
    assert first == pytest.approx(0.7845485260467616, rel=1e-12, abs=1e-12)
