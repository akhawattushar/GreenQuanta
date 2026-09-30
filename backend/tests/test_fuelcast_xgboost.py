"""FuelCast stays isolated until the live request has verified input semantics."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api.routes.prediction import model_info, predict_fuel
from app.core.config import Settings
from app.schemas.models import PredictionRequest
from app.services.fuelcast_xgboost import (
    RAW_FEATURES, RELATIVE_DIR, TARGET_UNIT, TRANSFORMED_FEATURES,
    FuelCastXGBoost, load_fuelcast_xgboost,
)
from app.services.model_registry import (
    FeatureValidationError, ModelUnavailableError, UnknownModelError,
    get_bundle, get_predictor,
)

TRANSFORMED = (
    "speed_over_ground", "wind_speed", "wind_direction_sin",
    "wind_direction_cos", "wave_height", "wave_period", "current_speed",
)
ROW = dict(zip(RAW_FEATURES, (7.0, 8.0, 90.0, 2.0, 5.0, 1.0)))


def _request(model_id="legacy"):
    return PredictionRequest.model_validate({
        "model_id": model_id,
        "sailing_speed": 15.5,
        "vessel": {"vessel_type": "Tanker Ship", "displacement": 12, "trim": 0},
        "environment": {
            "wind_speed": 14, "wind_direction_relative": 90,
            "combined_wave_height": 3, "combined_wave_period": 5,
            "sea_current_speed": 0.5, "sea_current_direction_relative": 90,
            "sea_water_temperature": 17,
        },
    })


def test_legacy_remains_default_and_unknown_ids_fail():
    assert _request().model_id == "legacy"
    assert get_predictor() is get_bundle()
    with pytest.raises(UnknownModelError, match="Unknown model ID"):
        get_predictor("surprise")
    with pytest.raises(HTTPException) as exc:
        predict_fuel(_request("surprise"), {}, None)
    assert exc.value.status_code == 422


def test_explicit_fuelcast_model_selection(monkeypatch):
    from app.services import fuelcast_xgboost, model_registry

    selected = object()
    monkeypatch.setattr(fuelcast_xgboost, "load_fuelcast_xgboost", lambda settings: selected)
    monkeypatch.setattr(model_registry, "_fuelcast_bundle", None)
    monkeypatch.setattr(model_registry, "_fuelcast_error", None)
    assert get_predictor("fuelcast_xgboost") is selected
    assert get_predictor("fuelcast_xgboost") is selected
    assert get_predictor() is get_bundle()


def test_fuelcast_request_does_not_reuse_legacy_fields():
    with pytest.raises(ValidationError, match="requires fuelcast_inputs"):
        _request("fuelcast_xgboost")


def test_model_info_keeps_legacy_and_reports_fuelcast_gate():
    info = model_info({})
    assert info.loaded is True
    assert info.models["legacy"]["loaded"] is True
    assert info.models["fuelcast_xgboost"]["input_contract"] == "explicit_fuelcast_inputs"
    assert info.models["fuelcast_xgboost"]["api_available"] == info.models["fuelcast_xgboost"]["loaded"]
    assert info.models["fuelcast_xgboost"]["target_unit"] == "kg/s"


def _write_export(tmp_path):
    root = tmp_path / RELATIVE_DIR
    (root / "model").mkdir(parents=True, exist_ok=True)
    schema = {"raw_features": list(RAW_FEATURES), "classical_features": list(TRANSFORMED)}
    candidate = {"run_id": RELATIVE_DIR.name, "feature_order": list(TRANSFORMED), "target_unit": TARGET_UNIT}
    model = {"learner": {"feature_names": list(TRANSFORMED)}}
    for name, data in (("feature_schema.json", schema), ("model/candidate_manifest.json", candidate),
                       ("model/model.json", model), ("metrics.json", {})):
        (root / name).write_text(json.dumps(data))
    (root / "preprocessor.pkl").write_bytes(b"mock pickle")
    names = ("preprocessor.pkl", "feature_schema.json", "model/model.json",
             "model/candidate_manifest.json", "metrics.json")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    (root / "model_manifest.json").write_text(json.dumps({
        "run_id": RELATIVE_DIR.name, "target": "fuel_consumption_kg_s", "artifact_sha256": hashes,
    }))
    return root


def test_json_loading_prediction_order_and_kg_s_label(monkeypatch, tmp_path):
    root = _write_export(tmp_path)
    calls = {}

    class Preprocessor:
        def transform(self, frame):
            calls["raw_order"] = list(frame.columns)
            return np.ones((len(frame), len(TRANSFORMED)))

    class XGBRegressor:
        def load_model(self, path):
            calls["loaded_path"] = path

        def get_booster(self):
            return SimpleNamespace(feature_names=list(TRANSFORMED))

        def predict(self, matrix):
            assert matrix.shape[1] == 7
            calls["transformed_order"] = list(matrix.columns)
            return np.full(matrix.shape[0], 0.25)

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=XGBRegressor))
    monkeypatch.setattr("joblib.load", lambda path: Preprocessor())
    predictor = load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))
    assert calls["loaded_path"] == str(root / "model/model.json")
    result = predictor.predict([ROW, ROW])
    assert result["fuel_rates"] == [0.25, 0.25]
    assert result["fuel_rate_unit"] == "kg/s"
    assert result["unit_verified"] is True
    assert calls["raw_order"] == list(RAW_FEATURES)
    assert calls["transformed_order"] == list(TRANSFORMED)
    assert predictor.metadata()["target_unit"] == "kg/s"


def test_fuelcast_rejects_missing_extra_and_invalid_features(monkeypatch, tmp_path):
    _write_export(tmp_path)
    class Preprocessor:
        def transform(self, frame):
            return np.ones((len(frame), 7))
    class XGBRegressor:
        def load_model(self, path):
            pass
        def get_booster(self):
            return SimpleNamespace(feature_names=list(TRANSFORMED))
        def predict(self, matrix):
            return np.ones(matrix.shape[0])
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=XGBRegressor))
    monkeypatch.setattr("joblib.load", lambda path: Preprocessor())
    predictor = load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))
    for bad in ({k: v for k, v in ROW.items() if k != "wind_direction"},
                {**ROW, "wind_direction_relative": 90},
                {**ROW, "speed_over_ground": float("nan")}):
        with pytest.raises(FeatureValidationError):
            predictor.predict([bad])


def test_mismatched_hash_fails_before_model_load(tmp_path):
    root = _write_export(tmp_path)
    (root / "model/model.json").write_text("{}")
    with pytest.raises(ModelUnavailableError, match="hash mismatch"):
        load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))


def test_missing_and_corrupt_artifacts_fail_clearly(tmp_path):
    root = _write_export(tmp_path)
    (root / "preprocessor.pkl").unlink()
    with pytest.raises(ModelUnavailableError, match="artifact preprocessor.pkl is missing"):
        load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))

    root = _write_export(tmp_path)
    (root / "model_manifest.json").write_text("{")
    with pytest.raises(ModelUnavailableError, match="model_manifest.json cannot be read"):
        load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))


def test_manifest_and_schema_feature_order_must_agree(tmp_path):
    root = _write_export(tmp_path)
    candidate_path = root / "model/candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["feature_order"] = list(reversed(TRANSFORMED_FEATURES))
    candidate_path.write_text(json.dumps(candidate))
    bundle_path = root / "model_manifest.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["artifact_sha256"]["model/candidate_manifest.json"] = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    bundle_path.write_text(json.dumps(bundle))
    with pytest.raises(ModelUnavailableError, match="metadata disagree"):
        load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))


def test_preprocessor_output_width_must_match_manifest(monkeypatch, tmp_path):
    _write_export(tmp_path)

    class WrongWidthPreprocessor:
        def transform(self, frame):
            return np.ones((len(frame), 6))

    class XGBRegressor:
        def load_model(self, path):
            pass

        def get_booster(self):
            return SimpleNamespace(feature_names=list(TRANSFORMED))

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=XGBRegressor))
    monkeypatch.setattr("joblib.load", lambda path: WrongWidthPreprocessor())
    predictor = load_fuelcast_xgboost(Settings(artifacts_dir=tmp_path))
    with pytest.raises(ModelUnavailableError, match="7 columns"):
        predictor.predict([ROW])


@pytest.mark.parametrize("matrix", [
    np.ones(7),
    np.ones((2, 7)),
    np.ones((1, 6)),
    np.array([[1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0]]),
    np.array([[1.0, 2.0, float("inf"), 4.0, 5.0, 6.0, 7.0]]),
    np.array([["1", "2", "3", "4", "5", "6", "7"]]),
])
def test_invalid_transformed_matrix_is_rejected(matrix):
    predictor = FuelCastXGBoost(
        model=SimpleNamespace(predict=lambda frame: np.array([0.25])),
        preprocessor=SimpleNamespace(transform=lambda frame: matrix),
        transformed_features=TRANSFORMED_FEATURES,
    )
    with pytest.raises(ModelUnavailableError, match="preprocessor"):
        predictor.predict([ROW])


@pytest.mark.parametrize("values", [
    np.array([float("nan")]), np.array([float("inf")]),
    np.array(["bad"]), np.array([0.1, 0.2]),
])
def test_invalid_prediction_is_rejected(values):
    predictor = FuelCastXGBoost(
        model=SimpleNamespace(predict=lambda frame: values),
        preprocessor=SimpleNamespace(transform=lambda frame: np.ones((1, 7))),
        transformed_features=TRANSFORMED_FEATURES,
    )
    with pytest.raises(ModelUnavailableError, match="FuelCast model"):
        predictor.predict([ROW])
