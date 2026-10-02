"""Offline integrity checks for the packaged normal VQR candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

RUN_ID = "fuelcast-phase1-20260928-002"
ATTEMPT_ID = "5ef14aa8c84e4834ba9992eee821ab5a"
ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/fuelcast/vqr"
    / RUN_ID
    / "normal"
    / ATTEMPT_ID
)
FEATURES = [
    "speed_over_ground", "wind_speed", "wind_direction",
    "wave_height", "wave_period", "current_speed",
]
PACKAGED = {
    "weights.npz", "circuit_config.json", "vqr_preprocessor.pkl",
    "vqr_target_scaler.pkl", "feature_schema.json",
    "candidate_manifest.json", "validation_metrics.json",
}


def _json(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inference_manifest_identifies_normal_candidate():
    manifest = _json("inference_manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["model_id"] == "fuelcast_vqr"
    assert manifest["model_family"] == "variational_quantum_regressor"
    assert (manifest["run_id"], manifest["mode"], manifest["attempt_id"]) == (
        RUN_ID, "normal", ATTEMPT_ID
    )
    assert manifest["application_status"] == "candidate"
    assert manifest["source_candidate_manifest"]["filename"] == "candidate_manifest.json"
    assert manifest["source_candidate_manifest"]["sha256"] == _sha256(ROOT / "candidate_manifest.json")
    assert manifest["reference_source"]["source_root_relative_path"] == (
        "greenfleet/ml_pipeline/quantum/fuelcast_vqr_model.py"
    )
    assert manifest["reference_source"]["identity_limitation"]


def test_packaged_files_match_both_manifests():
    manifest = _json("inference_manifest.json")
    candidate = _json("candidate_manifest.json")
    assert set(manifest["packaged_artifacts"]) == PACKAGED
    assert {path.name for path in ROOT.iterdir() if path.is_file()} == PACKAGED | {"inference_manifest.json"}
    for name, entry in manifest["packaged_artifacts"].items():
        assert entry["filename"] == name
        path = ROOT / name
        assert path.is_file()
        assert entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == _sha256(path)
        if name in candidate["artifact_sha256"]:
            assert entry["sha256"] == candidate["artifact_sha256"][name]
    assert candidate["run_id"] == RUN_ID
    assert candidate["mode"] == "normal"
    assert candidate["attempt_id"] == ATTEMPT_ID


def test_feature_target_and_circuit_contract():
    manifest = _json("inference_manifest.json")
    schema = _json("feature_schema.json")
    circuit = _json("circuit_config.json")
    assert manifest["raw_features"] == schema["raw_features"] == schema["vqr_features"] == FEATURES
    assert manifest["feature_units"] == {name: schema["units"][name] for name in FEATURES}
    assert [manifest["feature_units"][name] for name in FEATURES] == [
        "m/s", "m/s", "degrees", "m", "s", "m/s",
    ]
    assert manifest["target"]["name"] == schema["target"] == "fuel_consumption_kg_s"
    assert manifest["target"]["unit"] == schema["units"][schema["target"]] == "kg/s"
    assert manifest["target"]["source_field"] == "Consumer_Total_MomentaryFuel"
    assert "inverse_transform" in manifest["target"]["inverse_scaling"]
    assert manifest["circuit"]["num_qubits"] == circuit["num_qubits"] == 6
    assert manifest["circuit"]["weight_shape"] == [circuit["ansatz"]["weights"]] == [12]
    assert circuit["feature_map"]["qubit_feature_order"] == FEATURES
    assert circuit["feature_map"]["encoding"] == "RY"
    assert circuit["ansatz"]["repetitions"] == manifest["circuit"]["ansatz_repetitions"] == 1
    assert manifest["circuit"]["cnot_direction"] == "0->1, 1->2, 2->3, 3->4, 4->5"
    assert manifest["circuit"]["inference_batch_size"] == 256


def test_weights_and_fitted_target_scaler():
    with np.load(ROOT / "weights.npz", allow_pickle=False) as saved:
        assert saved.files == ["weights"]
        weights = saved["weights"]
    assert weights.shape == (12,)
    assert np.isfinite(weights).all()
    scaler = joblib.load(ROOT / "vqr_target_scaler.pkl")
    assert scaler.feature_range == (-1, 1)
    assert scaler.n_features_in_ == 1
    assert np.isfinite(scaler.data_min_).all()
    assert np.isfinite(scaler.data_max_).all()
    assert scaler.data_max_[0] > scaler.data_min_[0]


def test_validation_and_simulator_labels():
    manifest = _json("inference_manifest.json")
    candidate = _json("candidate_manifest.json")
    metrics = _json("validation_metrics.json")
    assert manifest["execution_type"] == "exact quantum simulator"
    assert manifest["simulator_classification"] == "QMLEstimator statevector simulation"
    assert manifest["hardware_execution"] is False
    assert candidate["real_quantum_hardware"] is False
    assert manifest["validation"]["split"] == "validation"
    assert manifest["validation"]["sample_count"] == metrics["validation_rows"] == 26096
    assert manifest["validation"]["metrics"] == {
        "mae_kg_s": metrics["validation_metrics"]["mae"],
        "rmse_kg_s": metrics["validation_metrics"]["rmse"],
        "r2": metrics["validation_metrics"]["r2"],
    }
    assert "not held-out test metrics" in manifest["validation"]["limitation"]
    assert manifest["dependency_versions"]["python"] == "3.11.16"
    assert "not recorded" in manifest["dependency_versions"]["python_provenance"]
    assert manifest["greenquanta_runtime"]["compatibility"] == "unverified"
    assert "unverified" in manifest["wind_direction_convention"]
    assert set(manifest["absent_model_features"]) == {"cargo_load", "fuel_type", "shore_power"}
    assert manifest["environmental_snapshot_limitation"]


def test_quick_is_excluded_from_the_application_bundle():
    manifest = _json("inference_manifest.json")
    assert not (ROOT.parent.parent / "quick").exists()
    assert "smoke test" in manifest["quick_candidate_exclusion"]
