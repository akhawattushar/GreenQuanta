"""One-row FuelCast VQR inference using the verified normal simulator export."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.core.config import Settings
from app.schemas.models import FuelCastInputs
from app.services.model_registry import FeatureValidationError, ModelUnavailableError

MODEL_ID = "fuelcast_vqr"
RUN_ID = "fuelcast-phase1-20260928-002"
MODE = "normal"
ATTEMPT_ID = "5ef14aa8c84e4834ba9992eee821ab5a"
RELATIVE_DIR = Path("fuelcast/vqr") / RUN_ID / MODE / ATTEMPT_ID
RAW_FEATURES = (
    "speed_over_ground", "wind_speed", "wind_direction",
    "wave_height", "wave_period", "current_speed",
)
RAW_UNITS = {
    "speed_over_ground": "m/s", "wind_speed": "m/s",
    "wind_direction": "degrees", "wave_height": "m",
    "wave_period": "s", "current_speed": "m/s",
}
TARGET = "fuel_consumption_kg_s"
TARGET_UNIT = "kg/s"
PACKAGED = frozenset({
    "weights.npz", "circuit_config.json", "vqr_preprocessor.pkl",
    "vqr_target_scaler.pkl", "feature_schema.json",
    "candidate_manifest.json", "validation_metrics.json",
})
EXPECTED_CIRCUIT = {
    "algorithm": "VQR",
    "num_qubits": 6,
    "feature_map": {
        "class": "QuantumCircuit", "encoding": "RY", "repetitions": 1,
        "input_parameters": 6, "qubit_feature_order": list(RAW_FEATURES),
        "angle_ranges": [[0, "pi"], [0, "pi"], [0, "2pi"],
                         [0, "pi"], [0, "pi"], [0, "pi"]],
    },
    "ansatz": {
        "factory": "real_amplitudes", "repetitions": 1, "entanglement": "linear",
        "gate": "CX", "weights": 12, "skip_final_rotation_layer": False,
    },
    "observable": {"name": "mean_single_qubit_Z", "terms": 6, "coefficient": 1 / 6},
    "estimator": "QMLEstimator", "default_precision": 0.0,
    "backend": "simulator", "real_quantum_hardware": False,
}
EXPECTED_MANIFEST_CIRCUIT = {
    "num_qubits": 6,
    "weight_shape": [12],
    "feature_map": "One RY rotation on each qubit: x[0] through x[5] follow raw_features order.",
    "ansatz": "Qiskit real_amplitudes with theta[0] through theta[11], one repetition and a final rotation layer.",
    "entanglement": "linear",
    "cnot_direction": "0->1, 1->2, 2->3, 3->4, 4->5",
    "ansatz_repetitions": 1,
    "observable": ("Mean of the six single-qubit Z observables, each with coefficient 1/6; "
                   "Qiskit Pauli labels use the rightmost character for qubit 0."),
    "estimator": "qiskit_machine_learning.primitives.QMLEstimator through EstimatorQNN",
    "default_precision": 0.0,
    "inference_batch_size": 256,
}


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelUnavailableError(f"FuelCast VQR metadata {path.name} cannot be read: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelUnavailableError(f"FuelCast VQR metadata {path.name} must be an object.")
    return value


def _verified_path(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or name not in PACKAGED
            or Path(name).parts != (name,)):
        raise ModelUnavailableError("FuelCast VQR manifest contains an invalid artifact path.")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ModelUnavailableError(f"FuelCast VQR artifact {name} is missing or is a link.")
    try:
        if path.resolve(strict=True).parent != root:
            raise ModelUnavailableError("FuelCast VQR artifact escapes its bundle.")
    except OSError as exc:
        raise ModelUnavailableError(f"FuelCast VQR artifact {name} cannot be resolved: {exc}") from exc
    return path


def _verify_bundle(root: Path) -> tuple[dict, dict]:
    manifest_path = root / "inference_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ModelUnavailableError("FuelCast VQR inference manifest is missing or is a link.")
    manifest = _read_json(manifest_path)
    if (manifest.get("schema_version") != 1
            or manifest.get("model_id") != MODEL_ID
            or manifest.get("model_family") != "variational_quantum_regressor"
            or manifest.get("run_id") != RUN_ID
            or manifest.get("mode") != MODE
            or manifest.get("attempt_id") != ATTEMPT_ID
            or manifest.get("application_status") != "candidate"
            or manifest.get("execution_type") != "exact quantum simulator"
            or manifest.get("hardware_execution") is not False
            or manifest.get("raw_features") != list(RAW_FEATURES)
            or manifest.get("feature_units") != RAW_UNITS
            or not isinstance(manifest.get("target"), dict)
            or manifest["target"].get("name") != TARGET
            or manifest["target"].get("unit") != TARGET_UNIT):
        raise ModelUnavailableError("FuelCast VQR inference manifest contract differs from the selected model.")

    entries = manifest.get("packaged_artifacts")
    if not isinstance(entries, dict) or set(entries) != PACKAGED:
        raise ModelUnavailableError("FuelCast VQR inference manifest artifact set is invalid.")
    for name, entry in entries.items():
        if not isinstance(entry, dict) or entry.get("filename") != name:
            raise ModelUnavailableError("FuelCast VQR manifest contains an invalid artifact filename.")
        path = _verified_path(root, name)
        expected_bytes = entry.get("bytes")
        expected_hash = entry.get("sha256")
        if (not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool)
                or expected_bytes < 1 or not isinstance(expected_hash, str)
                or len(expected_hash) != 64):
            raise ModelUnavailableError(f"FuelCast VQR integrity metadata is invalid for {name}.")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ModelUnavailableError(f"FuelCast VQR artifact {name} cannot be read: {exc}") from exc
        if path.stat().st_size != expected_bytes or digest != expected_hash:
            raise ModelUnavailableError(f"FuelCast VQR artifact integrity check failed: {name}.")
    source = manifest.get("source_candidate_manifest")
    if (not isinstance(source, dict) or source.get("filename") != "candidate_manifest.json"
            or source.get("sha256") != entries["candidate_manifest.json"]["sha256"]):
        raise ModelUnavailableError("FuelCast VQR source manifest identity is invalid.")
    candidate = _read_json(root / "candidate_manifest.json")
    if (candidate.get("candidate") != "vqr" or candidate.get("backend") != "simulator"
            or candidate.get("real_quantum_hardware") is not False
            or candidate.get("run_id") != RUN_ID or candidate.get("mode") != MODE
            or candidate.get("attempt_id") != ATTEMPT_ID
            or candidate.get("feature_order") != list(RAW_FEATURES)
            or candidate.get("target") != TARGET or candidate.get("target_unit") != TARGET_UNIT):
        raise ModelUnavailableError("FuelCast VQR candidate metadata disagrees with the inference manifest.")
    candidate_hashes = candidate.get("artifact_sha256")
    if not isinstance(candidate_hashes, dict) or any(
            candidate_hashes.get(name) != entries[name]["sha256"]
            for name in PACKAGED - {"candidate_manifest.json"}):
        raise ModelUnavailableError("FuelCast VQR candidate artifact hashes disagree.")
    schema = _read_json(root / "feature_schema.json")
    if (schema.get("raw_features") != list(RAW_FEATURES)
            or schema.get("vqr_features") != list(RAW_FEATURES)
            or schema.get("target") != TARGET
            or not isinstance(schema.get("units"), dict)
            or {name: schema["units"].get(name) for name in RAW_FEATURES} != RAW_UNITS
            or schema["units"].get(TARGET) != TARGET_UNIT):
        raise ModelUnavailableError("FuelCast VQR feature schema differs from the inference manifest.")
    config = _read_json(root / "circuit_config.json")
    declared = manifest.get("circuit")
    if config != EXPECTED_CIRCUIT or declared != EXPECTED_MANIFEST_CIRCUIT:
        raise ModelUnavailableError("FuelCast VQR circuit configuration differs from the inference manifest.")
    return manifest, config


def _construct_qnn():
    """Match SIH's audited RY, real_amplitudes, and little-endian Z construction."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.circuit.library import real_amplitudes
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.primitives import QMLEstimator
    except (ImportError, OSError) as exc:
        raise ModelUnavailableError(f"FuelCast VQR simulator dependency is unavailable: {exc}") from exc
    try:
        feature_map = QuantumCircuit(6, name="fuelcast_ry")
        for qubit, angle in enumerate(ParameterVector("x", 6)):
            feature_map.ry(angle, qubit)
        ansatz = real_amplitudes(
            6, reps=1, entanglement="linear",
            skip_final_rotation_layer=False, parameter_prefix="theta",
        )
        if ([str(p) for p in feature_map.parameters] != [f"x[{n}]" for n in range(6)]
                or [str(p) for p in ansatz.parameters] != [f"theta[{n}]" for n in range(12)]):
            raise ModelUnavailableError("FuelCast VQR circuit parameter order changed.")
        chain = [
            [ansatz.find_bit(qubit).index for qubit in operation.qubits]
            for operation in ansatz.decompose().data if operation.operation.name == "cx"
        ]
        if chain != [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]:
            raise ModelUnavailableError("FuelCast VQR entanglement order changed.")
        paulis = [
            ("I" * (5 - qubit) + "Z" + "I" * qubit, 1 / 6)
            for qubit in range(6)
        ]
        observable = SparsePauliOp.from_list(paulis)
        return EstimatorQNN(
            circuit=feature_map.compose(ansatz),
            estimator=QMLEstimator(default_precision=0.0),
            observables=observable,
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
        )
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"FuelCast VQR circuit cannot be constructed: {exc}") from exc


@dataclass
class FuelCastVQR:
    preprocessor: Any
    target_scaler: Any
    weights: Any
    qnn: Any
    _prediction_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def metadata(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "display_name": "FuelCast Variational Quantum Regressor",
            "model_family": "variational_quantum_regressor",
            "run_id": RUN_ID,
            "mode": MODE,
            "attempt_id": ATTEMPT_ID,
            "raw_feature_order": list(RAW_FEATURES),
            "raw_units": dict(RAW_UNITS),
            "target_column": TARGET,
            "target_unit": TARGET_UNIT,
            "target_unit_verified": True,
            "execution_type": "exact quantum simulator",
            "hardware_execution": False,
            "limitations": [
                "Wind-direction absolute/relative and from/toward convention is unverified.",
                "Only an explicit FuelCast environmental snapshot is used.",
                "Validation metrics are not held-out test metrics.",
                "Voyage, optimization, and scenario VQR integration is pending.",
            ],
        }

    def predict(self, rows: Sequence[dict]) -> dict:
        if len(rows) != 1 or not isinstance(rows[0], dict):
            raise FeatureValidationError("FuelCast VQR requires exactly one raw feature row.")
        row = rows[0]
        missing = [name for name in RAW_FEATURES if name not in row]
        extra = [name for name in row if name not in RAW_FEATURES]
        if missing or extra:
            raise FeatureValidationError(
                "FuelCast VQR requires exactly six raw inputs; "
                f"missing: {', '.join(missing) or 'none'}; unexpected: {', '.join(extra) or 'none'}."
            )
        if any(isinstance(row[name], bool) or not isinstance(row[name], (int, float))
               or not math.isfinite(row[name]) for name in RAW_FEATURES):
            raise FeatureValidationError("FuelCast VQR inputs must be finite numbers.")
        from pydantic import ValidationError
        try:
            validated = FuelCastInputs.model_validate(row).model_dump()
        except ValidationError as exc:
            fields = sorted({str(error["loc"][0]) for error in exc.errors() if error["loc"]})
            raise FeatureValidationError(
                f"FuelCast VQR input range is invalid: {', '.join(fields)}."
            ) from exc

        import numpy as np

        try:
            raw = np.asarray([[validated[name] for name in RAW_FEATURES]], dtype=np.float64)
            angles = np.asarray(self.preprocessor.transform(raw))
            if angles.shape != (1, 6) or not np.isrealobj(angles) or not np.issubdtype(angles.dtype, np.number):
                raise ModelUnavailableError("FuelCast VQR preprocessor returned an invalid shape or type.")
            if not np.isfinite(angles).all():
                raise ModelUnavailableError("FuelCast VQR preprocessor returned non-finite values.")
            bounds = np.array([np.pi, np.pi, 2 * np.pi, np.pi, np.pi, np.pi])
            if (angles < -1e-12).any() or (angles > bounds + 1e-12).any():
                raise ModelUnavailableError("FuelCast VQR preprocessor returned angles outside the saved ranges.")
            with self._prediction_lock:
                expectation = np.asarray(self.qnn.forward(angles, self.weights))
            if expectation.shape != (1, 1) or not np.isrealobj(expectation) or not np.issubdtype(expectation.dtype, np.number):
                raise ModelUnavailableError("FuelCast VQR simulator returned an invalid prediction shape or type.")
            if not np.isfinite(expectation).all() or (np.abs(expectation) > 1 + 1e-8).any():
                raise ModelUnavailableError("FuelCast VQR simulator returned an invalid expectation.")
            fuel = np.asarray(self.target_scaler.inverse_transform(expectation))
            if (fuel.shape != (1, 1) or not np.isrealobj(fuel)
                    or not np.issubdtype(fuel.dtype, np.number)
                    or not np.isfinite(fuel).all() or (fuel < 0).any()):
                raise ModelUnavailableError("FuelCast VQR inverse scaling returned an invalid fuel rate.")
            return {
                "model_id": MODEL_ID, "run_id": RUN_ID, "attempt_id": ATTEMPT_ID,
                "fuel_rates": [float(fuel[0, 0])], "fuel_rate_unit": TARGET_UNIT,
                "unit_verified": True,
            }
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"FuelCast VQR prediction failed: {exc}") from exc


def load_fuelcast_vqr(settings: Settings) -> FuelCastVQR:
    try:
        root = (settings.artifacts_dir / RELATIVE_DIR).resolve(strict=True)
    except OSError as exc:
        raise ModelUnavailableError(f"FuelCast VQR bundle is unavailable: {exc}") from exc
    if not root.is_dir():
        raise ModelUnavailableError("FuelCast VQR bundle is not a directory.")
    try:
        _verify_bundle(root)  # No pickle or quantum import occurs before integrity checks.
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"FuelCast VQR metadata is invalid: {exc}") from exc
    try:
        import joblib
        import numpy as np
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from greenfleet.ml_pipeline.transformation.fuelcast import FuelCastImputer, VQRAngleEncoder
        with np.load(root / "weights.npz", allow_pickle=False) as saved:
            if set(saved.files) != {"weights"}:
                raise ModelUnavailableError("FuelCast VQR weights artifact has invalid keys.")
            weights = np.asarray(saved["weights"], dtype=np.float64)
        if weights.shape != (12,) or not np.isfinite(weights).all():
            raise ModelUnavailableError("FuelCast VQR weights must be twelve finite values.")
        weights.setflags(write=False)
        preprocessor = joblib.load(root / "vqr_preprocessor.pkl")
        target_scaler = joblib.load(root / "vqr_target_scaler.pkl")
        if (not isinstance(preprocessor, Pipeline)
                or [(name, type(step)) for name, step in preprocessor.steps] != [
                    ("imputer", FuelCastImputer), ("angle", VQRAngleEncoder),
                ]):
            raise ModelUnavailableError("FuelCast VQR fitted preprocessor classes do not match the export.")
        if (not isinstance(target_scaler, MinMaxScaler)
                or target_scaler.feature_range != (-1, 1)
                or target_scaler.n_features_in_ != 1
                or not np.isfinite(target_scaler.data_min_).all()
                or not np.isfinite(target_scaler.data_max_).all()
                or target_scaler.data_max_[0] <= target_scaler.data_min_[0]):
            raise ModelUnavailableError("FuelCast VQR target scaler does not match the export.")
        qnn = _construct_qnn()
        return FuelCastVQR(preprocessor, target_scaler, weights, qnn)
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"FuelCast VQR runtime cannot be initialized: {exc}") from exc
