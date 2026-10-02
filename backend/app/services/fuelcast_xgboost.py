"""Isolated FuelCast XGBoost inference for rows in the export's own units."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.core.config import Settings
from app.services.model_registry import FeatureValidationError, ModelUnavailableError

MODEL_ID = "fuelcast_xgboost"
RUN_ID = "fuelcast-phase1-20260928-002"
RELATIVE_DIR = Path("fuelcast/classical/xgboost") / RUN_ID
RAW_FEATURES = (
    "speed_over_ground", "wind_speed", "wind_direction",
    "wave_height", "wave_period", "current_speed",
)
TRANSFORMED_FEATURES = (
    "speed_over_ground", "wind_speed", "wind_direction_sin",
    "wind_direction_cos", "wave_height", "wave_period", "current_speed",
)
TARGET_UNIT = "kg/s"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelUnavailableError(f"FuelCast artifact {path.name} cannot be read: {exc}") from exc


@dataclass
class FuelCastXGBoost:
    model: Any
    preprocessor: Any
    transformed_features: tuple[str, ...]

    def metadata(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "run_id": RUN_ID,
            "model_class": "XGBRegressor",
            "preprocessor_class": type(self.preprocessor).__name__,
            "raw_feature_order": list(RAW_FEATURES),
            "transformed_feature_names": list(self.transformed_features),
            "target_column": "fuel_consumption_kg_s",
            "target_unit": TARGET_UNIT,
            "target_unit_verified": True,
            "raw_units": {
                "speed_over_ground": "m/s", "wind_speed": "m/s",
                "wind_direction": "degrees", "wave_height": "m",
                "wave_period": "s", "current_speed": "m/s",
            },
        }

    def predict(self, rows: Sequence[dict]) -> dict:
        if not rows:
            raise FeatureValidationError("At least one FuelCast feature row is required.")
        for index, row in enumerate(rows):
            missing = [name for name in RAW_FEATURES if name not in row]
            extra = [name for name in row if name not in RAW_FEATURES]
            if missing or extra:
                raise FeatureValidationError(
                    f"Row {index}: FuelCast requires exactly {', '.join(RAW_FEATURES)}; "
                    f"missing: {', '.join(missing) or 'none'}; unexpected: {', '.join(extra) or 'none'}."
                )
            for name in RAW_FEATURES:
                value = row[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise FeatureValidationError(f"Row {index}: {name!r} must be a finite number.")

        import numpy as np
        import pandas as pd

        frame = pd.DataFrame(rows, columns=RAW_FEATURES)
        try:
            matrix = self.preprocessor.transform(frame)
            if hasattr(matrix, "columns") and list(matrix.columns) != list(self.transformed_features):
                raise ModelUnavailableError("FuelCast preprocessor output order differs from the manifest.")
            if hasattr(matrix, "toarray"):
                matrix = matrix.toarray()
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape != (len(rows), len(self.transformed_features)):
                raise ModelUnavailableError(
                    f"FuelCast preprocessor must produce {len(rows)} row(s) and "
                    f"{len(self.transformed_features)} columns; got shape {matrix.shape}."
                )
            if (not np.issubdtype(matrix.dtype, np.number)
                    or not np.isrealobj(matrix)
                    or not np.isfinite(matrix).all()):
                raise ModelUnavailableError("FuelCast preprocessor output must contain only finite real numbers.")
            # The fitted custom imputer has no get_feature_names_out(). The
            # export schema and model JSON agree on this order; provide the
            # names explicitly so XGBoost can validate them at prediction.
            named_matrix = pd.DataFrame(matrix, columns=self.transformed_features)
            values = np.asarray(self.model.predict(named_matrix))
            if values.ndim != 1 or values.shape[0] != len(rows):
                raise ModelUnavailableError("FuelCast model must return one fuel rate per input row.")
            if (not np.issubdtype(values.dtype, np.number)
                    or not np.isrealobj(values)
                    or not np.isfinite(values).all()):
                raise ModelUnavailableError("FuelCast model returned a non-finite or non-numeric fuel rate.")
            if (values < 0).any():
                raise ModelUnavailableError("FuelCast model returned a negative fuel rate.")
            rates = [float(value) for value in values]
            return {"model_id": MODEL_ID, "run_id": RUN_ID,
                    "fuel_rates": rates, "fuel_rate_unit": TARGET_UNIT,
                    "unit_verified": True}
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"FuelCast inference failed: {exc}") from exc


def load_fuelcast_xgboost(settings: Settings) -> FuelCastXGBoost:
    root = settings.artifacts_dir / RELATIVE_DIR
    manifest = _read_json(root / "model_manifest.json")
    schema = _read_json(root / "feature_schema.json")
    candidate = _read_json(root / "model/candidate_manifest.json")
    model_path = root / "model/model.json"
    model_json = _read_json(model_path)
    if manifest.get("run_id") != RUN_ID or candidate.get("run_id") != RUN_ID:
        raise ModelUnavailableError("FuelCast run ID does not match its directory.")
    for name in ("preprocessor.pkl", "feature_schema.json", "model/model.json",
                 "model/candidate_manifest.json", "metrics.json"):
        path = root / name
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ModelUnavailableError(f"FuelCast artifact {name} is missing: {exc}") from exc
        if digest != manifest.get("artifact_sha256", {}).get(name):
            raise ModelUnavailableError(f"FuelCast artifact hash mismatch: {name}.")
    transformed = tuple(candidate.get("feature_order", ()))
    if (tuple(schema.get("raw_features", ())) != RAW_FEATURES
            or transformed != TRANSFORMED_FEATURES
            or transformed != tuple(schema.get("classical_features", ()))
            or transformed != tuple(model_json.get("learner", {}).get("feature_names", ()))
            or candidate.get("target_unit") != TARGET_UNIT
            or manifest.get("target") != "fuel_consumption_kg_s"):
        raise ModelUnavailableError("FuelCast manifest, schema, and model feature or target metadata disagree.")
    try:
        import joblib
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ModelUnavailableError(f"FuelCast inference dependency is missing: {exc}") from exc
    except Exception as exc:
        raise ModelUnavailableError(f"FuelCast XGBoost runtime cannot be loaded: {exc}") from exc
    try:
        preprocessor = joblib.load(root / "preprocessor.pkl")
    except (ImportError, AttributeError, ModuleNotFoundError) as exc:
        raise ModelUnavailableError(f"FuelCast preprocessor class is unavailable: {exc}") from exc
    except Exception as exc:
        raise ModelUnavailableError(f"FuelCast preprocessor cannot be loaded: {exc}") from exc
    if not hasattr(preprocessor, "transform"):
        raise ModelUnavailableError("FuelCast preprocessor must expose transform.")
    try:
        model = XGBRegressor()
        model.load_model(str(model_path))  # XGBoost's supported JSON model API
        if tuple(model.get_booster().feature_names or ()) != transformed:
            raise ModelUnavailableError("FuelCast model feature order differs from the manifest.")
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"FuelCast XGBoost JSON cannot be loaded: {exc}") from exc
    return FuelCastXGBoost(model, preprocessor, transformed)
