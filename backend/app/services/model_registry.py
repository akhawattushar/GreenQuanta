"""Loads the real training artifacts and runs inference through them.

Nothing here invents a prediction. If any artifact is missing or the loaded
preprocessor/model pair is inconsistent, `ModelUnavailableError` is raised.
API boundaries return a stable public error and log the internal reason.
"""

from __future__ import annotations

import json
import logging
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)
MODEL_UNAVAILABLE_MESSAGE = "Selected prediction model is unavailable."
PREDICTION_FAILED_MESSAGE = "Prediction could not be completed."
MODEL_HEALTH_FAILED_MESSAGE = "Model health check failed."


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model or preprocessor cannot be used."""


class FeatureValidationError(ValueError):
    """Raised when inference input does not match the training schema."""


class UnknownModelError(ValueError):
    """Raised for a model ID that the registry does not recognize."""


@dataclass
class ModelBundle:
    """Everything needed to turn raw feature dicts into predictions."""

    model: Any
    preprocessor: Any
    raw_feature_order: list  # columns the preprocessor expects, in order
    numerical_features: list
    categorical_features: list
    transformed_feature_names: list
    categorical_categories: dict = field(default_factory=dict)
    target_column: str = "fuel_consumption_rate"
    model_class: str = ""
    preprocessor_class: str = ""
    n_transformed_features: int = 0
    artifact_paths: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def metadata(self, settings: Settings | None = None) -> dict:
        settings = settings or get_settings()
        return {
            "model_class": self.model_class,
            "preprocessor_class": self.preprocessor_class,
            "target_column": self.target_column,
            "target_unit": settings.target_unit,
            "target_unit_verified": settings.target_unit_verified,
            "raw_feature_order": list(self.raw_feature_order),
            "numerical_features": list(self.numerical_features),
            "categorical_features": list(self.categorical_features),
            "categorical_categories": {k: list(v) for k, v in self.categorical_categories.items()},
            "transformed_feature_names": list(self.transformed_feature_names),
            "transformed_feature_count": self.n_transformed_features,
            "artifact_paths": {name: Path(path).name for name, path in self.artifact_paths.items()},
            "warnings": list(self.warnings),
        }


_bundle: ModelBundle | None = None
_load_error: str | None = None
_lock = threading.Lock()
_fuelcast_bundle: Any | None = None
_fuelcast_error: str | None = None
_fuelcast_lock = threading.Lock()


def _require(path: Path, label: str) -> None:
    if not path.is_file():
        raise ModelUnavailableError(
            f"{label} not found at {path}. Set the matching environment variable "
            "(ARTIFACTS_DIR / MODEL_PATH / PREPROCESSOR_PATH) or restore the artifact."
        )


def _load_bundle(settings: Settings) -> ModelBundle:
    try:
        import joblib  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ModelUnavailableError("joblib is not installed; cannot load the trained model.") from exc

    model_path = settings.resolved_model_path
    pre_path = settings.resolved_preprocessor_path
    _require(model_path, "Trained model")
    _require(pre_path, "Preprocessor")

    load_warnings: list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(pre_path)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
            raise ModelUnavailableError(f"Failed to unpickle artifacts: {exc}") from exc
        seen: set = set()
        for w in caught:
            text = str(w.message).split("\n")[0]
            if "version" not in text.lower():
                continue
            # Collapse the one-per-estimator sklearn version warnings into one line.
            key = text[:60]
            if key in seen:
                continue
            seen.add(key)
            load_warnings.append(text)

    if not hasattr(model, "predict"):
        raise ModelUnavailableError(
            f"Loaded object of type {type(model).__name__} has no .predict(); it is not a usable estimator."
        )
    if not hasattr(preprocessor, "transform"):
        raise ModelUnavailableError(
            f"Loaded preprocessor of type {type(preprocessor).__name__} has no .transform()."
        )

    # ---- schema, read from the artifacts rather than assumed ----------
    report: dict = {}
    if settings.resolved_transformation_report_path.is_file():
        report = json.loads(settings.resolved_transformation_report_path.read_text(encoding="utf-8"))

    numerical = list(report.get("numerical_features", []))
    categorical = list(report.get("categorical_features", []))

    raw_order: list = []
    categories: dict = {}
    transformers = getattr(preprocessor, "transformers_", None)
    if transformers:
        for name, transformer, columns in transformers:
            if isinstance(columns, (list, tuple)):
                raw_order.extend(columns)
            if name == "categorical" or (categorical and set(columns) & set(categorical)):
                encoder = None
                if hasattr(transformer, "named_steps"):
                    for step in transformer.named_steps.values():
                        if hasattr(step, "categories_"):
                            encoder = step
                elif hasattr(transformer, "categories_"):
                    encoder = transformer
                if encoder is not None:
                    for col, cats in zip(columns, encoder.categories_):
                        categories[col] = [str(c) for c in cats]
    if not raw_order:
        raw_order = numerical + categorical
    if not numerical:
        numerical = [c for c in raw_order if c not in categories]
    if not categorical:
        categorical = [c for c in raw_order if c in categories]

    transformed_names: list = []
    fn_path = settings.resolved_feature_names_path
    if fn_path.is_file():
        transformed_names = [str(n) for n in json.loads(fn_path.read_text(encoding="utf-8"))]
    if not transformed_names and hasattr(preprocessor, "get_feature_names_out"):
        try:
            transformed_names = [str(n) for n in preprocessor.get_feature_names_out()]
        except Exception:  # noqa: BLE001 - names are informational only
            transformed_names = []

    n_expected = getattr(model, "n_features_in_", None)
    expected_from_report = report.get("transformed_feature_count")
    if n_expected is not None and expected_from_report and int(n_expected) != int(expected_from_report):
        raise ModelUnavailableError(
            f"Model expects {n_expected} transformed features but the transformation report "
            f"records {expected_from_report}. The model and preprocessor do not match."
        )
    if n_expected is not None and transformed_names and len(transformed_names) != int(n_expected):
        load_warnings.append(
            f"feature_names.json lists {len(transformed_names)} names but the model expects {n_expected}."
        )

    bundle = ModelBundle(
        model=model,
        preprocessor=preprocessor,
        raw_feature_order=raw_order,
        numerical_features=numerical,
        categorical_features=categorical,
        transformed_feature_names=transformed_names,
        categorical_categories=categories,
        target_column=report.get("target_column", "fuel_consumption_rate"),
        model_class=type(model).__name__,
        preprocessor_class=type(preprocessor).__name__,
        n_transformed_features=int(n_expected) if n_expected is not None else len(transformed_names),
        artifact_paths={"model": str(model_path), "preprocessor": str(pre_path)},
        warnings=load_warnings,
    )

    # ---- smoke test: one round-trip through preprocessor + model ------
    probe = {}
    for col in bundle.raw_feature_order:
        if col in categories and categories[col]:
            probe[col] = categories[col][0]
        else:
            probe[col] = 1.0
    try:
        matrix = bundle.transform([probe])
        bundle.model.predict(matrix)
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailableError(f"Artifact smoke test failed: {exc}") from exc

    logger.info(
        "Loaded %s + %s (%d transformed features)",
        bundle.model_class,
        bundle.preprocessor_class,
        bundle.n_transformed_features,
    )
    return bundle


def get_bundle(*, force_reload: bool = False) -> ModelBundle:
    """Return the cached bundle, loading it on first use."""
    global _bundle, _load_error
    with _lock:
        if force_reload:
            _bundle, _load_error = None, None
        if _bundle is not None:
            return _bundle
        if _load_error is not None:
            raise ModelUnavailableError(_load_error)
        try:
            _bundle = _load_bundle(get_settings())
        except ModelUnavailableError as exc:
            _load_error = str(exc)
            raise
        return _bundle


def get_predictor(model_id: str = "legacy", *, force_reload: bool = False):
    """Select an independent predictor without changing the legacy default."""
    if model_id == "legacy":
        return get_bundle(force_reload=force_reload)
    if model_id != "fuelcast_xgboost":
        raise UnknownModelError(f"Unknown model ID {model_id!r}. Available IDs: legacy, fuelcast_xgboost.")
    from app.services.fuelcast_xgboost import load_fuelcast_xgboost

    global _fuelcast_bundle, _fuelcast_error
    with _fuelcast_lock:
        if force_reload:
            _fuelcast_bundle, _fuelcast_error = None, None
        if _fuelcast_bundle is not None:
            return _fuelcast_bundle
        if _fuelcast_error is not None:
            raise ModelUnavailableError(_fuelcast_error)
        try:
            _fuelcast_bundle = load_fuelcast_xgboost(get_settings())
        except ModelUnavailableError as exc:
            _fuelcast_error = str(exc)
            raise
        return _fuelcast_bundle


def fuelcast_status() -> dict:
    """Report model readiness for explicitly supplied FuelCast raw inputs."""
    required_api_inputs = [
        "speed_over_ground (m/s)", "wind_direction (FuelCast degrees)",
        "wind_speed (m/s)", "wave_height (m)", "wave_period (s)", "current_speed (m/s)",
    ]
    try:
        predictor = get_predictor("fuelcast_xgboost")
    except ModelUnavailableError:
        logger.exception("FuelCast model status check failed")
        return {"model_id": "fuelcast_xgboost", "run_id": "fuelcast-phase1-20260928-002",
                "loaded": False, "error": MODEL_HEALTH_FAILED_MESSAGE, "api_available": False,
                "input_contract": "explicit_fuelcast_inputs", "required_api_inputs": required_api_inputs,
                "wind_direction_convention_verified": False, "target_unit": "kg/s"}
    return {"loaded": True, "error": None, "api_available": True,
            "input_contract": "explicit_fuelcast_inputs", "required_api_inputs": required_api_inputs,
            "wind_direction_convention_verified": False, **predictor.metadata()}


def status() -> dict:
    """Non-raising health summary for /health and /admin."""
    try:
        bundle = get_bundle()
    except ModelUnavailableError:
        logger.exception("Legacy model status check failed")
        return {"loaded": False, "error": MODEL_HEALTH_FAILED_MESSAGE}
    return {"loaded": True, "error": None, **bundle.metadata()}


# --------------------------------------------------------------------------
# inference helpers (attached to ModelBundle)
# --------------------------------------------------------------------------
def _validate_rows(bundle: ModelBundle, rows: Sequence[dict]) -> None:
    if not rows:
        raise FeatureValidationError("At least one feature row is required.")
    for index, row in enumerate(rows):
        missing = [c for c in bundle.raw_feature_order if c not in row]
        if missing:
            raise FeatureValidationError(
                f"Row {index}: missing required feature(s): {', '.join(missing)}."
            )
        for col in bundle.numerical_features:
            value = row[col]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FeatureValidationError(f"Row {index}: {col!r} must be numeric, got {value!r}.")
            if value != value or value in (float("inf"), float("-inf")):
                raise FeatureValidationError(f"Row {index}: {col!r} must be a finite number.")
        for col, allowed in bundle.categorical_categories.items():
            value = str(row.get(col, ""))
            if allowed and value not in allowed:
                raise FeatureValidationError(
                    f"Row {index}: {col!r}={value!r} is not a category seen during training. "
                    f"Allowed: {', '.join(allowed)}."
                )


def _transform(self: ModelBundle, rows: Sequence[dict]):
    import pandas as pd  # noqa: PLC0415

    _validate_rows(self, rows)
    frame = pd.DataFrame([{c: row[c] for c in self.raw_feature_order} for row in rows])
    frame = frame[self.raw_feature_order]
    try:
        matrix = self.preprocessor.transform(frame)
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailableError(f"Preprocessing failed: {exc}") from exc
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    if self.n_transformed_features and matrix.shape[1] != self.n_transformed_features:
        raise ModelUnavailableError(
            f"Preprocessor produced {matrix.shape[1]} features, model expects {self.n_transformed_features}."
        )
    return matrix


def _predict(self: ModelBundle, rows: Sequence[dict]) -> list:
    matrix = self.transform(rows)
    try:
        values = self.model.predict(matrix)
    except Exception as exc:  # noqa: BLE001
        raise ModelUnavailableError(f"Model inference failed: {exc}") from exc
    return [float(v) for v in list(values)]


ModelBundle.transform = _transform  # type: ignore[attr-defined]
ModelBundle.predict = _predict  # type: ignore[attr-defined]


#: Above this pairwise |r|, two numerical feature columns are treated as
#: duplicates for the multicollinearity diagnostic below.
_COLLINEARITY_THRESHOLD = 0.999


def _collinearity_diagnostic(features, target, feature_names: list[str]) -> dict:
    """Flag near-duplicate feature columns and a near-linear single-feature fit.

    A plain linear model reaching R^2 ~ 1.0 on physically independent signals
    (wind, waves, currents, temperature, speed, displacement...) is not
    plausible. This check makes that visible instead of only reporting the
    headline metric.
    """
    import numpy as np  # noqa: PLC0415

    numeric = features[:, : min(len(feature_names), features.shape[1])]
    n_cols = numeric.shape[1]
    dup_pairs: list[dict] = []
    if n_cols >= 2:
        std = numeric.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(numeric.T)
        for i in range(n_cols):
            if std[i] == 0:
                continue
            for j in range(i + 1, n_cols):
                if std[j] == 0:
                    continue
                r = corr[i, j]
                if np.isfinite(r) and abs(r) >= _COLLINEARITY_THRESHOLD:
                    name_i = feature_names[i] if i < len(feature_names) else f"col_{i}"
                    name_j = feature_names[j] if j < len(feature_names) else f"col_{j}"
                    dup_pairs.append({"a": name_i, "b": name_j, "corr": round(float(r), 6)})

    # Best single-feature linear fit against the target, to show how much of
    # R^2 = 1.0 is explained by one collapsed variable rather than a learned
    # combination of genuinely independent inputs.
    best = None
    if n_cols >= 1 and len(target) > 1:
        for i in range(n_cols):
            col = numeric[:, i]
            if col.std() == 0:
                continue
            r = float(np.corrcoef(col, target)[0, 1])
            if best is None or abs(r) > abs(best["corr"]):
                name_i = feature_names[i] if i < len(feature_names) else f"col_{i}"
                best = {"feature": name_i, "corr": round(r, 6)}

    return {
        "duplicate_or_near_duplicate_feature_pairs": len(dup_pairs),
        "example_pairs": dup_pairs[:8],
        "single_feature_best_fit": best,
        "flag": bool(dup_pairs) or (best is not None and abs(best["corr"]) > 0.999),
    }


def evaluate_holdout() -> dict:
    """Measured metrics on the shipped held-out test split, or a reason why not."""
    settings = get_settings()
    path = settings.resolved_test_set_path
    if not path.is_file():
        logger.warning("Model evaluation test split not found: %s", path)
        return {"available": False, "reason": "Model evaluation data is unavailable."}
    try:
        import numpy as np  # noqa: PLC0415
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        logger.exception("Model evaluation dependency unavailable")
        return {"available": False, "reason": "Model evaluation is unavailable."}

    bundle = get_bundle()
    try:
        with np.load(path, allow_pickle=False) as data:
            if "features" not in data or "target" not in data:
                logger.warning("Model evaluation test split has unexpected keys: %s", list(data.files))
                return {"available": False, "reason": "Model evaluation data is invalid."}
            features = data["features"]
            target = data["target"]
        predicted = bundle.model.predict(features)
    except Exception:
        logger.exception("Model evaluation failed")
        return {"available": False, "reason": "Model evaluation could not be completed."}

    result = {
        "available": True,
        "split": "held-out test set shipped in artifacts/",
        "n_samples": int(features.shape[0]),
        "mae": float(mean_absolute_error(target, predicted)),
        "rmse": float(mean_squared_error(target, predicted) ** 0.5),
        "r2": float(r2_score(target, predicted)),
        "note": (
            "Metrics are recomputed at request time from the shipped test split. "
            "They are not copied from any report."
        ),
    }

    try:
        diag = _collinearity_diagnostic(features, target, list(bundle.numerical_features))
    except Exception:  # noqa: BLE001 - diagnostic must never break the metrics endpoint
        diag = None
    if diag and diag["flag"]:
        result["data_quality_warning"] = (
            f"{diag['duplicate_or_near_duplicate_feature_pairs']} pairs of numerical features in the "
            "shipped dataset are duplicates or near-duplicates of each other (|r| >= "
            f"{_COLLINEARITY_THRESHOLD}), and the target is almost perfectly linear in a single "
            "collapsed feature"
            + (f" ({diag['single_feature_best_fit']['feature']}, r={diag['single_feature_best_fit']['corr']})" if diag["single_feature_best_fit"] else "")
            + ". This is why a plain linear model reaches R2 ~ 1.0: the dataset does not carry "
            "independent signal across the 10 numerical fields it claims to. Treat this R2/MAE/RMSE "
            "as a measurement of fit to the shipped (likely synthetic) data, not as evidence of "
            "real-world predictive accuracy. Vessel type shows near-zero correlation with the target "
            "in this data. Retraining on a dataset with genuinely independent features is needed "
            "before these metrics can be used to claim real-world performance."
        )
        result["data_quality_diagnostic"] = diag
    return result
