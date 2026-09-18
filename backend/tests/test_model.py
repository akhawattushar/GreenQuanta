"""The trained artifacts must load and predict, and reject bad input."""

import pytest

from app.services.model_registry import (
    FeatureValidationError,
    ModelUnavailableError,
    evaluate_holdout,
    get_bundle,
    status,
)

EXPECTED_RAW_FEATURES = [
    "sailing_speed",
    "displacement",
    "trim",
    "wind_speed",
    "wind_direction_relative",
    "combined_wave_height",
    "combined_wave_period",
    "sea_current_speed",
    "sea_current_direction_relative",
    "sea_water_temperature",
    "vessel_type",
]

VALID_ROW = {
    "sailing_speed": 15.5,
    "displacement": 12.0,
    "trim": 0.0,
    "wind_speed": 14.0,
    "wind_direction_relative": 90.0,
    "combined_wave_height": 3.0,
    "combined_wave_period": 5.5,
    "sea_current_speed": 0.5,
    "sea_current_direction_relative": 90.0,
    "sea_water_temperature": 17.0,
    "vessel_type": "Tanker Ship",
}


def test_artifacts_load():
    bundle = get_bundle()
    assert bundle.model_class
    assert bundle.preprocessor_class == "ColumnTransformer"
    assert bundle.target_column == "fuel_consumption_rate"


def test_feature_order_matches_training():
    assert get_bundle().raw_feature_order == EXPECTED_RAW_FEATURES


def test_transformed_width_matches_model():
    bundle = get_bundle()
    assert bundle.n_transformed_features == 14
    assert bundle.transform([VALID_ROW]).shape == (1, 14)


def test_categories_come_from_the_encoder():
    categories = get_bundle().categorical_categories["vessel_type"]
    assert categories == ["Fishing Trawler", "Oil Service Boat", "Surfer Boat", "Tanker Ship"]


def test_prediction_is_finite_and_positive():
    [value] = get_bundle().predict([VALID_ROW])
    assert isinstance(value, float)
    assert value == value  # not NaN
    assert value > 0


def test_batch_prediction_matches_single():
    bundle = get_bundle()
    batch = bundle.predict([VALID_ROW, {**VALID_ROW, "sailing_speed": 18.0}])
    assert len(batch) == 2
    assert batch[0] == pytest.approx(bundle.predict([VALID_ROW])[0])
    # A faster ship burns more per hour in this model.
    assert batch[1] > batch[0]


def test_unknown_category_is_rejected():
    with pytest.raises(FeatureValidationError, match="not a category seen during training"):
        get_bundle().predict([{**VALID_ROW, "vessel_type": "Container"}])


def test_non_numeric_feature_is_rejected():
    with pytest.raises(FeatureValidationError, match="must be numeric"):
        get_bundle().predict([{**VALID_ROW, "sailing_speed": "fast"}])


def test_missing_feature_is_rejected():
    incomplete = {k: v for k, v in VALID_ROW.items() if k != "trim"}
    with pytest.raises(FeatureValidationError, match="missing required feature"):
        get_bundle().predict([incomplete])


def test_empty_batch_is_rejected():
    with pytest.raises(FeatureValidationError):
        get_bundle().predict([])


def test_holdout_metrics_are_measured():
    metrics = evaluate_holdout()
    assert metrics["available"] is True
    assert metrics["n_samples"] > 0
    for key in ("mae", "rmse", "r2"):
        assert isinstance(metrics[key], float)


def test_missing_artifacts_raise_clear_error(monkeypatch, tmp_path):
    """A missing model must produce an explicit failure, never a fake number."""
    from app.core import config
    from app.services import model_registry

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    model_registry._bundle = None
    model_registry._load_error = None
    try:
        with pytest.raises(ModelUnavailableError, match="not found"):
            model_registry.get_bundle(force_reload=True)
        assert status()["loaded"] is False
    finally:
        monkeypatch.delenv("ARTIFACTS_DIR", raising=False)
        config.get_settings.cache_clear()
        model_registry._bundle = None
        model_registry._load_error = None
        get_bundle(force_reload=True)
