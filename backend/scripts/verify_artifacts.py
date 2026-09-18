#!/usr/bin/env python3
"""Standalone artifact check — run this before starting the API.

Reports, from the artifacts themselves:
  * whether the model and preprocessor load,
  * the exact raw feature order and categorical vocabulary,
  * one real prediction on a valid row,
  * measured MAE / RMSE / R2 on the shipped held-out test split.

Exit code 0 means the inference path is usable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.model_registry import (  # noqa: E402
    ModelUnavailableError,
    evaluate_holdout,
    get_bundle,
)


def main() -> int:
    settings = get_settings()
    print("Artifacts directory :", settings.artifacts_dir)
    print("Model               :", settings.resolved_model_path)
    print("Preprocessor        :", settings.resolved_preprocessor_path)
    print("-" * 68)

    try:
        bundle = get_bundle(force_reload=True)
    except ModelUnavailableError as exc:
        print("FAILED to load artifacts:\n ", exc)
        return 1

    print("Model class         :", bundle.model_class)
    print("Preprocessor class  :", bundle.preprocessor_class)
    print("Target column       :", bundle.target_column)
    print("Target unit         :", settings.target_unit, "(verified:", settings.target_unit_verified, ")")
    print("Transformed features:", bundle.n_transformed_features)
    print("Raw feature order   :")
    for i, name in enumerate(bundle.raw_feature_order, 1):
        print(f"   {i:2d}. {name}")
    print("Categorical vocab   :", json.dumps(bundle.categorical_categories))
    if bundle.warnings:
        print("Warnings:")
        for warning in bundle.warnings:
            print("  -", warning)

    row = {}
    for name in bundle.raw_feature_order:
        if name in bundle.categorical_categories:
            row[name] = bundle.categorical_categories[name][-1]
        else:
            row[name] = 1.0
    row.update(
        {
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
        }
    )
    print("-" * 68)
    print("Sample input        :", json.dumps(row))
    print("Prediction          :", round(bundle.predict([row])[0], 4), settings.target_unit)

    print("-" * 68)
    metrics = evaluate_holdout()
    if metrics.get("available"):
        print(f"Held-out samples    : {metrics['n_samples']}")
        print(f"MAE                 : {metrics['mae']:.6g}")
        print(f"RMSE                : {metrics['rmse']:.6g}")
        print(f"R^2                 : {metrics['r2']:.6f}")
        if metrics["r2"] > 0.999:
            print(
                "\nNOTE: R^2 at (or indistinguishable from) 1.0 means the target is an almost exact\n"
                "      linear function of the inputs in this dataset. Treat it as a pipeline check,\n"
                "      not as evidence of real-world predictive skill."
            )
    else:
        print("Metrics unavailable :", metrics.get("reason"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
