"""Shared chronological partitions and training-fitted FuelCast transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from greenfleet.artifacts.fuelcast_transformation_artifact import FuelCastTransformationArtifact
from greenfleet.constants.fuelcast import (
    CANONICAL_COLUMNS, CANONICAL_FEATURES, CANONICAL_TARGET, CANONICAL_UNITS,
    DATASET_ID, EXPECTED_CONFIGS,
)

CLASSICAL_FEATURES = (
    "speed_over_ground", "wind_speed", "wind_direction_sin",
    "wind_direction_cos", "wave_height", "wave_period", "current_speed",
)
PARTITIONS = ("train", "validation", "test")
ARRAY_KEYS = (
    "features", "vqr_features", "target", "vqr_target",
    "record_id", "vessel_id", "time_index",
)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CircularWindEncoder(BaseEstimator, TransformerMixin):
    """Replace the third of six ordered FuelCast inputs by sine and cosine."""

    def fit(self, values, y=None):
        self.n_features_in_ = 6
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("FuelCast circular encoder requires six features")
        radians = np.deg2rad(values[:, 2])
        return np.column_stack((values[:, :2], np.sin(radians),
                                np.cos(radians), values[:, 3:]))


class FuelCastImputer(BaseEstimator, TransformerMixin):
    """Median for continuous inputs; circular mean for wind direction."""

    _other_indices = (0, 1, 3, 4, 5)

    def fit(self, values, y=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6 or np.isinf(values).any():
            raise ValueError("FuelCast imputer requires six finite-or-missing features")
        if np.isnan(values).all(axis=0).any():
            raise ValueError("FuelCast training partition has an entirely missing feature")
        self.medians_ = np.nanmedian(values[:, self._other_indices], axis=0)
        radians = np.deg2rad(values[:, 2])
        sine, cosine = np.nanmean(np.sin(radians)), np.nanmean(np.cos(radians))
        self.wind_direction_fill_ = (math.degrees(math.atan2(sine, cosine)) % 360
                                     if math.hypot(sine, cosine) > 1e-12 else 0.0)
        self.n_features_in_ = 6
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("FuelCast imputer requires six features")
        result = values.copy()
        for index, median in zip(self._other_indices, self.medians_):
            result[np.isnan(result[:, index]), index] = median
        result[np.isnan(result[:, 2]), 2] = self.wind_direction_fill_
        return result


class VQRAngleEncoder(BaseEstimator, TransformerMixin):
    """Training-fitted five-feature angle scaling plus periodic wind direction."""

    _other_indices = (0, 1, 3, 4, 5)

    def fit(self, values, y=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("FuelCast VQR angle encoder requires six features")
        self.scaler_ = MinMaxScaler(feature_range=(0, math.pi), clip=True)
        self.scaler_.fit(values[:, self._other_indices])
        self.n_features_in_ = 6
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 6:
            raise ValueError("FuelCast VQR angle encoder requires six features")
        result = np.empty_like(values)
        result[:, self._other_indices] = self.scaler_.transform(
            values[:, self._other_indices]
        )
        result[:, 2] = np.deg2rad(values[:, 2])
        return result


def _input(run_dir: Path) -> tuple[pd.DataFrame, dict, str]:
    canonical_path = run_dir / "02_etl" / "fuelcast_clean.csv"
    audit_path = run_dir / "02_etl" / "etl_audit.json"
    if not canonical_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("FuelCast canonical CSV and ETL audit are required")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    canonical_hash = _hash(canonical_path)
    version = audit.get("dataset_version")
    if (audit.get("dataset_id") != DATASET_ID or
            audit.get("source_run_id") != run_dir.name or
            audit.get("canonical_sha256") != canonical_hash or
            audit.get("canonical_columns") != list(CANONICAL_COLUMNS) or
            audit.get("units") != CANONICAL_UNITS or
            not isinstance(version, str) or
            re.fullmatch(r"[0-9a-f]{40}", version) is None):
        raise ValueError("FuelCast canonical audit mismatch")
    frame = pd.read_csv(canonical_path, dtype={
        "record_id": "string", "dataset_version": "string", "vessel_id": "string",
        "time_index": "string",
    })
    if list(frame.columns) != list(CANONICAL_COLUMNS):
        raise ValueError("FuelCast canonical column order mismatch")
    if frame.empty or frame[["record_id", "dataset_version", "vessel_id", "time_index"]].isna().any().any():
        raise ValueError("FuelCast canonical metadata is missing")
    if frame["record_id"].duplicated().any() or frame.duplicated(["vessel_id", "time_index"]).any():
        raise ValueError("FuelCast canonical IDs or vessel/time keys are duplicated")
    if (frame["dataset_version"] != version).any() or set(frame["vessel_id"]) != set(EXPECTED_CONFIGS):
        raise ValueError("FuelCast canonical version or vessels mismatch")
    if not frame["time_index"].str.fullmatch(r"(?:0|[1-9][0-9]*)").all():
        raise ValueError("FuelCast time indexes must be finite nonnegative integers")
    try:
        frame["time_index"] = frame["time_index"].astype(np.int64)
    except (ValueError, OverflowError) as exc:
        raise ValueError("FuelCast time indexes exceed int64 range") from exc
    target = pd.to_numeric(frame[CANONICAL_TARGET], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("FuelCast targets must be finite and nonnegative")
    frame[CANONICAL_TARGET] = target
    for name in CANONICAL_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if np.isinf(frame[name].to_numpy(dtype=np.float64)).any():
            raise ValueError(f"FuelCast {name} contains infinity")
    if ((frame["wind_direction"].dropna() < 0) | (frame["wind_direction"].dropna() > 360)).any():
        raise ValueError("FuelCast wind direction is out of bounds")
    counts = frame.groupby("vessel_id").size().to_dict()
    expected = audit.get("per_vessel")
    if (not isinstance(expected, dict) or set(expected) != set(EXPECTED_CONFIGS) or
            any(counts[name] != expected[name].get("output_rows") for name in EXPECTED_CONFIGS) or
            audit.get("totals", {}).get("output_rows") != len(frame)):
        raise ValueError("FuelCast canonical audit row counts mismatch")
    return frame, audit, canonical_hash


def _boundaries(frame: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    parts = {name: [] for name in PARTITIONS}
    per_vessel = {}
    for vessel in EXPECTED_CONFIGS:
        ordered = frame.loc[frame["vessel_id"] == vessel].sort_values("time_index", kind="stable")
        n = len(ordered)
        train_end, validation_end = 7 * n // 10, 85 * n // 100
        slices = (ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:])
        if any(part.empty for part in slices):
            raise ValueError(f"FuelCast {vessel} cannot populate all three partitions")
        if not (slices[0]["time_index"].max() < slices[1]["time_index"].min() and
                slices[1]["time_index"].max() < slices[2]["time_index"].min()):
            raise ValueError(f"FuelCast {vessel} chronological boundaries overlap")
        per_vessel[vessel] = {}
        for name, part in zip(PARTITIONS, slices):
            parts[name].append(part)
            per_vessel[vessel][name] = {
                "count": len(part), "first_time_index": int(part["time_index"].iloc[0]),
                "last_time_index": int(part["time_index"].iloc[-1]),
                "record_ids": part["record_id"].astype(str).tolist(),
            }
    combined = {name: pd.concat(parts[name], ignore_index=True) for name in PARTITIONS}
    ids = [set(part["record_id"]) for part in combined.values()]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("FuelCast split IDs overlap")
    return combined, per_vessel


def _artifact(output_dir: Path, version: str) -> FuelCastTransformationArtifact:
    return FuelCastTransformationArtifact(
        output_dir, output_dir / "train.npz", output_dir / "validation.npz",
        output_dir / "test.npz", output_dir / "preprocessor.pkl",
        output_dir / "vqr_preprocessor.pkl", output_dir / "vqr_target_scaler.pkl",
        output_dir / "split_manifest.json", output_dir / "feature_schema.json", version,
    )


def run_fuelcast_transformation(run_dir: Path) -> FuelCastTransformationArtifact:
    """Create the one immutable model data contract for a canonical run."""
    run_dir = Path(run_dir).resolve()
    output_dir = run_dir / "04_transformation"
    if os.path.lexists(output_dir):
        raise FileExistsError(f"FuelCast transformation stage already exists: {output_dir}")
    frame, audit, canonical_hash = _input(run_dir)
    partitions, per_vessel = _boundaries(frame)
    train = partitions["train"]
    if train[list(CANONICAL_FEATURES)].isna().all().any():
        raise ValueError("FuelCast training partition has an entirely missing feature")
    classical = Pipeline([
        ("imputer", FuelCastImputer()),
        ("direction", CircularWindEncoder()),
        ("scaler", StandardScaler()),
    ])
    vqr = Pipeline([
        ("imputer", FuelCastImputer()),
        ("angle", VQRAngleEncoder()),
    ])
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    train_features = train[list(CANONICAL_FEATURES)].to_numpy(dtype=np.float64)
    classical.fit(train_features)
    vqr.fit(train_features)
    target_scaler.fit(train[[CANONICAL_TARGET]].to_numpy(dtype=np.float64))
    stage = Path(tempfile.mkdtemp(prefix=".04_transformation-data-", dir=run_dir))
    try:
        artifact = _artifact(stage, audit["dataset_version"])
        joblib.dump(classical, artifact.preprocessor_path)
        joblib.dump(vqr, artifact.vqr_preprocessor_path)
        joblib.dump(target_scaler, artifact.vqr_target_scaler_path)
        for name in PARTITIONS:
            part = partitions[name]
            raw = part[list(CANONICAL_FEATURES)].to_numpy(dtype=np.float64)
            target = part[CANONICAL_TARGET].to_numpy(dtype=np.float64)
            np.savez_compressed(stage / f"{name}.npz",
                features=classical.transform(raw), vqr_features=vqr.transform(raw),
                target=target, vqr_target=target_scaler.transform(target.reshape(-1, 1)).ravel(),
                record_id=part["record_id"].astype(str).to_numpy(dtype=str),
                vessel_id=part["vessel_id"].astype(str).to_numpy(dtype=str),
                time_index=part["time_index"].to_numpy(dtype=np.int64),
            )
        schema = {
            "dataset_id": DATASET_ID, "dataset_version": audit["dataset_version"],
            "raw_features": list(CANONICAL_FEATURES), "classical_features": list(CLASSICAL_FEATURES),
            "vqr_features": list(CANONICAL_FEATURES), "target": CANONICAL_TARGET,
            "units": CANONICAL_UNITS, "array_keys": list(ARRAY_KEYS),
            "classical_preprocessing": "training median of five continuous features; circular-mean wind imputation; wind sine/cosine; standard scaling",
            "vqr_preprocessing": "training median of five continuous features; circular-mean wind imputation; non-direction MinMax [0, pi] clipped; direction degrees to radians",
            "vqr_target_scaling": "training MinMax [-1, 1]; inverse_transform returns kg/s",
        }
        artifact.feature_schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "dataset_id": DATASET_ID, "dataset_version": audit["dataset_version"],
            "source_run_id": run_dir.name, "canonical_sha256": canonical_hash,
            "etl_audit_sha256": _hash(run_dir / "02_etl" / "etl_audit.json"),
            "split_strategy": "per_vessel_chronological_70_15_15",
            "boundary_rule": "train_end=floor(0.70*n); validation_end=floor(0.85*n)",
            "per_vessel": per_vessel,
            "partition_counts": {name: len(partitions[name]) for name in PARTITIONS},
            "artifact_sha256": {path.name: _hash(path) for path in stage.iterdir() if path.is_file()},
        }
        artifact.split_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.symlink(stage.name, output_dir, target_is_directory=True)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return _artifact(output_dir, audit["dataset_version"])


def load_fuelcast_partitions(run_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Load saved partitions and reject altered arrays or identity order."""
    stage = Path(run_dir) / "04_transformation"
    manifest = json.loads((stage / "split_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((stage / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("dataset_version") != manifest.get("dataset_version") or schema.get("array_keys") != list(ARRAY_KEYS):
        raise ValueError("FuelCast transformation schema and manifest mismatch")
    for filename, digest in manifest["artifact_sha256"].items():
        if _hash(stage / filename) != digest:
            raise ValueError(f"FuelCast transformation artifact changed: {filename}")
    arrays = {}
    for name in PARTITIONS:
        with np.load(stage / f"{name}.npz", allow_pickle=False) as content:
            if set(content.files) != set(ARRAY_KEYS):
                raise ValueError(f"FuelCast {name} array schema mismatch")
            arrays[name] = {key: content[key] for key in ARRAY_KEYS}
        expected_ids = [record_id for vessel in EXPECTED_CONFIGS
                        for record_id in manifest["per_vessel"][vessel][name]["record_ids"]]
        if arrays[name]["record_id"].tolist() != expected_ids:
            raise ValueError(f"FuelCast {name} row identities mismatch")
        n = len(expected_ids)
        if (arrays[name]["features"].shape != (n, 7) or
                arrays[name]["vqr_features"].shape != (n, 6) or
                any(arrays[name][key].shape != (n,) for key in ARRAY_KEYS[2:])):
            raise ValueError(f"FuelCast {name} array lengths mismatch")
        cursor = 0
        for vessel in EXPECTED_CONFIGS:
            detail = manifest["per_vessel"][vessel][name]
            count = detail["count"]
            vessel_ids = arrays[name]["vessel_id"][cursor:cursor + count]
            times = arrays[name]["time_index"][cursor:cursor + count]
            if (len(vessel_ids) != count or not np.all(vessel_ids == vessel) or
                    int(times[0]) != detail["first_time_index"] or
                    int(times[-1]) != detail["last_time_index"] or
                    np.any(np.diff(times) <= 0)):
                raise ValueError(f"FuelCast {name} vessel/time metadata mismatch")
            cursor += count
    return arrays, schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Create shared FuelCast model partitions")
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing canonical FuelCast run directory")
    args = parser.parse_args()
    artifact = run_fuelcast_transformation(args.run_dir)
    print(f"FuelCast transformation: {artifact.output_dir}")


if __name__ == "__main__":
    # Import through the canonical module name so persisted custom transformers
    # remain loadable when the stage was launched with ``python -m``.
    from greenfleet.ml_pipeline.transformation.fuelcast import main as package_main

    package_main()
