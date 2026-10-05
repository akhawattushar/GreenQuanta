"""Published FuelCast model partition paths."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FuelCastTransformationArtifact:
    output_dir: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    preprocessor_path: Path
    vqr_preprocessor_path: Path
    vqr_target_scaler_path: Path
    split_manifest_path: Path
    feature_schema_path: Path
    dataset_version: str
