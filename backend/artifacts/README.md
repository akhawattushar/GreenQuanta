# Local model artifacts

The current backend still loads the legacy sklearn bundle from these paths,
relative to `backend/artifacts/`:

| Setting | Path |
| --- | --- |
| `MODEL_PATH` | `model_trainer/best_model.pkl` |
| `PREPROCESSOR_PATH` | `data_transformation/preprocessor.pkl` |
| `FEATURE_NAMES_PATH` | `data_transformation/feature_names.json` |
| `TRANSFORMATION_REPORT_PATH` | `data_transformation/transformation_report.json` |
| `TEST_SET_PATH` | `data_transformation/transformed_data/transformed_test.npz` |

The eight files in `01_ingestion/`, `02_validation/`,
`data_transformation/`, and `model_trainer/` were restored as copies from
`legacy model artifacts /`. Each copy matches the blob tracked at its original
path in Git. The relocated source remains in place.

The FuelCast XGBoost inference export is at
`fuelcast/classical/xgboost/fuelcast-phase1-20260928-002/`. Its
`model_manifest.json` is the bundle manifest; paths in its `artifact_sha256`
map are relative to that directory. `model/candidate_manifest.json` is the
candidate manifest; its `model_file` is relative to `model/`. The five present
inference and metadata files (`preprocessor.pkl`, `feature_schema.json`,
`model/model.json`, `model/candidate_manifest.json`, `metrics.json`) match the
bundle manifest's SHA-256 entries. The manifest also references absent
`model/validation_predictions.npz`, `model/search_results.json`, and
`leaderboard.json`; its evaluation manifest is absent too. `metrics.json` is
retained for provenance, but its test figures must not be published until the
split provenance is reconciled. The candidate manifest, feature schema, and
model JSON name the same seven transformed features in the same order. A local
smoke test with one synthetic row loaded the pickled preprocessor and model,
produced a seven-column transform in that order, and returned one finite
`kg/s` prediction. This checks runtime compatibility, not model accuracy.

| Required raw FuelCast input | Unit |
| --- | --- |
| `speed_over_ground` | m/s |
| `wind_speed` | m/s |
| `wind_direction` | degrees |
| `wave_height` | m |
| `wave_period` | s |
| `current_speed` | m/s |

The manifest does not define the wind-direction reference convention. Confirm
it with the export owner before mapping GreenQuanta's relative wind direction.

The preexisting `classical/` pair (`model.json`, `preprocessor.pkl`) has no
verified run manifest and is not wired into the backend.

SIH contains both `quick` and `normal` VQR candidates for run
`fuelcast-phase1-20260928-002`. The four files in the preexisting, untracked
`vqr/` directory match the **normal** candidate by SHA-256. They remain in
place. The normal application bundle is packaged separately at
`fuelcast/vqr/fuelcast-phase1-20260928-002/normal/5ef14aa8c84e4834ba9992eee821ab5a/`.
Its `inference_manifest.json` hashes seven files and records the six ordered
inputs, units, simulator circuit, dependencies, and validation provenance.
The included `candidate_manifest.json` also hashes training-only files that
remain in SIH; its full artifact set is not present in this inference bundle.
The quick candidate used 60 sampled training rows as a smoke test and was not
copied into the application bundle. The normal candidate is registered as
`fuelcast_vqr` for direct prediction using the exact simulator. It is not
connected to voyage, optimization, or scenario calculations.

## Contract gate for backend integration

- The direct-prediction adapter uses Qiskit and Qiskit Machine Learning on
  Python 3.13. Voyage, optimization, and scenario integration still require
  verified live-input semantics, particularly the wind-direction reference
  convention. The normal bundle is a candidate, not a production-ready model.
  Its reported metrics are from validation, not a held-out test.
- For XGBoost, confirm the export owner's wind-direction convention and
  manifest convention if the bundle and candidate roles differ. The local
  synthetic-row inference and transformed order checks have passed.
- Before automatically mapping GreenQuanta voyage/weather fields to any
  FuelCast candidate, resolve relative wind direction (degrees from vessel
  heading) against FuelCast's wind direction, verify sailing speed against
  speed over ground, and record all unit conversions. Direct prediction uses
  only the caller's explicit FuelCast fields. Reconcile the XGBoost test-metric
  claim with the project's split history before exposing metrics.

Keep the current legacy loader paths available until a candidate passes this
gate. The source export command is not available in this repository; record it
with the export owner before treating these files as reproducible builds.
