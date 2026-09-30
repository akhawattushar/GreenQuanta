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
verified run manifest. The visible VQR quartet remains in `vqr/` pending
version identification; the second VQR export is missing. Neither is wired
into the backend.

## Contract gate for backend integration

- For **each VQR version**, obtain a unique run ID and a manifest with hashes
  for its four files, circuit construction and parameter-order source,
  estimator and simulator package versions, importable input preprocessor and
  target scaler definitions, ordered features with units, and labeled
  validation and test reports. Validate each version independently.
- For XGBoost, confirm the export owner's wind-direction convention and
  manifest convention if the bundle and candidate roles differ. The local
  synthetic-row inference and transformed order checks have passed.
- Before mapping any FuelCast candidate into the API, resolve the API's
  relative wind direction (degrees from vessel heading) against FuelCast's
  wind direction, verify sailing speed against speed over ground, and record
  all unit conversions. Reconcile the XGBoost test-metric claim with the
  project's split history before exposing metrics.

Keep the current legacy loader paths available until a candidate passes this
gate. The source export command is not available in this repository; record it
with the export owner before treating these files as reproducible builds.
