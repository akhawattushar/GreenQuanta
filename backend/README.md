# QuantaFleet Backend (FastAPI)

Backend for **GreenQuanta / QuantaFleet**.

Fuel predictions come from the trained artifacts in `artifacts/`. Optimisation,
scenario analysis and voyage fuel figures use the explicitly selected predictor,
with the legacy predictor as the default. If artifacts cannot be loaded, the
affected endpoints return **503 with a path-free public message** — no
placeholder number is substituted for model output.

---

## Quick start

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit SECRET_KEY

python scripts/verify_artifacts.py     # confirms the model loads and predicts
uvicorn app.main:app --reload --port 8000
```

Or just `./run.sh`, which does all of the above.

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- Health: <http://localhost:8000/api/v1/health>

The first account registered on an empty database is promoted to
**administrator** so the admin panel is reachable. After that, the admin role
can only be granted by an existing administrator (or to an address listed in
`ADMIN_EMAILS`).

---

## What the artifacts actually contain

Read from the artifacts themselves, not assumed:

| Item | Value |
|---|---|
| Model | `artifacts/model_trainer/best_model.pkl` → `sklearn.linear_model.LinearRegression` |
| Preprocessor | `artifacts/data_transformation/preprocessor.pkl` → `ColumnTransformer` |
| Numerical pipeline | `SimpleImputer(median)` → `StandardScaler` |
| Categorical pipeline | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown="ignore")` |
| Target | `fuel_consumption_rate` |
| Transformed width | 14 (matches `model.n_features_in_`) |

**Raw feature order** (taken from `preprocessor.transformers_`, enforced on every
request):

```
sailing_speed, displacement, trim, wind_speed, wind_direction_relative,
combined_wave_height, combined_wave_period, sea_current_speed,
sea_current_direction_relative, sea_water_temperature, vessel_type
```

**`vessel_type` vocabulary** (read from the fitted `OneHotEncoder`, the only
values accepted):

```
Fishing Trawler · Oil Service Boat · Surfer Boat · Tanker Ship
```

### Four things to be aware of

1. **The model is near-degenerate.** Recomputed on the shipped held-out split
   (2172 rows): MAE ≈ 2.4e-06, RMSE ≈ 5.1e-06, **R² = 1.0**. The coefficients are
   ≈4.65 for all ten numeric features and ≈1e-8 for every `vessel_type` dummy.
   The target is an almost exact linear function of the inputs in this dataset.
   Treat that as a pipeline check, not as evidence of real-world predictive
   skill. `/prediction/model` and `/admin/model` recompute and return these
   numbers rather than copying them from a report.

2. **The target unit is undocumented.** Nothing in the artifacts records whether
   `fuel_consumption_rate` is kg/h. It is set by `MODEL_TARGET_UNIT` and every
   response carries `unit_verified: false` until you confirm it and set
   `MODEL_TARGET_UNIT_VERIFIED=true`.

3. **No confidence is reported.** The artifacts contain no interval estimator,
   so none is invented.

4. **scikit-learn version skew.** The artifacts were pickled with 1.9.1. Loading
   under a different minor version emits `InconsistentVersionWarning`; loading
   and inference still work, and the warnings are surfaced in `/health` and
   `/admin/model` rather than swallowed. Pin `scikit-learn==1.9.1` to silence them.

### Frontend schema mismatch (fixed)

The original prototype form collected `vesselType (Container/Bulk/Tanker/Ro-Ro/
Feeder)`, `fuelType`, `cargoLoadTonnes` and a `weather` dropdown. The model
accepts none of those. The Prediction, Optimization and Scenario pages now
collect the eleven real features, and the vessel-type dropdown is populated from
`/prediction/model` so it can never drift from the encoder.

---

## What is model output and what is not

| Quantity | Source |
|---|---|
| `fuel_rate` | **Trained model.** |
| `fuel_tonnes` | Legacy rate × duration with fuel assumptions; FuelCast kg/s × hours × 3.6 for Marine Diesel. |
| `cost_usd` / `cost_inr` | Fuel tonnes × configured price. **Assumption.** |
| `ghg_tonnes_co2e` | Fuel tonnes × IMO carbon factor. **Assumption.** |
| Shore-power saving | Configured port-stay allowance. **Assumption.** |
| Voyage progress | Simulated from departure time. **Not telemetry.** |

Every optimisation and scenario response embeds an `assumptions` block listing
the exact constants applied, so a reviewer can check them without reading code.
They live in `app/services/evaluator.py` and `.env`.

---

## Optimisation

Both solvers are fully implemented in `app/services/optimization.py`:

- **NSGA-II** — fast non-dominated sort, crowding distance, binary tournament,
  SBX crossover, polynomial mutation.
- **QIEA** (quantum-inspired) — a population of qubit probability registers
  collapsed each generation and updated with a rotation gate toward the
  incumbent best.

Decision variables: sailing speed (continuous), fuel type (categorical), shore
power (binary). Objectives: minimise cost and GHG, subject to an ETA limit and
optional cost/emission caps. The legacy model is the default and keeps its
existing speed search in knots. FuelCast currently searches only speed over
ground in m/s; Marine Diesel and no shore power are required because the export
has no verified fuel-type or shore-power adjustment.

For FuelCast, supply explicit m/s speed bounds to `POST /api/v1/optimization/run`:

```json
{
  "origin": "Mumbai", "destination": "Singapore", "distance_nm": 120,
  "vessel": {"vessel_type": "Tanker Ship", "displacement": 12, "trim": 0},
  "available_fuels": ["Marine Diesel"], "allow_shore_power": false,
  "algorithm": "nsga2", "model_id": "fuelcast_xgboost",
  "fuelcast_speed_bounds_m_s": [5, 10],
  "fuelcast_inputs": {
    "speed_over_ground": 8, "wind_speed": 5.5, "wind_direction": 120,
    "wave_height": 1.2, "wave_period": 7, "current_speed": 0.8
  }
}
```

The six FuelCast values use m/s for all speeds, degrees for wind direction,
metres for wave height, and seconds for wave period. The supplied
`speed_over_ground` is the reference-plan speed and must lie within the bounds.
Each optimizer candidate replaces it with its own m/s speed. Duration uses
`candidate_speed_m_s / 0.514444` knots, once; each candidate's kg/s prediction
becomes tonnes through `rate_kg_s × duration_hours × 3.6`, once. Cost and
emissions use those tonnes. Results identify the optimizer algorithm separately
from the prediction model, and record the selected speed, raw rate, duration,
fuel tonnes, and the fixed environmental snapshot. The wind-direction
absolute/relative and from/toward convention remains unverified. Holding one
environmental snapshot fixed across candidates is a prototype assumption and
does not establish physical validity. VQR integration remains pending.

**QIEA runs on a CPU. No quantum hardware is involved and no quantum advantage
is claimed.** The comparison table reports measured wall-clock runtime, model
evaluation counts and objective values for this problem on this machine, and
nothing more. A **baseline plan** is included so any improvement is measured
rather than asserted: the top of the speed band for legacy, or the supplied
reference speed for FuelCast, on the first listed fuel without shore power.

Legacy speed search is clamped to 12–19 knots, the range present in its training
data; searching outside it would be extrapolation.

## FuelCast scenarios

The existing `POST /api/v1/scenario/run` defaults to the legacy model. To use
`fuelcast_xgboost`, supply a complete six-field `fuelcast_inputs` snapshot.
This fixed-speed example compares the base case, a price change, explicit
weather, and an explicit speed-over-ground change:

```json
{
  "model_id": "fuelcast_xgboost",
  "scenarios": ["base", "high_fuel_price", "severe_weather", "fuelcast_speed_change"],
  "distance_nm": 120,
  "vessel": {"vessel_type": "Tanker Ship", "displacement": 12, "trim": 0},
  "available_fuels": ["Marine Diesel"],
  "shore_power": false,
  "optimize": false,
  "fuelcast_inputs": {
    "speed_over_ground": 8, "wind_speed": 5.5, "wind_direction": 120,
    "wave_height": 1.2, "wave_period": 7, "current_speed": 0.8
  },
  "fuelcast_scenario_inputs": {
    "severe_weather": {
      "speed_over_ground": 8, "wind_speed": 9, "wind_direction": 120,
      "wave_height": 2, "wave_period": 8, "current_speed": 0.8
    },
    "fuelcast_speed_change": {
      "speed_over_ground": 9, "wind_speed": 5.5, "wind_direction": 120,
      "wave_height": 1.2, "wave_period": 7, "current_speed": 0.8
    }
  }
}
```

Speed over ground, wind speed, and current speed are in m/s; wind direction is
in degrees, wave height in metres, and wave period in seconds. Weather and
speed changes require complete, explicit snapshots; legacy relative wind and
knots are never mapped to FuelCast inputs. For `optimize: true`, also supply
`fuelcast_speed_bounds_m_s` as `[minimum, maximum]`, with every reference speed
inside the bounds. FuelCast scenario runs use Marine Diesel only and disable
shore power. `high_cargo_demand` is unavailable because cargo load is not a
trained FuelCast feature. A price scenario changes only cost after prediction;
the emissions factor remains the stated Marine Diesel factor. Each row records
model/run identity, raw kg/s rate, input snapshot, overrides, duration, fuel
tonnes (`rate_kg_s × duration_hours × 3.6`), cost, emissions, and assumptions.
The exported wind-direction convention remains unverified. One snapshot
represents the whole voyage for each scenario; this runtime behavior does not
establish model accuracy.

---

## API surface

All routes are under `/api/v1`. Everything except `/health`, `/`, `/auth/login`
and `/auth/register` requires a bearer token.

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Public. DB, model and PDF-export status. |
| POST | `/auth/register` · `/auth/login` | Returns a JWT. |
| GET | `/auth/me` | Current user; used by the frontend to validate a stored token. |
| GET | `/prediction/model` | Metadata + recomputed holdout metrics. |
| POST | `/prediction/fuel` | Real inference. Optional `model_id` defaults to `legacy`; 503 if the requested model is unavailable. |
| GET | `/prediction/history` | |
| GET | `/optimization/algorithms` | Solver status and assumptions. |
| POST | `/optimization/run` | |
| GET | `/optimization/runs` · `/runs/latest` · `/runs/{id}` | |
| GET | `/scenario/catalog` | |
| POST | `/scenario/run` | |
| GET | `/voyage/active` · `/voyage/summary` · `/voyage/{id}` | Simulated, labelled. |
| GET | `/report/csv` · `/report/pdf` · `/report/history` | Read-only exports of stored prediction, voyage, optimization and scenario results. |
| GET/PATCH | `/admin/users`, `/admin/users/{id}/role` | Admin only. |
| GET/POST | `/admin/model`, `/admin/model/reload`, `/admin/logs`, `/admin/system` | Admin only. |

**Roles.** `admin` reaches everything. `operator`, `admin` and `researcher` can
run compute-heavy endpoints. `regulator` is read-only — it can view results,
voyages and reports but cannot trigger runs.

### Model-aware reports

`/report/csv` and `/report/pdf` keep their existing `kind` and `run_id`
parameters. `kind=voyage` also exports a stored voyage; optional `model_id`,
`optimization_algorithm`, and `source_type` filters select stored rows.
`/report/history` accepts the same filters. Prediction model identity
(`legacy` or `fuelcast_xgboost`) is separate from the optimization algorithm
(`nsga2` or quantum-inspired QIEA). QIEA is an optimizer, not a quantum
prediction model. Reports never rerun a predictor or infer accuracy from
operational results.

Exports distinguish raw prediction rates (FuelCast in kg/s), stored normalized
voyage fuel (tonnes), duration (hours), cost (USD or INR as labelled), and
emissions (tonnes CO2e). They never convert a stored fuel total again or add
rates with different units. `normalized_fuel_unit_verified` states whether a
stored tonne value has verified units; unverified legacy values are retained
with a warning and are ineligible for verified-tonne aggregates. Older records
remain readable; an `unknown` model ID or a data-quality warning means the
stored record cannot establish that
provenance or unit. Older voyages did not persist modeled fuel results, so
reports leave those values unavailable. A FuelCast report also notes that its
single environmental snapshot is neither a time-series forecast nor measured
voyage telemetry; its physical representativeness remains unverified.

The FuelCast XGBoost bundle is registered as `fuelcast_xgboost` for direct
service-level inference with `speed_over_ground` (m/s), `wind_speed` (m/s),
`wind_direction` (degrees), `wave_height` (m), `wave_period` (s), and
`current_speed` (m/s). Its output is `kg/s`. The export does not establish
the wind-direction reference convention. The caller must supply the direction
value expected by the exported FuelCast preprocessor; whether it is absolute
or relative, and from or toward, still needs confirmation before automatic
mapping from voyage weather fields.
Send all six inputs explicitly to the existing `POST /api/v1/prediction/fuel`:

```json
{
  "model_id": "fuelcast_xgboost",
  "fuelcast_inputs": {
    "speed_over_ground": 8.0,
    "wind_speed": 5.5,
    "wind_direction": 120.0,
    "wave_height": 1.2,
    "wave_period": 7.0,
    "current_speed": 0.8
  }
}
```

The response identifies `fuelcast_xgboost` and labels `fuel_rate` in `kg/s`;
`voyage_fuel_tonnes` and `duration_hours` are null. Omit `model_id` for the
existing legacy request flow. Mixing `fuelcast_inputs` with legacy inputs
returns 422. `/prediction/model` reports each predictor's load and API status.

The normal FuelCast VQR candidate is also available for **direct prediction**.
Use the same six-field `fuelcast_inputs` object above with
`"model_id": "fuelcast_vqr"`. Speed over ground, wind speed, and current speed
are in m/s; wind direction is in degrees in [0, 360); wave height is in metres;
and wave period is in seconds. The response includes `model_id`,
`model_run_id` (`fuelcast-phase1-20260928-002`), `attempt_id`
(`5ef14aa8c84e4834ba9992eee821ab5a`), and `execution_type`
(`exact quantum simulator`). It returns one raw `fuel_rate` in kg/s, without
voyage tonnes, cost, or emissions. This runs on an exact simulator, not quantum
hardware. The exported validation metrics are validation results, not held-out
test results or evidence of application accuracy. Cargo load, fuel type, and
shore power are not VQR model features. The wind-direction absolute/relative
and from/toward convention remains unverified; callers must provide the
explicit FuelCast value, not a translation of legacy relative wind. VQR use in
voyage, optimization, and scenario services remains pending.
Voyage records can also select `fuelcast_xgboost` explicitly. The legacy model
remains the default. For example, send this to `POST /api/v1/voyage`:

```json
{
  "vessel": "MV Test",
  "vessel_type": "Tanker Ship",
  "origin": "Mumbai",
  "destination": "Singapore",
  "distance_nm": 120.0,
  "speed_knots": 15.55,
  "fuel_loaded_tonnes": 300.0,
  "model_id": "fuelcast_xgboost",
  "fuelcast_inputs": {
    "speed_over_ground": 8.0,
    "wind_speed": 5.5,
    "wind_direction": 120.0,
    "wave_height": 1.2,
    "wave_period": 7.0,
    "current_speed": 0.8
  }
}
```

FuelCast inputs are speed over ground, wind speed, and current speed in m/s;
wind direction in degrees; wave height in metres; and wave period in seconds.
The voyage speed in knots determines duration only; it is never substituted
for FuelCast speed over ground. Both values must be supplied and agree within
2% or 0.10 m/s after checking `speed_knots × 0.514444` against the explicit
FuelCast value; disagreement returns 422. For the selected model, planned fuel
in tonnes is
`rate_kg_s × duration_hours × 3.6`, with the conversion performed once before
cost and emissions calculations. Cost and emissions use the existing Marine
Diesel price and tank-to-wake factor. The result records the run ID, raw kg/s
rate, duration, and normalized tonnes. One explicitly supplied environmental
snapshot represents the entire voyage; progress estimates are not measured
consumption. The export still does not confirm whether wind direction is
absolute or relative, or whether it is from or toward. Optimization and
scenario XGBoost integration are complete; VQR integration in those downstream
services remains pending.

---

## Layout

```
backend/
  app/
    main.py                  application factory, CORS, lifespan, exception handlers
    core/config.py           env-driven settings (+ tiny .env loader)
    core/security.py         PBKDF2 password hashing, HS256 JWT
    db/database.py           MongoDB repositories (the only driver-aware module)
    schemas/models.py        Pydantic v2 request/response models
    services/
      model_registry.py      artifact loading, validation, inference, metrics
      evaluator.py           the common evaluator (fuel → cost → GHG → feasibility)
      optimization.py        NSGA-II and QIEA
      scenario.py            scenario modifiers
      voyage.py              user-recorded voyage monitoring
      reporting.py           CSV (stdlib) and PDF (ReportLab)
    api/deps.py              DB handle, JWT auth, role guards
    api/routes/              one router per domain
  artifacts/                 the real trained artifacts
  tests/                     pytest suite
  scripts/verify_artifacts.py
  scripts/run_tests.py       fallback runner for machines without pytest
```

### Database

`app/db/database.py` is the only module that touches a driver. It persists
users, prediction/optimization/scenario history and voyages in MongoDB
(Atlas or self-hosted) via `pymongo`. To move to another store, re-implement
`Database`, `UserRepository`, `RunRepository`, `VoyageRepository` and
`AuditRepository` with the same method signatures. Nothing else changes.

### A note on the auth implementation

Password hashing (PBKDF2-HMAC-SHA256) and JWT signing (HS256) are implemented on
the standard library in `app/core/security.py`, so the whole auth path is
unit-testable without native crypto wheels. The tokens are standard compact JWS
and decode unchanged with `PyJWT` or `python-jose` if you prefer to swap the
module out.

---

## Testing

```bash
pytest -q                          # full suite
python scripts/run_tests.py        # fallback, no pytest required
python scripts/verify_artifacts.py # artifact + prediction + metrics check
```

The suite covers artifact loading and feature-order enforcement, rejection of
unknown categories / non-numeric / missing features, the missing-model error
path, recomputed holdout metrics, evaluator arithmetic and constraints, both
solvers (including determinism under a fixed seed and improvement over the
baseline), scenarios, voyages, repositories, JWT signing/expiry/tampering,
report rendering, and the HTTP layer (auth, 401 on every protected route, 403
for role violations, validation errors, exports).

See "Verification status" below for what was and was not run.

---

## Environment variables

See `.env.example`. The ones that matter most:

| Variable | Purpose |
|---|---|
| `SECRET_KEY` | JWT signing key. **Change it.** |
| `MONGODB_URI` | MongoDB Atlas (or local) connection string. |
| `MONGODB_DB` | Database name, `quantafleet` by default. |
| `ARTIFACTS_DIR` | Where the `.pkl` files live. |
| `MODEL_TARGET_UNIT` / `_VERIFIED` | Unit reporting for the prediction target. |
| `FUEL_PRICE_USD_PER_TONNE`, `USD_TO_INR` | Cost assumptions. |
| `CORS_ORIGINS` | Must include the Vite dev origin. |
| `ADMIN_EMAILS` | Addresses allowed to self-register as admin. |

---

## Verification status

**Run and passing (46 tests):** artifact loading, feature order, encoder
vocabulary, real predictions, input rejection, missing-artifact errors, holdout
metrics, evaluator, NSGA-II, QIEA, scenarios, voyages, repositories, JWT,
CSV/PDF rendering.

**Not run:** the 28 HTTP tests in `tests/test_api.py`. The machine this was built
on had no network access, so `fastapi`, `uvicorn`, `pydantic`, `pytest` and
`httpx` could not be installed and the server was never started. The route layer
compiles (`python -m compileall app`) but has not been exercised over HTTP.
**Run `pytest -q` after `pip install -r requirements.txt` before relying on it.**
