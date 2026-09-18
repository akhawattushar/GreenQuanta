# GreenQuanta / QuantaFleet

*AI + Quantum-inspired optimisation for greener ocean fleets*

```
backend/    FastAPI service + the real trained ML artifacts
frontend/   React 19 + TypeScript + Vite UI
```

## Run both

**Terminal 1 — backend**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # edit SECRET_KEY
python scripts/verify_artifacts.py      # confirms the model loads and predicts
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — frontend**

```bash
cd frontend
npm install
cp .env.example .env                    # VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

Open <http://localhost:5173>, click **Get Started**, and register. The first
account on an empty database becomes the administrator.

## The short version

- Fuel predictions come from the artifacts you supplied — `LinearRegression` +
  `ColumnTransformer`, 11 raw features, 14 transformed. Feature order and the
  `vessel_type` vocabulary are read from the fitted preprocessor and enforced on
  every request.
- If the artifacts cannot load, prediction / optimisation / scenario endpoints
  return **503 with the reason**. Nothing fake is substituted.
- Cost and greenhouse-gas numbers are derived from the model's fuel output using
  configured IMO conversion factors. They are assumptions, disclosed in an
  `assumptions` block on every response.
- NSGA-II and a quantum-inspired evolutionary algorithm are both fully
  implemented. Both are classical; runtimes and objectives are measured, and no
  quantum advantage is claimed.
- Voyage monitoring uses voyages you record yourself (no AIS/telemetry feed),
  labelled `data_source: "user_entered"`; progress/fuel burn are computed from
  the trained model, not measured.

## Two things worth knowing before a demo

**The model is near-degenerate.** On the shipped held-out split it scores
R² = 1.0 with MAE ≈ 2.4e-06. Root cause, confirmed by direct inspection of the
shipped `transformed_train/test.npz`: 9 of the 10 numerical feature columns
(`sailing_speed`, `displacement`, `trim`, `wind_speed`, `wind_direction_relative`,
`combined_wave_height`, `sea_current_speed`, `sea_current_direction_relative`,
`sea_water_temperature`) are exact duplicates of each other (pairwise r = 1.0,
zero residual), and the target is an almost-exact linear function of that one
collapsed value (r = 0.9999999999999968). `vessel_type` is essentially
uncorrelated with the target (r ≈ -0.01 to 0.02). This is a dataset artifact
(most likely a synthetic-generation bug), not something a different model
architecture would fix, and no raw CSV or training pipeline ships in this repo
to regenerate the data from. `evaluate_holdout()` now runs this collinearity
check automatically and attaches a `data_quality_warning` to `/prediction/model`
and `/admin/model` (surfaced in the UI) whenever it fires, so the R²/MAE/RMSE
numbers are never shown without the caveat. The inference pipeline itself is
correct — this is not evidence of real-world predictive skill, and a genuine
accuracy claim needs a retrain on a dataset with independent feature columns.

**The target unit is undocumented.** Nothing in the artifacts records whether
`fuel_consumption_rate` is kg/h. The API reports `unit_verified: false` until
you confirm it against the source dataset.

See `backend/README.md` for the full API surface, assumptions table and
verification status.
