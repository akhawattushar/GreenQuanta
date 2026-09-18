import { useEffect, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { Gauge, History } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { EmptyState, ErrorNote, LoadingState } from "../components/ui/States";
import { AssumptionNotice, InfoNotice, ServiceErrorNotice } from "../components/ui/Notices";
import { Table } from "../components/ui/Table";
import {
  EMPTY_ENVIRONMENT,
  EMPTY_VESSEL,
  EnvironmentFields,
  VesselFields,
  environmentIsComplete,
  vesselIsComplete,
} from "../components/forms/VoyageInputs";
import { useApi } from "../lib/useApi";
import { api, errorMessage, type PredictionHistoryItem, type PredictionResponse } from "../services/api";

export default function Prediction() {
  const model = useApi(() => api.modelInfo(), []);
  const vesselTypes = model.data?.metadata?.categorical_categories?.vessel_type ?? [];
  const history = useApi(() => api.predictionHistory(), []);
  const [searchParams, setSearchParams] = useSearchParams();

  // Nothing is prefilled — opening this page directly shows an empty form.
  const [speed, setSpeed] = useState(0);
  const [distance, setDistance] = useState(0);
  const [vessel, setVessel] = useState(EMPTY_VESSEL);
  const [environment, setEnvironment] = useState(EMPTY_ENVIRONMENT);

  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const complete = speed > 0 && vesselIsComplete(vessel) && environmentIsComplete(environment);

  // Deep-linkable: /prediction?run=pre-xxxx reloads that saved run from
  // MongoDB (via the history list already fetched), not from local state.
  useEffect(() => {
    const runId = searchParams.get("run");
    if (!runId || !history.data) return;
    const item = history.data.items.find((i) => i.id === runId);
    if (item) {
      setResult(item.response);
      setSelectedId(item.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [history.data]);

  function openHistoryItem(item: PredictionHistoryItem) {
    setResult(item.response);
    setSelectedId(item.id);
    setSearchParams({ run: item.id }, { replace: true });
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!complete) {
      setError("Enter the sailing speed, vessel details and sea state before predicting.");
      return;
    }
    setError("");
    setBusy(true);
    setResult(null);
    setSelectedId(null);
    try {
      const response = await api.predictFuel({ sailing_speed: speed, vessel, environment, distance_nm: distance || null });
      setResult(response);
      if (response.prediction_id) setSearchParams({ run: response.prediction_id }, { replace: true });
      history.reload();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setSpeed(0);
    setDistance(0);
    setVessel(EMPTY_VESSEL);
    setEnvironment(EMPTY_ENVIRONMENT);
    setResult(null);
    setSelectedId(null);
    setError("");
    setSearchParams({}, { replace: true });
  }

  return (
    <AppShell title="Fuel Prediction" subtitle="Estimate consumption with the trained model.">
      {model.data && !model.data.loaded && (
        <ServiceErrorNotice
          title="Model not loaded"
          message={model.data.error ?? "The backend could not load the trained model, so predictions are disabled."}
        />
      )}
      {model.error && <ServiceErrorNotice message={model.error} />}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
        <Card className="lg:col-span-3">
          <CardHeader
            title="Voyage parameters"
            subtitle="These are the exact features the model was trained on."
            icon={<Gauge size={18} />}
          />
          <form onSubmit={onSubmit} className="space-y-5" noValidate>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Input
                label="Sailing speed"
                type="number"
                min={0}
                step={0.1}
                suffix="knots"
                placeholder="e.g. 15.5"
                value={speed || ""}
                onChange={(e) => setSpeed(Number(e.target.value))}
              />
              <Input
                label="Voyage distance (optional)"
                type="number"
                min={0}
                suffix="nm"
                placeholder="e.g. 1450"
                hint="Adds total voyage fuel and duration to the result."
                value={distance || ""}
                onChange={(e) => setDistance(Number(e.target.value))}
              />
            </div>

            <div>
              <p className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">Vessel</p>
              <VesselFields value={vessel} onChange={setVessel} vesselTypes={vesselTypes} />
            </div>

            <div>
              <p className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">Sea state and weather</p>
              <EnvironmentFields value={environment} onChange={setEnvironment} />
            </div>

            {error && <ErrorNote message={error} />}

            <div className="flex flex-wrap gap-2">
              <Button type="submit" disabled={busy || model.data?.loaded === false}>
                {busy ? "Predicting…" : "Predict Fuel"}
              </Button>
              <Button type="button" variant="outline" onClick={reset}>
                Reset
              </Button>
            </div>
          </form>
        </Card>

        <div className="space-y-5 lg:col-span-2">
          <Card>
            <CardHeader title="Prediction result" subtitle="Returned by POST /prediction/fuel." />
            {busy && <LoadingState label="Running the model…" />}
            {!busy && !result && (
              <EmptyState
                title="No prediction yet"
                description="Fill in the voyage parameters and run a prediction."
              />
            )}
            {!busy && result && (
              <div className="space-y-4">
                <div className="rounded-xl border border-teal-200 bg-teal-50 p-4">
                  <p className="text-3xl font-extrabold tracking-tight text-teal-800">
                    {result.fuel_rate.toLocaleString("en-IN")}
                    <span className="ml-1 text-base font-semibold text-teal-600">{result.fuel_rate_unit}</span>
                  </p>
                  <p className="mt-1 text-xs font-medium text-teal-700">
                    Predicted {result.model_metadata.target_column.replace(/_/g, " ")}
                  </p>
                </div>

                <dl className="space-y-2.5 text-sm">
                  {result.voyage_fuel_tonnes !== null && (
                    <div className="flex justify-between gap-3 border-b border-slate-100 pb-2">
                      <dt className="text-slate-500">Voyage fuel</dt>
                      <dd className="font-semibold text-navy-900">{result.voyage_fuel_tonnes} t</dd>
                    </div>
                  )}
                  {result.duration_hours !== null && (
                    <div className="flex justify-between gap-3 border-b border-slate-100 pb-2">
                      <dt className="text-slate-500">Duration</dt>
                      <dd className="font-semibold text-navy-900">{result.duration_hours} h</dd>
                    </div>
                  )}
                  <div className="flex justify-between gap-3 border-b border-slate-100 pb-2">
                    <dt className="text-slate-500">Model</dt>
                    <dd className="text-right font-semibold text-navy-900">{result.model_metadata.model_class}</dd>
                  </div>
                  <div className="flex justify-between gap-3">
                    <dt className="text-slate-500">Unit verified</dt>
                    <dd className="font-semibold text-navy-900">{result.unit_verified ? "Yes" : "No"}</dd>
                  </div>
                </dl>

                <InfoNotice message={result.note} />
              </div>
            )}
          </Card>

          {model.data?.metrics?.available && (
            <Card>
              <CardHeader title="Model performance" subtitle={model.data.metrics.split} />
              <dl className="space-y-2.5 text-sm">
                {[
                  ["Samples", model.data.metrics.n_samples],
                  ["MAE", model.data.metrics.mae?.toPrecision(4)],
                  ["RMSE", model.data.metrics.rmse?.toPrecision(4)],
                  ["R²", model.data.metrics.r2?.toFixed(6)],
                ].map(([k, v]) => (
                  <div key={String(k)} className="flex justify-between gap-3 border-b border-slate-100 pb-2 last:border-0">
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="font-semibold text-navy-900">{String(v)}</dd>
                  </div>
                ))}
              </dl>
              {model.data.metrics.data_quality_warning && (
                <div className="mt-3">
                  <AssumptionNotice message={model.data.metrics.data_quality_warning} />
                </div>
              )}
              {model.data.metrics.note && (
                <p className="mt-3 text-[11px] leading-relaxed text-slate-500">{model.data.metrics.note}</p>
              )}
            </Card>
          )}

          <Card>
            <CardHeader
              title="Prediction history"
              subtitle="Your saved runs, stored in MongoDB against your account."
              icon={<History size={18} />}
            />
            {history.loading && <LoadingState label="Loading prediction history…" />}
            {history.error && <ServiceErrorNotice message={history.error} />}
            {!history.loading && !history.error && (history.data?.items.length ?? 0) === 0 && (
              <EmptyState
                title="No predictions yet"
                description="Runs you make above are saved automatically and will appear here."
              />
            )}
            {!history.loading && (history.data?.items.length ?? 0) > 0 && (
              <Table<PredictionHistoryItem>
                keyField={(r) => r.id}
                data={history.data!.items}
                columns={[
                  {
                    header: "Run",
                    accessor: (r) => (
                      <button
                        type="button"
                        onClick={() => openHistoryItem(r)}
                        className={`font-semibold underline-offset-2 hover:underline ${
                          selectedId === r.id ? "text-teal-700" : "text-navy-700"
                        }`}
                      >
                        {r.id}
                      </button>
                    ),
                  },
                  { header: "Vessel type", accessor: (r) => r.request.vessel.vessel_type },
                  {
                    header: "Fuel rate",
                    accessor: (r) => `${r.response.fuel_rate.toLocaleString("en-IN")} ${r.response.fuel_rate_unit}`,
                    align: "right",
                  },
                  {
                    header: "Recorded",
                    accessor: (r) => new Date(r.created_at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }),
                    align: "right",
                  },
                ]}
              />
            )}
          </Card>
        </div>
      </div>
    </AppShell>
  );
}
