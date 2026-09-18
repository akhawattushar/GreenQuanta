import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { FlaskConical, History } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Checkbox, Input } from "../components/ui/Input";
import { Badge } from "../components/ui/Badge";
import { AssumptionNotice, ServiceErrorNotice } from "../components/ui/Notices";
import { EmptyState, ErrorNote, LoadingState } from "../components/ui/States";
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
import { api, errorMessage, type FuelType, type ScenarioHistoryItem, type ScenarioResult, type ScenarioRow } from "../services/api";

const FUELS: FuelType[] = ["Marine Diesel", "LNG", "Methanol", "Ammonia", "Hydrogen"];

export default function Scenarios() {
  const catalog = useApi(() => api.scenarioCatalog(), []);
  const model = useApi(() => api.modelInfo(), []);
  const vesselTypes = model.data?.metadata?.categorical_categories?.vessel_type ?? [];
  const history = useApi(() => api.scenarioHistory(), []);
  const [searchParams, setSearchParams] = useSearchParams();

  const [selected, setSelected] = useState<string[]>([]);
  const [distance, setDistance] = useState(0);
  const [maxEta, setMaxEta] = useState(0);
  const [vessel, setVessel] = useState(EMPTY_VESSEL);
  const [environment, setEnvironment] = useState(EMPTY_ENVIRONMENT);
  const [fuels, setFuels] = useState<FuelType[]>([]);

  const [result, setResult] = useState<ScenarioResult | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const toggle = (key: string) =>
    setSelected((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  const toggleFuel = (fuel: FuelType) =>
    setFuels((prev) => (prev.includes(fuel) ? prev.filter((f) => f !== fuel) : [...prev, fuel]));

  // Deep-linkable: /scenarios?run=sce-xxxx reloads that saved run from
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

  function openHistoryItem(item: ScenarioHistoryItem) {
    setResult(item.response);
    setSelectedId(item.id);
    setSearchParams({ run: item.id }, { replace: true });
  }

  async function run() {
    if (selected.length === 0) return setError("Select at least one scenario.");
    if (distance <= 0) return setError("Enter the voyage distance.");
    if (!vesselIsComplete(vessel)) return setError("Complete the vessel details.");
    if (!environmentIsComplete(environment)) return setError("Complete the sea state, including wave period.");
    if (fuels.length === 0) return setError("Select at least one available fuel.");

    setError("");
    setBusy(true);
    setResult(null);
    setSelectedId(null);
    try {
      const response = await api.runScenarios({
        scenarios: selected,
        distance_nm: distance,
        vessel,
        environment,
        available_fuels: fuels,
        max_eta_hours: maxEta > 0 ? maxEta : null,
        optimize: true,
      });
      setResult(response.result);
      setSearchParams({ run: response.run_id }, { replace: true });
      history.reload();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <AppShell title="Scenario Analysis" subtitle="Stress-test a voyage against price, weather and regulation.">
      {model.data && !model.data.loaded && (
        <ServiceErrorNotice
          title="Model not loaded"
          message={model.data.error ?? "Scenario analysis needs the trained model."}
        />
      )}
      {catalog.error && <ServiceErrorNotice message={catalog.error} />}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-5">
        <Card className="xl:col-span-2">
          <CardHeader title="Scenario setup" icon={<FlaskConical size={18} />} />
          <div className="space-y-5">
            <fieldset>
              <legend className="field-label">Scenarios to evaluate</legend>
              {catalog.loading && <LoadingState label="Loading scenarios…" />}
              <div className="space-y-3">
                {(catalog.data?.scenarios ?? []).map((scenario) => (
                  <div key={scenario.key}>
                    <Checkbox
                      label={scenario.label}
                      checked={selected.includes(scenario.key)}
                      onChange={() => toggle(scenario.key)}
                    />
                    <p className="ml-6 mt-0.5 text-[11px] leading-relaxed text-slate-500">{scenario.description}</p>
                  </div>
                ))}
              </div>
            </fieldset>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <Input
                label="Distance"
                type="number"
                min={0}
                suffix="nm"
                placeholder="e.g. 1450"
                value={distance || ""}
                onChange={(e) => setDistance(Number(e.target.value))}
              />
              <Input
                label="Max ETA (optional)"
                type="number"
                min={0}
                suffix="hours"
                placeholder="e.g. 120"
                value={maxEta || ""}
                onChange={(e) => setMaxEta(Number(e.target.value))}
              />
            </div>

            <VesselFields value={vessel} onChange={setVessel} vesselTypes={vesselTypes} />
            <EnvironmentFields value={environment} onChange={setEnvironment} />

            <fieldset>
              <legend className="field-label">Available fuels</legend>
              <div className="grid grid-cols-2 gap-2.5">
                {FUELS.map((fuel) => (
                  <Checkbox key={fuel} label={fuel} checked={fuels.includes(fuel)} onChange={() => toggleFuel(fuel)} />
                ))}
              </div>
            </fieldset>

            {error && <ErrorNote message={error} />}

            <Button onClick={run} disabled={busy || model.data?.loaded === false} fullWidth>
              {busy ? "Running scenarios…" : "Run Scenario"}
            </Button>
          </div>
        </Card>

        <div className="space-y-5 xl:col-span-3">
          <Card>
            <CardHeader
              title="Scenario results"
              subtitle="Each row is re-optimised through the same model and evaluator."
            />
            {busy && <LoadingState label="Evaluating scenarios…" />}
            {!busy && !result && (
              <EmptyState title="No scenarios run yet" description="Pick scenarios on the left and run the analysis." />
            )}
            {!busy && result && (
              <Table<ScenarioRow>
                keyField={(r) => r.key}
                data={result.rows}
                columns={[
                  { header: "Scenario", accessor: (r) => <span className="font-semibold">{r.scenario}</span> },
                  { header: "Speed", accessor: (r) => `${r.speed_knots} kn`, align: "right" },
                  { header: "Fuel", accessor: (r) => r.fuel_type },
                  { header: "Consumption", accessor: (r) => `${r.fuel_tonnes} t`, align: "right" },
                  { header: "Cost (₹ Cr)", accessor: (r) => r.cost_inr_crore.toFixed(4), align: "right" },
                  { header: "GHG (t)", accessor: (r) => r.ghg_tonnes_co2e.toLocaleString("en-IN"), align: "right" },
                  {
                    header: "Feasible",
                    accessor: (r) => (r.feasible ? <Badge tone="green">Yes</Badge> : <Badge tone="red">No</Badge>),
                    align: "center",
                  },
                ]}
              />
            )}
          </Card>

          {result && <AssumptionNotice message={result.note} />}

          <Card>
            <CardHeader
              title="Scenario history"
              subtitle="Your saved runs, stored in MongoDB against your account."
              icon={<History size={18} />}
            />
            {history.loading && <LoadingState label="Loading scenario history…" />}
            {history.error && <ServiceErrorNotice message={history.error} />}
            {!history.loading && !history.error && (history.data?.items.length ?? 0) === 0 && (
              <EmptyState
                title="No scenario runs yet"
                description="Runs you make above are saved automatically and will appear here."
              />
            )}
            {!history.loading && (history.data?.items.length ?? 0) > 0 && (
              <Table<ScenarioHistoryItem>
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
                  { header: "Scenarios", accessor: (r) => r.request.scenarios.length, align: "right" },
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
