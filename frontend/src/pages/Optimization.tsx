import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Sliders } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Checkbox, Input, Radio } from "../components/ui/Input";
import { ErrorNote } from "../components/ui/States";
import { InfoNotice, ServiceErrorNotice } from "../components/ui/Notices";
import {
  EMPTY_ENVIRONMENT,
  EMPTY_VESSEL,
  EnvironmentFields,
  VesselFields,
  environmentIsComplete,
  vesselIsComplete,
} from "../components/forms/VoyageInputs";
import { useApi } from "../lib/useApi";
import { storeRunId } from "../lib/runStore";
import { api, errorMessage, type AlgorithmChoice, type FuelType } from "../services/api";

const FUELS: FuelType[] = ["Marine Diesel", "LNG", "Methanol", "Ammonia", "Hydrogen"];

export default function Optimization() {
  const navigate = useNavigate();
  const model = useApi(() => api.modelInfo(), []);
  const solvers = useApi(() => api.algorithms(), []);
  const vesselTypes = model.data?.metadata?.categorical_categories?.vessel_type ?? [];

  const [origin, setOrigin] = useState("");
  const [destination, setDestination] = useState("");
  const [distance, setDistance] = useState(0);
  const [maxEta, setMaxEta] = useState(0);
  const [vessel, setVessel] = useState(EMPTY_VESSEL);
  const [environment, setEnvironment] = useState(EMPTY_ENVIRONMENT);
  const [fuels, setFuels] = useState<FuelType[]>([]);
  const [shorePower, setShorePower] = useState(true);
  const [algorithm, setAlgorithm] = useState<AlgorithmChoice>("both");

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const toggleFuel = (f: FuelType) =>
    setFuels((prev) => (prev.includes(f) ? prev.filter((x) => x !== f) : [...prev, f]));

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!origin.trim() || !destination.trim()) return setError("Enter an origin and a destination port.");
    if (origin.trim().toLowerCase() === destination.trim().toLowerCase())
      return setError("Origin and destination must be different.");
    if (distance <= 0) return setError("Enter the voyage distance in nautical miles.");
    if (!vesselIsComplete(vessel)) return setError("Complete the vessel details.");
    if (!environmentIsComplete(environment)) return setError("Complete the sea state, including wave period.");
    if (fuels.length === 0) return setError("Select at least one available fuel.");

    setError("");
    setBusy(true);
    try {
      const response = await api.runOptimization({
        origin: origin.trim(),
        destination: destination.trim(),
        distance_nm: distance,
        vessel,
        environment,
        available_fuels: fuels,
        allow_shore_power: shorePower,
        max_eta_hours: maxEta > 0 ? maxEta : null,
        algorithm,
      });
      storeRunId(response.run_id);
      navigate("/results");
    } catch (err) {
      setError(errorMessage(err));
      setBusy(false);
    }
  }

  return (
    <AppShell title="Fleet Optimization" subtitle="Search for a feasible, low-emission voyage plan.">
      {model.data && !model.data.loaded && (
        <ServiceErrorNotice
          title="Model not loaded"
          message={model.data.error ?? "Optimisation needs the trained model, which the backend could not load."}
        />
      )}
      {solvers.data && (
        <InfoNotice
          message={`${solvers.data.algorithms
            .map((a) => `${a.label} — ${a.status}`)
            .join(" · ")}. ${solvers.data.speed_bounds_note}`}
        />
      )}

      <Card>
        <CardHeader
          title="Optimisation inputs"
          subtitle="Sent to POST /optimization/run."
          icon={<Sliders size={18} />}
        />
        <form onSubmit={onSubmit} className="space-y-6" noValidate>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <Input label="Origin" placeholder="e.g. Mumbai" value={origin} onChange={(e) => setOrigin(e.target.value)} />
            <Input
              label="Destination"
              placeholder="e.g. Singapore"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
            />
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

          <div>
            <p className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">Vessel</p>
            <VesselFields value={vessel} onChange={setVessel} vesselTypes={vesselTypes} />
          </div>

          <div>
            <p className="mb-2 text-xs font-bold uppercase tracking-wide text-slate-500">Sea state and weather</p>
            <EnvironmentFields value={environment} onChange={setEnvironment} />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <fieldset>
              <legend className="field-label">Available fuels</legend>
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
                {FUELS.map((f) => (
                  <Checkbox key={f} label={f} checked={fuels.includes(f)} onChange={() => toggleFuel(f)} />
                ))}
                <Checkbox
                  label="Allow shore power"
                  checked={shorePower}
                  onChange={(e) => setShorePower(e.target.checked)}
                />
              </div>
            </fieldset>

            <fieldset>
              <legend className="field-label">Optimisation algorithm</legend>
              <div className="space-y-2.5">
                {([
                  ["nsga2", "GA / NSGA-II"],
                  ["quantum", "Quantum-Inspired (QIEA)"],
                  ["both", "Compare both"],
                ] as const).map(([value, label]) => (
                  <Radio
                    key={value}
                    name="algorithm"
                    label={label}
                    checked={algorithm === value}
                    onChange={() => setAlgorithm(value)}
                  />
                ))}
              </div>
            </fieldset>
          </div>

          {error && <ErrorNote message={error} />}

          <Button type="submit" size="lg" disabled={busy || model.data?.loaded === false}>
            {busy ? "Running optimisation…" : "Run Optimization"}
          </Button>
        </form>
      </Card>
    </AppShell>
  );
}
