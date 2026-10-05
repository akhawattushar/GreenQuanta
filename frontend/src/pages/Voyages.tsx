import { useState } from "react";
import { Anchor, Navigation, Plus, Trash2 } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Input, Select } from "../components/ui/Input";
import { Modal } from "../components/ui/Modal";
import { AssumptionNotice, ServiceErrorNotice, SimulatedBadge } from "../components/ui/Notices";
import { EmptyState, LoadingState } from "../components/ui/States";
import { useApi } from "../lib/useApi";
import { useAuth } from "../auth/AuthContext";
import { api, errorMessage, type Voyage, type VoyageCreate } from "../services/api";

const tone = (status: Voyage["status"]) =>
  status === "On Track" ? "green" : status === "Delayed" ? "red" : status === "At Risk" ? "amber" : "slate";

const formatTime = (iso: string) =>
  new Date(iso).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });

const EMPTY_FORM: VoyageCreate = {
  vessel: "",
  vessel_type: "",
  origin: "",
  destination: "",
  distance_nm: 0,
  speed_knots: 0,
  fuel_loaded_tonnes: 0,
};

function NewVoyageModal({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: () => void }) {
  const model = useApi(() => api.modelInfo(), []);
  const vesselTypes = model.data?.metadata?.categorical_categories?.vessel_type ?? [];
  const [form, setForm] = useState<VoyageCreate>(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = <K extends keyof VoyageCreate>(k: K, v: VoyageCreate[K]) => setForm((f) => ({ ...f, [k]: v }));
  const isComplete =
    form.vessel.trim() &&
    form.vessel_type.trim() &&
    form.origin.trim() &&
    form.destination.trim() &&
    form.distance_nm > 0 &&
    form.speed_knots > 0 &&
    form.fuel_loaded_tonnes > 0;

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      await api.createVoyage(form);
      setForm(EMPTY_FORM);
      onCreated();
      onClose();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} title="Record a voyage">
      <div className="space-y-3">
        {error && <ServiceErrorNotice message={error} />}
        <Input label="Vessel name" value={form.vessel} onChange={(e) => set("vessel", e.target.value)} placeholder="e.g. MV Ganga Pride" />
        <Select
          label="Vessel type"
          value={form.vessel_type}
          disabled={vesselTypes.length === 0}
          onChange={(e) => set("vessel_type", e.target.value)}
        >
          <option value="">{vesselTypes.length ? "Select…" : "Loading from model…"}</option>
          {vesselTypes.map((t: string) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </Select>
        <div className="grid grid-cols-2 gap-3">
          <Input label="Origin" value={form.origin} onChange={(e) => set("origin", e.target.value)} placeholder="e.g. Mumbai" />
          <Input label="Destination" value={form.destination} onChange={(e) => set("destination", e.target.value)} placeholder="e.g. Singapore" />
        </div>
        <div className="grid grid-cols-3 gap-3">
          <Input
            label="Distance"
            type="number"
            min={0}
            suffix="nm"
            value={form.distance_nm || ""}
            onChange={(e) => set("distance_nm", Number(e.target.value))}
          />
          <Input
            label="Speed"
            type="number"
            min={0}
            suffix="kn"
            value={form.speed_knots || ""}
            onChange={(e) => set("speed_knots", Number(e.target.value))}
          />
          <Input
            label="Fuel loaded"
            type="number"
            min={0}
            suffix="t"
            value={form.fuel_loaded_tonnes || ""}
            onChange={(e) => set("fuel_loaded_tonnes", Number(e.target.value))}
          />
        </div>
        <div className="mt-2 flex justify-end gap-2">
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={submit} disabled={!isComplete || submitting}>
            {submitting ? "Saving…" : "Save voyage"}
          </Button>
        </div>
      </div>
    </Modal>
  );
}

export default function Voyages() {
  const { user } = useAuth();
  const voyages = useApi(() => api.listVoyages(), []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const list = voyages.data?.voyages ?? [];
  const active = list.find((v) => v.id === selectedId) ?? list[0] ?? null;
  const canWrite = user && ["operator", "admin", "researcher"].includes(user.role);

  const remove = async (id: string) => {
    setDeletingId(id);
    try {
      await api.deleteVoyage(id);
      if (selectedId === id) setSelectedId(null);
      voyages.reload();
    } catch {
      /* surfaced via reload's error state on retry */
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <AppShell
      title="Voyage Monitoring"
      subtitle="Progress, fuel burn and ETA for the voyages you record."
      actions={
        canWrite && (
          <Button size="sm" icon={<Plus size={16} />} onClick={() => setModalOpen(true)}>
            New voyage
          </Button>
        )
      }
    >
      <AssumptionNotice message="No AIS or telemetry feed is connected. Positions and progress are computed from each voyage's departure time; fuel burn is the trained model's rate for that vessel and speed under a nominal sea state, scaled by elapsed progress. These are not metered readings." />

      {voyages.error && <ServiceErrorNotice message={voyages.error} />}
      {voyages.loading && <LoadingState label="Loading voyages…" />}

      {!voyages.loading && !voyages.error && list.length === 0 && (
        <Card>
          <EmptyState
            title="No voyages yet"
            description="Record your first voyage to start tracking progress and fuel burn."
            action={
              canWrite && (
                <Button size="sm" icon={<Plus size={16} />} onClick={() => setModalOpen(true)}>
                  New voyage
                </Button>
              )
            }
          />
        </Card>
      )}

      {list.length > 0 && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-5">
          <Card className="lg:col-span-2" padded={false}>
            <div className="flex items-start justify-between gap-2 border-b border-slate-200 px-5 py-4">
              <div>
                <h3 className="text-[15px] font-bold text-navy-900">Active voyages</h3>
                <p className="mt-0.5 text-xs text-slate-500">{list.length} in progress</p>
              </div>
              <SimulatedBadge label="Computed" />
            </div>
            <ul className="max-h-[28rem] divide-y divide-slate-100 overflow-y-auto">
              {list.map((voyage) => (
                <li key={voyage.id}>
                  <div
                    className={`flex items-start gap-2 px-5 py-3.5 transition-colors hover:bg-slate-50 ${
                      active?.id === voyage.id ? "bg-teal-50/60" : ""
                    }`}
                  >
                    <button onClick={() => setSelectedId(voyage.id)} className="min-w-0 flex-1 text-left">
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-semibold text-navy-900">{voyage.vessel}</p>
                          <p className="truncate text-xs text-slate-500">
                            {voyage.origin} → {voyage.destination}
                          </p>
                        </div>
                        <Badge tone={tone(voyage.status)}>{voyage.status}</Badge>
                      </div>
                      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                        <div className="h-full rounded-full bg-teal-500" style={{ width: `${voyage.progress_pct}%` }} />
                      </div>
                    </button>
                    {(user?.role === "admin" || user?.id === voyage.user_id) && (
                      <button
                        onClick={() => remove(voyage.id)}
                        disabled={deletingId === voyage.id}
                        className="mt-0.5 shrink-0 rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-red-50 hover:text-red-600 disabled:opacity-50"
                        aria-label="Delete voyage"
                      >
                        <Trash2 size={15} />
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </Card>

          {active && (
            <Card className="lg:col-span-3">
              <CardHeader
                title={`${active.origin} → ${active.destination}`}
                subtitle={`${active.vessel} · ${active.vessel_type} · ${active.id}`}
                icon={<Navigation size={18} />}
                action={<Badge tone={tone(active.status)}>{active.status}</Badge>}
              />

              <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50 p-5">
                <div className="flex items-center gap-3">
                  <Anchor size={16} className="shrink-0 text-slate-400" />
                  <div className="relative h-1.5 flex-1 rounded-full bg-slate-200">
                    <div
                      className="absolute left-0 top-0 h-full rounded-full bg-teal-500"
                      style={{ width: `${active.progress_pct}%` }}
                    />
                    <span
                      className="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-navy-800 p-1"
                      style={{ left: `${active.progress_pct}%` }}
                    >
                      <Navigation size={10} className="text-white" />
                    </span>
                  </div>
                  <Anchor size={16} className="shrink-0 text-slate-400" />
                </div>
                <p className="mt-3 text-center text-xs font-medium text-slate-500">
                  {active.progress_pct}% complete · {active.distance_covered_nm.toLocaleString("en-IN")} of{" "}
                  {active.distance_nm.toLocaleString("en-IN")} nm
                </p>
              </div>

              <dl className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-4">
                {[
                  ["Speed", `${active.speed_knots} kn`],
                  ["Fuel loaded", `${active.fuel_loaded_tonnes} t`],
                  [
                    "Consumed",
                    active.fuel_consumed_tonnes === null || active.fuel_consumed_tonnes === undefined
                      ? "Unavailable"
                      : `${active.fuel_consumed_tonnes} t`,
                  ],
                  [
                    "Remaining",
                    active.fuel_remaining_tonnes === null || active.fuel_remaining_tonnes === undefined
                      ? "Unavailable"
                      : `${active.fuel_remaining_tonnes} t`,
                  ],
                  ["Departed", formatTime(active.departed_at)],
                  ["ETA", formatTime(active.eta)],
                  ["Elapsed", `${active.elapsed_hours} h`],
                  ["Planned", `${active.planned_duration_hours} h`],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-xl border border-slate-200 p-3">
                    <dt className="text-[11px] font-medium text-slate-500">{label}</dt>
                    <dd className="mt-1 text-sm font-bold text-navy-900">{value}</dd>
                  </div>
                ))}
              </dl>

              {active.created_by_name && (
                <p className="mt-4 text-[11px] text-slate-500">Recorded by {active.created_by_name}</p>
              )}
              {active.fuel_error && <div className="mt-4"><ServiceErrorNotice title="Fuel model unavailable" message={active.fuel_error} /></div>}
              {active.fuel_note && <p className="mt-4 text-[11px] leading-relaxed text-slate-500">{active.fuel_note}</p>}
            </Card>
          )}
        </div>
      )}

      <NewVoyageModal open={modalOpen} onClose={() => setModalOpen(false)} onCreated={voyages.reload} />
    </AppShell>
  );
}
