import { Link } from "react-router-dom";
import { ArrowRight, Fuel, Route, Ship, Waves } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { StatCard } from "../components/ui/StatCard";
import { Badge } from "../components/ui/Badge";
import { InfoDisclosure, ServiceErrorNotice } from "../components/ui/Notices";
import { LoadingState } from "../components/ui/States";
import { useApi } from "../lib/useApi";
import { api } from "../services/api";
import { roleLabels, useAuth } from "../auth/AuthContext";

export default function Dashboard() {
  const { user } = useAuth();
  const summary = useApi(() => api.fleetSummary(), []);
  const voyages = useApi(() => api.listVoyages(), []);
  const health = useApi(() => api.health(), []);

  return (
    <AppShell
      title={`Welcome back, ${user?.name ?? "operator"}`}
      subtitle={user ? `${roleLabels[user.role]} · fleet overview` : undefined}
    >
      {health.data && !health.data.model_loaded && (
        <ServiceErrorNotice
          title="Model not loaded"
          message={health.data.model_error ?? "The API could not load the trained model. Prediction, optimisation and scenario pages will return an error."}
        />
      )}
      {summary.error && <ServiceErrorNotice message={summary.error} />}

      {summary.loading && <LoadingState label="Loading fleet summary…" />}

      {summary.data && (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <StatCard label="Vessels" value={summary.data.vessels} icon={<Ship size={19} />} tone="navy" />
            <StatCard
              label="Active voyages"
              value={summary.data.active_voyages}
              icon={<Route size={19} />}
              tone="teal"
            />
            <StatCard
              label="Planned fuel (all voyages)"
              value={summary.data.planned_fuel_tonnes.toLocaleString("en-IN")}
              unit="t"
              icon={<Fuel size={19} />}
              tone="amber"
            />
            <StatCard
              label="Fuel consumed to date"
              value={summary.data.fuel_consumed_tonnes.toLocaleString("en-IN")}
              unit="t"
              icon={<Waves size={19} />}
              tone="green"
            />
          </div>
          <InfoDisclosure
            summary="Insights are calculated from stored voyage records and model estimates. Verified savings require historical baseline data."
            details={summary.data.note}
          />
        </>
      )}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader
            title="Ongoing voyages"
            subtitle="Computed progress from recorded voyage data."
            action={
              <Link
                to="/voyages"
                className="inline-flex items-center gap-1 text-xs font-semibold text-teal-700 hover:underline"
              >
                View all <ArrowRight size={13} />
              </Link>
            }
          />
          {voyages.loading && <LoadingState label="Loading voyages…" />}
          {voyages.error && <ServiceErrorNotice message={voyages.error} />}
          <ul className="divide-y divide-slate-100">
            {(voyages.data?.voyages ?? []).slice(0, 4).map((voyage) => (
              <li key={voyage.id} className="flex flex-wrap items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-navy-900">{voyage.vessel}</p>
                  <p className="truncate text-xs text-slate-500">
                    {voyage.origin} → {voyage.destination} · {voyage.id}
                  </p>
                </div>
                <div className="w-full sm:w-40">
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full bg-teal-500" style={{ width: `${voyage.progress_pct}%` }} />
                  </div>
                  <p className="mt-1 text-[11px] text-slate-500">{voyage.progress_pct}% complete</p>
                </div>
                <Badge
                  tone={
                    voyage.status === "On Track"
                      ? "green"
                      : voyage.status === "Delayed"
                        ? "red"
                        : voyage.status === "At Risk"
                          ? "amber"
                          : "slate"
                  }
                >
                  {voyage.status}
                </Badge>
              </li>
            ))}
          </ul>
        </Card>

        <Card>
          <CardHeader title="Service status" subtitle="Reported by GET /health." />
          {health.loading && <LoadingState label="Checking…" />}
          {health.error && <ServiceErrorNotice message={health.error} />}
          {health.data && (
            <dl className="space-y-3 text-sm">
              {[
                ["API version", health.data.version],
                ["Environment", health.data.environment],
                ["Database", health.data.database ? "Connected" : "Unavailable"],
                ["Model loaded", health.data.model_loaded ? "Yes" : "No"],
                ["PDF export", health.data.pdf_export ? "Available" : "Unavailable"],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="flex items-center justify-between gap-3 border-b border-slate-100 pb-2.5 last:border-0"
                >
                  <dt className="text-slate-500">{label}</dt>
                  <dd className="font-semibold text-navy-900">{value}</dd>
                </div>
              ))}
            </dl>
          )}
        </Card>
      </div>
    </AppShell>
  );
}
