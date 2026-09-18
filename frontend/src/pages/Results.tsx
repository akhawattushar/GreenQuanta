import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { BarChart3, CheckCircle2 } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Badge } from "../components/ui/Badge";
import { AssumptionNotice, ServiceErrorNotice } from "../components/ui/Notices";
import { EmptyState, LoadingState } from "../components/ui/States";
import { Table, type Column } from "../components/ui/Table";
import { useApi } from "../lib/useApi";
import { api, downloadReport, errorMessage, type PlanResult } from "../services/api";

type ComparisonRow = Record<string, string | number | boolean | null>;

const SERIES_COLORS: Record<string, string> = { nsga2: "#F59E0B", quantum: "#0E8C90" };

export default function ResultsPage() {
  const [searchParams] = useSearchParams();
  // A specific run id in the URL (e.g. linked from Reports) always wins, so a
  // page refresh reloads that exact saved run from MongoDB rather than
  // silently falling back to whatever happens to be most recent. With no
  // run id, this shows the latest run on the account, same as before.
  const runId = searchParams.get("run");
  const run = useApi(() => (runId ? api.optimizationRun(runId) : api.latestOptimization()), [runId]);
  const [tab, setTab] = useState<"plan" | "comparison">("plan");
  const [exportError, setExportError] = useState("");

  if (run.loading) {
    return (
      <AppShell title="Results & Comparison" subtitle="Output of the most recent optimisation run.">
        <Card>
          <LoadingState label="Loading the latest run…" />
        </Card>
      </AppShell>
    );
  }

  // A 404 here simply means nothing has been run yet on this account (or
  // the linked run id does not belong to this account / no longer exists).
  if (run.error || !run.data) {
    const notFound = (run.error ?? "").toLowerCase().includes("no optimisation") || (run.error ?? "").toLowerCase().includes("not found");
    return (
      <AppShell title="Results & Comparison" subtitle="Output of the most recent optimisation run.">
        {!notFound && run.error && <ServiceErrorNotice message={run.error} />}
        <Card>
          <EmptyState
            title={runId ? "That run could not be found" : "No optimisation has been run yet"}
            description={
              runId
                ? "It may belong to a different account or no longer exist. Run a new optimisation or pick another saved run from Reports."
                : "Set your constraints on the Fleet Optimization page and run the optimiser. The recommended plan and the algorithm comparison will appear here."
            }
            action={
              <Link to="/optimization">
                <Button size="sm">Go to Fleet Optimization</Button>
              </Link>
            }
          />
        </Card>
      </AppShell>
    );
  }

  const request = run.data.request;
  const result = run.data.response;
  const best = result.best_plan;
  const algorithms = result.algorithms_run;

  const cmpColumns: Column<ComparisonRow>[] = [
    { header: "Metric", accessor: (r) => <span className="font-medium">{String(r.metric)}</span> },
    ...algorithms.map((key) => ({
      header: result.algorithm_labels[key] ?? key,
      accessor: (r: ComparisonRow) =>
        typeof r[key] === "boolean" ? (r[key] ? "Yes" : "No") : String(r[key] ?? "—"),
      align: "right" as const,
    })),
  ];

  async function onExport(format: "csv" | "pdf") {
    setExportError("");
    try {
      await downloadReport(format, "optimization", run.data?.id);
    } catch (err) {
      setExportError(errorMessage(err));
    }
  }

  return (
    <AppShell
      title="Results & Comparison"
      subtitle={`${request.origin} → ${request.destination} · ${request.distance_nm.toLocaleString("en-IN")} nm`}
    >
      <AssumptionNotice message={result.disclaimer} />
      {exportError && <ServiceErrorNotice title="Export failed" message={exportError} />}

      <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1">
        {([
          ["plan", "Recommended plan"],
          ["comparison", "Algorithm comparison"],
        ] as const).map(([value, label]) => (
          <button
            key={value}
            onClick={() => setTab(value)}
            className={`rounded-lg px-4 py-2 text-xs font-semibold transition-colors sm:text-sm ${
              tab === value ? "bg-navy-800 text-white" : "text-navy-600 hover:bg-slate-50"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "plan" ? (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          {best && (
            <Card className="lg:col-span-1">
              <CardHeader
                title={`Best plan · ${best.label}`}
                icon={<CheckCircle2 size={18} />}
                action={best.feasible ? <Badge tone="green">Feasible</Badge> : <Badge tone="red">Infeasible</Badge>}
              />
              <dl className="space-y-2.5 text-sm">
                {[
                  ["Vessel type", best.vessel_type],
                  ["Speed", `${best.speed_knots} knots`],
                  ["Fuel", best.fuel_type],
                  ["Shore power", best.shore_power ? "Enabled" : "Disabled"],
                  ["Modelled rate", `${best.fuel_rate} ${best.fuel_rate_unit}`],
                  ["Fuel consumption", `${best.fuel_tonnes} t`],
                  ["Operational cost", `₹ ${best.cost_inr_lakh} lakh`],
                  ["GHG emissions", `${best.ghg_tonnes_co2e.toLocaleString("en-IN")} t CO₂e`],
                  ["ETA", `${best.eta_hours} hours`],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-3 border-b border-slate-100 pb-2 last:border-0">
                    <dt className="text-slate-500">{k}</dt>
                    <dd className="text-right font-semibold text-navy-900">{v}</dd>
                  </div>
                ))}
              </dl>
              {result.improvement && (
                <p className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 px-3 py-2.5 text-[11px] leading-relaxed text-emerald-900">
                  {result.improvement.vs_baseline_cost_pct}% lower cost and{" "}
                  {result.improvement.vs_baseline_ghg_pct ?? "—"}% lower emissions than the baseline plan.{" "}
                  {result.improvement.note}
                </p>
              )}
            </Card>
          )}

          <Card className="lg:col-span-2">
            <CardHeader title="All plans" subtitle="Ranked by the scalar objective; baseline included for reference." />
            <Table<PlanResult>
              keyField={(r) => r.label}
              data={result.plans}
              columns={[
                { header: "Plan", accessor: (r) => <span className="font-semibold">{r.label}</span> },
                { header: "Speed", accessor: (r) => `${r.speed_knots} kn`, align: "right" },
                { header: "Fuel", accessor: (r) => r.fuel_type },
                { header: "Shore power", accessor: (r) => (r.shore_power ? "Yes" : "No"), align: "center" },
                { header: "Consumption", accessor: (r) => `${r.fuel_tonnes} t`, align: "right" },
                { header: "Cost", accessor: (r) => `₹${r.cost_inr_lakh} L`, align: "right" },
                { header: "GHG", accessor: (r) => `${r.ghg_tonnes_co2e.toLocaleString("en-IN")} t`, align: "right" },
                { header: "ETA", accessor: (r) => `${r.eta_hours} h`, align: "right" },
                {
                  header: "Status",
                  accessor: (r) =>
                    r.feasible ? <Badge tone="green">Feasible</Badge> : <Badge tone="red">Infeasible</Badge>,
                  align: "center",
                },
              ]}
            />
          </Card>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader title="Convergence" subtitle="Scalar objective per generation." icon={<BarChart3 size={18} />} />
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={result.convergence} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="iteration" tick={{ fontSize: 11, fill: "#64748B" }} stroke="#CBD5E1" />
                  <YAxis tick={{ fontSize: 11, fill: "#64748B" }} stroke="#CBD5E1" domain={["auto", "auto"]} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #E2E8F0", fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  {algorithms.map((key) => (
                    <Line
                      key={key}
                      type="monotone"
                      dataKey={key}
                      name={result.algorithm_labels[key] ?? key}
                      stroke={SERIES_COLORS[key] ?? "#3F5F8A"}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card>
            <CardHeader title="Metric comparison" subtitle="Measured on this machine, for this problem." />
            <Table<ComparisonRow>
              keyField={(r) => String(r.metric)}
              data={result.comparison as ComparisonRow[]}
              columns={cmpColumns}
            />
          </Card>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        <Link to="/optimization">
          <Button variant="outline">Run another optimisation</Button>
        </Link>
        <Button variant="ghost" onClick={() => onExport("csv")}>
          Export CSV
        </Button>
        <Button variant="ghost" onClick={() => onExport("pdf")}>
          Export PDF
        </Button>
      </div>
    </AppShell>
  );
}
