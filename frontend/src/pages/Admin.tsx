import { useState } from "react";
import { Database, KeyRound, ScrollText, Server, Users } from "lucide-react";
import { AppShell } from "../components/layout/AppShell";
import { Card, CardHeader } from "../components/ui/Card";
import { Badge } from "../components/ui/Badge";
import { AssumptionNotice, InfoNotice, ServiceErrorNotice } from "../components/ui/Notices";
import { EmptyState, LoadingState } from "../components/ui/States";
import { Table } from "../components/ui/Table";
import { Select } from "../components/ui/Input";
import { useApi } from "../lib/useApi";
import { api, errorMessage, type AuditEntry, type Role, type User } from "../services/api";
import { roleLabels, useAuth } from "../auth/AuthContext";

const tabs = [
  { id: "users", label: "Manage users", icon: Users },
  { id: "models", label: "Model registry", icon: Server },
  { id: "data", label: "Dataset status", icon: Database },
  { id: "access", label: "Access control", icon: KeyRound },
  { id: "logs", label: "System logs", icon: ScrollText },
] as const;

/** What each role can actually do, mirrored from the API's route guards. */
const ROLE_PERMISSIONS: Record<Role, string> = {
  admin: "Full access, including user management and the model registry",
  operator: "Run predictions, optimisations and scenarios; view results and reports",
  researcher: "Run predictions, optimisations and scenarios; view results and reports",
  regulator: "Read-only: view results, voyages and reports",
};

export default function Admin() {
  const { user: me } = useAuth();
  const [tab, setTab] = useState<(typeof tabs)[number]["id"]>("users");
  const [roleError, setRoleError] = useState("");

  const users = useApi(() => api.adminUsers(), []);
  const model = useApi(() => api.adminModel(), []);
  const logs = useApi(() => api.adminLogs(), []);
  const system = useApi(() => api.adminSystem(), []);

  async function changeRole(userId: string, role: Role) {
    setRoleError("");
    try {
      await api.setUserRole(userId, role);
      users.reload();
    } catch (err) {
      setRoleError(errorMessage(err));
    }
  }

  const registry = (model.data?.registry ?? {}) as Record<string, unknown>;
  const metrics = (model.data?.metrics ?? {}) as Record<string, unknown>;
  const warnings = (registry.warnings as string[] | undefined) ?? [];
  const numericalCount = (registry.numerical_features as string[] | undefined)?.length;
  const categoricalCount = (registry.categorical_features as string[] | undefined)?.length;
  const rawFeatureCount =
    numericalCount !== undefined && categoricalCount !== undefined ? numericalCount + categoricalCount : undefined;

  return (
    <AppShell title="Admin Panel" subtitle="Administrator-only controls for users, data and models.">
      <div className="flex flex-wrap gap-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`inline-flex items-center gap-2 rounded-xl border px-3.5 py-2 text-xs font-semibold transition-colors sm:text-sm ${
              tab === id
                ? "border-navy-800 bg-navy-800 text-white"
                : "border-slate-200 bg-white text-navy-600 hover:bg-slate-50"
            }`}
          >
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      {tab === "users" && (
        <Card>
          <CardHeader title="Users" subtitle="Role-based access control across the platform." />
          {roleError && <ServiceErrorNotice title="Could not change role" message={roleError} />}
          {users.loading && <LoadingState label="Loading users…" />}
          {users.error && <ServiceErrorNotice message={users.error} />}
          {users.data && (
            <Table<User>
              keyField={(u) => u.id}
              data={users.data.users}
              columns={[
                { header: "Name", accessor: (u) => <span className="font-semibold">{u.name}</span> },
                { header: "Email", accessor: (u) => u.email },
                {
                  header: "Role",
                  accessor: (u) =>
                    u.id === me?.id ? (
                      <Badge tone="navy">{roleLabels[u.role]} (you)</Badge>
                    ) : (
                      <Select
                        value={u.role}
                        onChange={(e) => changeRole(u.id, e.target.value as Role)}
                        aria-label={`Role for ${u.email}`}
                      >
                        {(Object.keys(roleLabels) as Role[]).map((r) => (
                          <option key={r} value={r}>
                            {roleLabels[r]}
                          </option>
                        ))}
                      </Select>
                    ),
                },
                {
                  header: "Status",
                  accessor: (u) => <Badge tone={u.status === "active" ? "green" : "amber"}>{u.status}</Badge>,
                  align: "center",
                },
              ]}
            />
          )}
        </Card>
      )}

      {tab === "models" && (
        <Card>
          <CardHeader title="Model registry" subtitle="Loaded artifacts and metrics measured on the shipped test split." />
          {model.loading && <LoadingState label="Loading model status…" />}
          {model.error && <ServiceErrorNotice message={model.error} />}
          {model.data && (
            <>
              <dl className="space-y-2.5 text-sm">
                {[
                  ["Loaded", registry.loaded ? "Yes" : "No"],
                  ["Model class", String(registry.model_class ?? "—")],
                  ["Preprocessor", String(registry.preprocessor_class ?? "—")],
                  ["Target column", String(registry.target_column ?? "—")],
                  ["Target unit", `${registry.target_unit ?? "—"} (verified: ${registry.target_unit_verified ? "yes" : "no"})`],
                  ["Transformed features", String(registry.transformed_feature_count ?? "—")],
                  ["Test samples", String(metrics.n_samples ?? "—")],
                  ["MAE", metrics.mae !== undefined ? Number(metrics.mae).toPrecision(4) : "—"],
                  ["RMSE", metrics.rmse !== undefined ? Number(metrics.rmse).toPrecision(4) : "—"],
                  ["R²", metrics.r2 !== undefined ? Number(metrics.r2).toFixed(6) : "—"],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="flex justify-between gap-3 border-b border-slate-100 pb-2 last:border-0"
                  >
                    <dt className="text-slate-500">{label}</dt>
                    <dd className="text-right font-semibold text-navy-900">{value}</dd>
                  </div>
                ))}
              </dl>
              {registry.error ? <div className="mt-4"><ServiceErrorNotice message={String(registry.error)} /></div> : null}
              {warnings.length > 0 && (
                <div className="mt-4 space-y-2">
                  {warnings.map((warning) => (
                    <p
                      key={warning}
                      className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] leading-relaxed text-amber-900"
                    >
                      {warning}
                    </p>
                  ))}
                </div>
              )}
              {metrics.data_quality_warning ? (
                <div className="mt-4">
                  <AssumptionNotice message={String(metrics.data_quality_warning)} />
                </div>
              ) : null}
              {metrics.note ? <p className="mt-3 text-[11px] leading-relaxed text-slate-500">{String(metrics.note)}</p> : null}
            </>
          )}
        </Card>
      )}

      {tab === "data" && (
        <Card>
          <CardHeader title="Dataset status" subtitle="Ingestion and validation run in the offline training pipeline." />
          {model.loading && <LoadingState label="Loading dataset status…" />}
          {model.error && <ServiceErrorNotice message={model.error} />}
          {model.data && (
            <dl className="space-y-2.5 text-sm">
              {[
                ["Preprocessor loaded", registry.loaded ? "Yes" : "No"],
                ["Raw features", rawFeatureCount !== undefined ? String(rawFeatureCount) : "—"],
                ["Transformed features", String(registry.transformed_feature_count ?? "—")],
                ["Held-out test samples", String(metrics.n_samples ?? "—")],
              ].map(([label, value]) => (
                <div key={label} className="flex justify-between gap-3 border-b border-slate-100 pb-2 last:border-0">
                  <dt className="text-slate-500">{label}</dt>
                  <dd className="text-right font-semibold text-navy-900">{value}</dd>
                </div>
              ))}
            </dl>
          )}
          <InfoNotice message="Dataset upload and re-ingestion are not available from this panel. The artifacts currently loaded are versioned under Model registry." />
        </Card>
      )}

      {tab === "access" && (
        <Card>
          <CardHeader title="Access control" subtitle="Role-based permissions enforced by the API on every request." />
          <Table<{ role: Role; permissions: string }>
            keyField={(r) => r.role}
            data={(Object.keys(ROLE_PERMISSIONS) as Role[]).map((role) => ({ role, permissions: ROLE_PERMISSIONS[role] }))}
            columns={[
              { header: "Role", accessor: (r) => <Badge tone="navy">{roleLabels[r.role]}</Badge> },
              { header: "Permissions", accessor: (r) => r.permissions },
            ]}
          />

          <div className="mt-6">
            <CardHeader title="Runtime configuration" subtitle="No secrets are shown." />
            {system.loading && <LoadingState label="Loading configuration…" />}
            {system.error && <ServiceErrorNotice message={system.error} />}
            {system.data && (
              <dl className="space-y-2.5 text-sm">
                {Object.entries(system.data).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-3 border-b border-slate-100 pb-2 last:border-0">
                    <dt className="text-slate-500">{key.replace(/_/g, " ")}</dt>
                    <dd className="max-w-[60%] break-words text-right font-semibold text-navy-900">
                      {Array.isArray(value) ? value.join(", ") : String(value)}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
        </Card>
      )}

      {tab === "logs" && (
        <Card>
          <CardHeader title="Audit log" subtitle="Recent authenticated actions recorded by the API." />
          {logs.loading && <LoadingState label="Loading logs…" />}
          {logs.error && <ServiceErrorNotice message={logs.error} />}
          {logs.data && logs.data.entries.length === 0 && <EmptyState title="No entries yet" />}
          {logs.data && logs.data.entries.length > 0 && (
            <Table<AuditEntry>
              keyField={(entry) => entry.id}
              data={logs.data.entries}
              columns={[
                { header: "Action", accessor: (e) => <span className="font-semibold">{e.action}</span> },
                { header: "Detail", accessor: (e) => e.detail || "—" },
                {
                  header: "When",
                  accessor: (e) =>
                    new Date(e.created_at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }),
                  align: "right",
                },
              ]}
            />
          )}
        </Card>
      )}
    </AppShell>
  );
}
