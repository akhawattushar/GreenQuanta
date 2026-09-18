/**
 * API client for the QuantaFleet FastAPI backend.
 *
 * Every call goes to the real service. There is no local fallback and no
 * synthetic result: when the backend is down or the trained model cannot be
 * loaded, the call throws and the page shows the reason.
 */

const RAW_BASE = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").trim();
export const API_BASE_URL = RAW_BASE.replace(/\/+$/, "");
export const API_PREFIX = "/api/v1";
export const apiUrl = (path: string) => `${API_BASE_URL}${API_PREFIX}${path}`;

const TOKEN_KEY = "gq.auth.token";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export class NetworkError extends ApiError {
  constructor(message = "Could not reach the API server. Is the backend running?") {
    super(message, 0);
    this.name = "NetworkError";
  }
}

/** 503 — the backend is up but the trained model could not be loaded. */
export class ModelUnavailableError extends ApiError {
  constructor(message: string) {
    super(message, 503);
    this.name = "ModelUnavailableError";
  }
}

export function getAuthToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setAuthToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable */
  }
}

/** Called by the auth layer when the server rejects the stored token. */
let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(handler: (() => void) | null) {
  onUnauthorized = handler;
}

async function extractDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const detail = body?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: { loc?: (string | number)[]; msg?: string }) => {
          const field = (d.loc ?? []).filter((p) => p !== "body").join(".");
          return field ? `${field}: ${d.msg}` : d.msg;
        })
        .filter(Boolean)
        .join(" · ");
    }
  } catch {
    /* fall through */
  }
  return `${res.status} ${res.statusText}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getAuthToken();
  let res: Response;
  try {
    res = await fetch(apiUrl(path), {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init.headers ?? {}),
      },
    });
  } catch {
    throw new NetworkError();
  }

  if (res.status === 401) {
    setAuthToken(null);
    onUnauthorized?.();
    throw new ApiError(await extractDetail(res), 401);
  }
  if (res.status === 503) throw new ModelUnavailableError(await extractDetail(res));
  if (!res.ok) throw new ApiError(await extractDetail(res), res.status);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/* ------------------------------------------------------------------ */
/* Types — these mirror app/schemas/models.py                          */
/* ------------------------------------------------------------------ */

export type Role = "operator" | "admin" | "regulator" | "researcher";
export type FuelType = "Marine Diesel" | "LNG" | "Methanol" | "Ammonia" | "Hydrogen";
export type AlgorithmChoice = "nsga2" | "quantum" | "both";

export interface User {
  id: string;
  name: string;
  email: string;
  role: Role;
  status?: string;
  created_at?: string | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

/** The exact feature set the trained model was fitted on. */
export interface EnvironmentIn {
  wind_speed: number;
  wind_direction_relative: number;
  combined_wave_height: number;
  combined_wave_period: number;
  sea_current_speed: number;
  sea_current_direction_relative: number;
  sea_water_temperature: number;
}

export interface VesselIn {
  vessel_type: string;
  displacement: number;
  trim: number;
}

export interface PredictionRequest {
  sailing_speed: number;
  vessel: VesselIn;
  environment: EnvironmentIn;
  distance_nm?: number | null;
}

export interface ModelMetadata {
  model_class: string;
  preprocessor_class: string;
  target_column: string;
  target_unit: string;
  target_unit_verified: boolean;
  raw_feature_order: string[];
  numerical_features: string[];
  categorical_features: string[];
  categorical_categories: Record<string, string[]>;
  transformed_feature_names: string[];
  transformed_feature_count: number;
  artifact_paths: Record<string, string>;
  warnings: string[];
}

export interface PredictionResponse {
  fuel_rate: number;
  fuel_rate_unit: string;
  unit_verified: boolean;
  voyage_fuel_tonnes: number | null;
  duration_hours: number | null;
  model_metadata: ModelMetadata;
  features_used: Record<string, unknown>;
  prediction_id: string | null;
  note: string;
}

export interface ModelInfoResponse {
  loaded: boolean;
  error: string | null;
  metadata: ModelMetadata | null;
  metrics: {
    available: boolean;
    reason?: string;
    split?: string;
    n_samples?: number;
    mae?: number;
    rmse?: number;
    r2?: number;
    note?: string;
    data_quality_warning?: string;
  } | null;
}

export interface OptimizationRequest {
  origin: string;
  destination: string;
  distance_nm: number;
  vessel: VesselIn;
  environment: EnvironmentIn;
  available_fuels: FuelType[];
  allow_shore_power: boolean;
  max_eta_hours?: number | null;
  max_ghg_tonnes?: number | null;
  cost_weight?: number;
  algorithm: AlgorithmChoice;
  population_size?: number;
  generations?: number;
  seed?: number;
}

export interface PlanResult {
  label: string;
  vessel_type: string;
  speed_knots: number;
  fuel_type: string;
  shore_power: boolean;
  fuel_rate: number;
  fuel_rate_unit: string;
  fuel_tonnes: number;
  cost_usd: number;
  cost_inr: number;
  cost_inr_lakh: number;
  ghg_tonnes_co2e: number;
  eta_hours: number;
  feasible: boolean;
  violations: string[];
  objective: number;
}

export interface OptimizationResult {
  algorithms_run: string[];
  algorithm_labels: Record<string, string>;
  plans: PlanResult[];
  best_plan: PlanResult | null;
  pareto_front: PlanResult[];
  convergence: Record<string, number>[];
  comparison: Record<string, string | number | boolean | null>[];
  improvement: {
    vs_baseline_cost_pct: number;
    vs_baseline_ghg_pct: number | null;
    note: string;
  } | null;
  runtime_seconds: Record<string, number>;
  settings: Record<string, unknown>;
  assumptions: Record<string, unknown>;
  disclaimer: string;
}

export interface OptimizationRunResponse {
  run_id: string;
  request: OptimizationRequest;
  result: OptimizationResult;
}

export interface ScenarioRow {
  scenario: string;
  key: string;
  description: string;
  speed_knots: number;
  fuel_type: string;
  shore_power: boolean;
  fuel_rate: number;
  fuel_rate_unit: string;
  fuel_tonnes: number;
  cost_usd: number;
  cost_inr: number;
  cost_inr_crore: number;
  ghg_tonnes_co2e: number;
  eta_hours: number;
  feasible: boolean;
  violations: string[];
}

export interface ScenarioResult {
  rows: ScenarioRow[];
  mode: string;
  inputs: Record<string, unknown>;
  assumptions: Record<string, unknown>;
  note: string;
}

/** One stored run, as returned by the /history, /runs and /runs/{id} endpoints. */
export interface PredictionHistoryItem {
  id: string;
  user_id?: string;
  request: PredictionRequest;
  response: PredictionResponse;
  created_at: string;
}

export interface ScenarioHistoryItem {
  id: string;
  user_id?: string;
  request: {
    scenarios: string[];
    distance_nm: number;
    vessel: VesselIn;
    environment: EnvironmentIn;
    available_fuels: FuelType[];
    speed_knots?: number | null;
    max_eta_hours?: number | null;
  };
  response: ScenarioResult;
  created_at: string;
}

export interface OptimizationHistoryItem {
  id: string;
  user_id?: string;
  request: OptimizationRequest;
  response: OptimizationResult;
  created_at: string;
}

export interface Voyage {
  id: string;
  user_id?: string;
  created_by_name?: string;
  vessel: string;
  vessel_type: string;
  origin: string;
  destination: string;
  distance_nm: number;
  speed_knots: number;
  departed_at: string;
  planned_duration_hours: number;
  elapsed_hours: number;
  progress_pct: number;
  distance_covered_nm: number;
  distance_remaining_nm: number;
  eta: string;
  fuel_loaded_tonnes: number;
  status: "On Track" | "At Risk" | "Delayed" | "Completed";
  data_source: string;
  live_telemetry: boolean;
  fuel_model_available?: boolean;
  modelled_fuel_rate?: number;
  fuel_rate_unit?: string;
  planned_fuel_tonnes?: number;
  fuel_consumed_tonnes?: number | null;
  fuel_remaining_tonnes?: number | null;
  fuel_note?: string;
  fuel_error?: string;
}

export interface FleetSummary {
  vessels: number;
  active_voyages: number;
  total_voyages: number;
  planned_fuel_tonnes: number;
  fuel_consumed_tonnes: number;
  fuel_model_available: boolean;
  data_source: string;
  live_telemetry: boolean;
  note: string;
}

export interface VoyageCreate {
  vessel: string;
  vessel_type: string;
  origin: string;
  destination: string;
  distance_nm: number;
  speed_knots: number;
  fuel_loaded_tonnes: number;
  departed_at?: string;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  database: boolean;
  model_loaded: boolean;
  model_error: string | null;
  pdf_export: boolean;
}

export interface ScenarioCatalogItem {
  key: string;
  label: string;
  description: string;
}

export interface AuditEntry {
  id: string;
  user_id: string | null;
  action: string;
  detail: string;
  created_at: string;
}

/* ------------------------------------------------------------------ */
/* Endpoints                                                           */
/* ------------------------------------------------------------------ */

export const api = {
  health: () => request<HealthResponse>("/health"),

  // auth
  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  register: (name: string, email: string, password: string, role: Role) =>
    request<TokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ name, email, password, role }),
    }),
  me: () => request<User>("/auth/me"),

  // prediction
  modelInfo: () => request<ModelInfoResponse>("/prediction/model"),
  predictFuel: (body: PredictionRequest) =>
    request<PredictionResponse>("/prediction/fuel", { method: "POST", body: JSON.stringify(body) }),
  predictionHistory: (limit = 20) =>
    request<{ items: PredictionHistoryItem[] }>(`/prediction/history?limit=${limit}`),

  // optimization
  algorithms: () =>
    request<{
      algorithms: { key: string; label: string; status: string; description: string }[];
      speed_bounds: number[];
      speed_bounds_note: string;
      assumptions: Record<string, unknown>;
    }>("/optimization/algorithms"),
  runOptimization: (body: OptimizationRequest) =>
    request<OptimizationRunResponse>("/optimization/run", { method: "POST", body: JSON.stringify(body) }),
  latestOptimization: () => request<OptimizationHistoryItem>("/optimization/runs/latest"),
  optimizationRun: (runId: string) => request<OptimizationHistoryItem>(`/optimization/runs/${runId}`),
  optimizationHistory: (limit = 20) =>
    request<{ items: OptimizationHistoryItem[] }>(`/optimization/runs?limit=${limit}`),

  // scenarios
  scenarioCatalog: () => request<{ scenarios: ScenarioCatalogItem[] }>("/scenario/catalog"),
  runScenarios: (body: {
    scenarios: string[];
    distance_nm: number;
    vessel: VesselIn;
    environment: EnvironmentIn;
    available_fuels: FuelType[];
    speed_knots?: number | null;
    max_eta_hours?: number | null;
    optimize?: boolean;
    population_size?: number;
    generations?: number;
  }) =>
    request<{ run_id: string; result: ScenarioResult }>("/scenario/run", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  scenarioRun: (runId: string) => request<ScenarioHistoryItem>(`/scenario/runs/${runId}`),
  scenarioHistory: (limit = 20) => request<{ items: ScenarioHistoryItem[] }>(`/scenario/runs?limit=${limit}`),

  // voyages
  listVoyages: () =>
    request<{ voyages: Voyage[]; data_source: string; live_telemetry: boolean }>("/voyage/active"),
  fleetSummary: () => request<FleetSummary>("/voyage/summary"),
  createVoyage: (payload: VoyageCreate) =>
    request<Voyage>("/voyage", { method: "POST", body: JSON.stringify(payload) }),
  deleteVoyage: (id: string) => request<void>(`/voyage/${id}`, { method: "DELETE" }),

  // reports
  reportHistory: () =>
    request<{ items: { id: string; kind: string; created_at: string }[]; pdf_available: boolean }>(
      "/report/history"
    ),

  // admin
  adminUsers: () => request<{ users: User[] }>("/admin/users"),
  adminModel: () =>
    request<{ registry: Record<string, unknown>; metrics: Record<string, unknown> | null }>("/admin/model"),
  adminLogs: () => request<{ entries: AuditEntry[] }>("/admin/logs"),
  adminSystem: () => request<Record<string, unknown>>("/admin/system"),
  setUserRole: (userId: string, role: Role) =>
    request<{ id: string; role: Role }>(`/admin/users/${userId}/role`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    }),
};

/**
 * Downloads a report. Uses fetch + blob rather than a plain link so the
 * Authorization header is sent and server errors surface properly.
 */
export async function downloadReport(format: "csv" | "pdf", kind: string, runId?: string): Promise<void> {
  const token = getAuthToken();
  const query = new URLSearchParams({ kind, ...(runId ? { run_id: runId } : {}) });
  let res: Response;
  try {
    res = await fetch(apiUrl(`/report/${format}?${query.toString()}`), {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
  } catch {
    throw new NetworkError();
  }
  if (!res.ok) throw new ApiError(await extractDetail(res), res.status);

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `quantafleet-${kind}.${format}`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

/** Human-readable message for any thrown error. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}
