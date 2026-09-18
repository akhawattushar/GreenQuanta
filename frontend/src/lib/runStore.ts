import type { OptimizationRunResponse } from "../services/api";

/**
 * Remembers the run id of the last optimisation so the Results page can reload
 * it after navigation or a refresh. Only the id is cached; the payload itself
 * is always re-fetched from the backend.
 */
const KEY = "gq.lastOptimizationRunId";

export function storeRunId(runId: string) {
  try {
    sessionStorage.setItem(KEY, runId);
  } catch {
    /* ignore */
  }
}

export function readRunId(): string | null {
  try {
    return sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function clearRunId() {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export type { OptimizationRunResponse };
