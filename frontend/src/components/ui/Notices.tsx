import { useState } from "react";
import { AlertTriangle, Info, ServerCrash } from "lucide-react";

/** Backend is unreachable or the trained model could not be loaded. */
export function ServiceErrorNotice({ message, title = "Service unavailable" }: { message: string; title?: string }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3">
      <ServerCrash size={16} className="mt-0.5 shrink-0 text-red-600" />
      <div className="min-w-0">
        <p className="text-xs font-bold uppercase tracking-wide text-red-800">{title}</p>
        <p className="mt-1 text-xs leading-relaxed text-red-900">{message}</p>
      </div>
    </div>
  );
}

/** Values derived from configured factors rather than learned by the model. */
export function AssumptionNotice({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
      <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-700" />
      <p className="min-w-0 text-xs leading-relaxed text-amber-900">{message}</p>
    </div>
  );
}

/** Neutral provenance note. */
export function InfoNotice({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
      <Info size={16} className="mt-0.5 shrink-0 text-slate-500" />
      <p className="min-w-0 text-xs leading-relaxed text-slate-600">{message}</p>
    </div>
  );
}

/**
 * A one-line summary with an "ⓘ" toggle that expands to the full explanation.
 * Use this instead of a long paragraph wherever the detail matters for
 * transparency but would otherwise dominate the screen.
 */
export function InfoDisclosure({ summary, details }: { summary: string; details: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
      <div className="flex items-start gap-2">
        <p className="min-w-0 flex-1 text-xs leading-relaxed text-slate-600">{summary}</p>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          className="flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold text-teal-700 hover:bg-teal-50"
        >
          <Info size={13} />
          {open ? "Less" : "Details"}
        </button>
      </div>
      {open && <p className="mt-2 border-t border-slate-200 pt-2 text-xs leading-relaxed text-slate-500">{details}</p>}
    </div>
  );
}

/** Marks data that is simulated rather than measured. */
export function SimulatedBadge({ label = "Simulated" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide text-amber-800">
      {label}
    </span>
  );
}
