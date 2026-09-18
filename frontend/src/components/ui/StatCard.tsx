import { type ReactNode } from "react";

export function StatCard({
  label,
  value,
  unit,
  icon,
  tone = "teal",
  badge,
}: {
  label: string;
  value: string | number;
  unit?: string;
  icon: ReactNode;
  tone?: "teal" | "green" | "navy" | "amber";
  badge?: ReactNode;
}) {
  const tones = {
    teal: "bg-teal-50 text-teal-700",
    green: "bg-emerald-50 text-emerald-700",
    navy: "bg-navy-50 text-navy-700",
    amber: "bg-amber-50 text-amber-700",
  } as const;

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-card sm:p-5">
      <div className="flex items-start justify-between gap-2">
        <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${tones[tone]}`}>{icon}</span>
        {badge}
      </div>
      <p className="mt-3 text-2xl font-extrabold leading-none tracking-tight text-navy-900 sm:text-[26px]">
        {value}
        {unit && <span className="ml-1 text-sm font-semibold text-slate-400">{unit}</span>}
      </p>
      <p className="mt-1.5 text-xs font-medium text-slate-500">{label}</p>
    </div>
  );
}
