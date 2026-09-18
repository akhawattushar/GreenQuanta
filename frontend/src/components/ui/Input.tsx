import { type InputHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from "react";

const control =
  "w-full rounded-xl border border-slate-300 bg-white text-navy-900 placeholder-slate-400 text-sm py-2.5 focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 outline-none transition-colors disabled:bg-slate-50 disabled:text-slate-400";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  icon?: ReactNode;
  hint?: string;
  suffix?: string;
}

export function Input({ label, icon, hint, suffix, className = "", ...rest }: InputProps) {
  return (
    <label className="block">
      {label && <span className="field-label">{label}</span>}
      <div className="relative">
        {icon && <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">{icon}</span>}
        <input className={`${control} ${icon ? "pl-9" : "pl-3"} ${suffix ? "pr-14" : "pr-3"} ${className}`} {...rest} />
        {suffix && (
          <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs font-medium text-slate-400">
            {suffix}
          </span>
        )}
      </div>
      {hint && <span className="mt-1 block text-[11px] text-slate-500">{hint}</span>}
    </label>
  );
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  children: ReactNode;
}

export function Select({ label, children, className = "", ...rest }: SelectProps) {
  return (
    <label className="block">
      {label && <span className="field-label">{label}</span>}
      <select className={`${control} px-3 ${className}`} {...rest}>
        {children}
      </select>
    </label>
  );
}

export function Checkbox({ label, ...rest }: InputHTMLAttributes<HTMLInputElement> & { label: string }) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-2 text-sm text-navy-700">
      <input
        type="checkbox"
        className="h-4 w-4 shrink-0 rounded border-slate-300 accent-teal-600"
        {...rest}
      />
      {label}
    </label>
  );
}

export function Radio({ label, ...rest }: InputHTMLAttributes<HTMLInputElement> & { label: string }) {
  return (
    <label className="flex cursor-pointer select-none items-center gap-2 text-sm text-navy-700">
      <input type="radio" className="h-4 w-4 shrink-0 accent-teal-600" {...rest} />
      {label}
    </label>
  );
}
