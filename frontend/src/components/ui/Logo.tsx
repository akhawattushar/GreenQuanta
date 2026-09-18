interface LogoProps {
  size?: number;
  className?: string;
}

/** GreenQuanta mark: leaf sprout above an ocean wave. */
export function Logo({ size = 40, className = "" }: LogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="GreenQuanta logo"
      className={className}
      style={{ width: size, height: size }}
    >
      <path d="M24 26c0-7.2 5.6-13 12.6-13C36.6 20.2 31 26 24 26Z" fill="#16A34A" />
      <path d="M24 26c0-7.2-5.6-13-12.6-13C11.4 20.2 17 26 24 26Z" fill="#34D399" opacity="0.9" />
      <path d="M23 24h2v12h-2z" fill="#15803D" />
      <path
        d="M4 36c4.5 0 4.5-3.2 9-3.2S17.5 36 22 36s4.5-3.2 9-3.2S35.5 36 40 36"
        stroke="#0E8C90"
        strokeWidth="3"
        strokeLinecap="round"
      />
      <path
        d="M8 42c4 0 4-2.6 8-2.6S20 42 24 42s4-2.6 8-2.6S36 42 40 42"
        stroke="#2FA9AC"
        strokeWidth="2.5"
        strokeLinecap="round"
        opacity="0.75"
      />
    </svg>
  );
}

/** Mark + wordmark + tagline. `onDark` switches the text to light colours. */
export function LogoLockup({
  compact = false,
  onDark = false,
  showTagline = true,
}: {
  compact?: boolean;
  onDark?: boolean;
  showTagline?: boolean;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <Logo size={compact ? 30 : 40} className="shrink-0" />
      <div className="min-w-0 leading-none">
        <p
          className={`${compact ? "text-base" : "text-xl sm:text-2xl"} font-extrabold tracking-tight ${
            onDark ? "text-white" : "text-navy-900"
          }`}
        >
          Green<span className={onDark ? "text-emerald-400" : "text-emerald-600"}>Quanta</span>
        </p>
        {showTagline && (
          <p
            className={`mt-1 text-[10px] font-medium tracking-wide sm:text-[11px] ${
              onDark ? "text-slate-200/90" : "text-slate-500"
            }`}
          >
            AI + Quantum for Greener Oceans
          </p>
        )}
      </div>
    </div>
  );
}
