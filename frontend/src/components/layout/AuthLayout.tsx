import { type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { LogoLockup } from "../ui/Logo";
import heroImage from "../../assets/ocean-hero.jpg";

export function AuthLayout({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  footer: ReactNode;
}) {
  return (
    <div className="flex min-h-screen w-full bg-slate-50">
      {/* Visual panel — desktop only */}
      <div className="relative hidden w-1/2 max-w-xl shrink-0 lg:block">
        <img src={heroImage} alt="" aria-hidden className="absolute inset-0 h-full w-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-b from-navy-950/80 to-navy-950/60" />
        <div className="relative flex h-full flex-col justify-between p-10">
          <LogoLockup onDark />
          <div>
            <p className="text-3xl font-extrabold leading-tight tracking-tight text-white">
              Sustainable decisions
              <br />
              for a cleaner ocean
            </p>
            <p className="mt-4 max-w-sm text-sm leading-relaxed text-slate-200">
              Fuel prediction, fleet optimisation and emission analytics in one operator workspace.
            </p>
          </div>
          <p className="text-[11px] text-slate-300">© 2026 GreenQuanta · QuantaFleet</p>
        </div>
      </div>

      {/* Form panel */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="px-5 pt-5 sm:px-8">
          <Link
            to="/"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-navy-800"
          >
            <ArrowLeft size={14} />
            Back to home
          </Link>
        </div>

        <div className="flex flex-1 items-center justify-center px-5 py-8 sm:px-8">
          <div className="w-full max-w-md">
            <div className="mb-6 lg:hidden">
              <LogoLockup />
            </div>
            <h1 className="text-2xl font-extrabold tracking-tight text-navy-900">{title}</h1>
            <p className="mt-1.5 text-sm text-slate-500">{subtitle}</p>

            <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-card sm:p-6">{children}</div>

            <div className="mt-5 text-center text-sm text-slate-600">{footer}</div>

            <p className="mt-6 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5 text-[11px] leading-relaxed text-slate-600">
              Accounts are stored by the QuantaFleet API and authenticated with a JWT. The first account
              created on an empty database is promoted to administrator so the admin panel is reachable.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
