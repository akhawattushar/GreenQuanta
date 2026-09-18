import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { LogoLockup } from "../components/ui/Logo";
import heroImage from "../assets/ocean-hero.jpg";

/**
 * Landing page — marketing only.
 * No fleet statistics, vessel cards, voyages or prediction results here.
 */
export default function Landing() {
  return (
    <div className="relative min-h-screen w-full overflow-hidden">
      {/* Full-screen realistic ocean photograph */}
      <img
        src={heroImage}
        alt="Container ship crossing open ocean at sunrise"
        className="absolute inset-0 h-full w-full object-cover"
        fetchPriority="high"
      />
      <div className="absolute inset-0 bg-gradient-to-b from-navy-950/75 via-navy-950/55 to-navy-950/85" />
      <div className="absolute inset-0 bg-gradient-to-r from-navy-950/70 via-transparent to-transparent" />

      <div className="relative flex min-h-screen flex-col">
        <header className="px-5 py-5 sm:px-8">
          <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4">
            <LogoLockup onDark />
            <Link
              to="/login"
              className="rounded-xl border border-white/30 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-white/10"
            >
              Login
            </Link>
          </div>
        </header>

        <main className="flex flex-1 items-center px-5 py-10 sm:px-8">
          <div className="mx-auto w-full max-w-6xl">
            <div className="max-w-2xl">
              <p className="inline-flex items-center rounded-full border border-emerald-300/40 bg-emerald-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-emerald-200">
                QuantaFleet
              </p>

              <h1 className="mt-5 text-[2rem] font-extrabold leading-[1.1] tracking-tight text-white sm:text-5xl lg:text-6xl">
                AI + Quantum for
                <br />
                Greener Oceans
              </h1>

              <p className="mt-5 max-w-xl text-base leading-relaxed text-slate-200 sm:text-lg">
                Sustainable decisions for a cleaner ocean. QuantaFleet pairs machine-learned fuel
                prediction with quantum-inspired optimisation so fleet operators can cut consumption,
                emissions and cost on every voyage.
              </p>

              <div className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-center">
                <Link
                  to="/signup"
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-500 px-6 py-3 text-[15px] font-semibold text-white transition-colors hover:bg-emerald-600"
                >
                  Get Started
                  <ArrowRight size={18} />
                </Link>
                <Link
                  to="/login"
                  className="inline-flex items-center justify-center rounded-xl border border-white/40 bg-white/5 px-6 py-3 text-[15px] font-semibold text-white backdrop-blur-sm transition-colors hover:bg-white/15"
                >
                  Login
                </Link>
              </div>
            </div>
          </div>
        </main>

        <footer className="px-5 py-5 sm:px-8">
          <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-2 border-t border-white/15 pt-4 text-[11px] text-slate-300">
            <p>© 2026 GreenQuanta · QuantaFleet</p>
          </div>
        </footer>
      </div>
    </div>
  );
}
