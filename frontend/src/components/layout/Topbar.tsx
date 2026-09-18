import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown, LogOut, Menu } from "lucide-react";
import { roleLabels, useAuth } from "../../auth/AuthContext";
import { Badge } from "../ui/Badge";
import { api } from "../../services/api";
import { useApi } from "../../lib/useApi";

export function Topbar({
  title,
  subtitle,
  onOpenMenu,
}: {
  title: string;
  subtitle?: string;
  onOpenMenu: () => void;
}) {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const health = useApi(() => api.health(), []);

  return (
    <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="flex items-center gap-3 px-4 py-3 sm:px-6">
        <button
          onClick={onOpenMenu}
          className="rounded-xl border border-slate-300 p-2 text-navy-700 hover:bg-slate-50 lg:hidden"
          aria-label="Open navigation menu"
        >
          <Menu size={18} />
        </button>

        <div className="min-w-0 flex-1">
          <h1 className="truncate text-base font-bold tracking-tight text-navy-900 sm:text-lg">{title}</h1>
          {subtitle && <p className="truncate text-xs text-slate-500">{subtitle}</p>}
        </div>

        <div className="hidden sm:block">
          {health.loading ? (
            <Badge tone="slate">Checking API…</Badge>
          ) : health.error || !health.data ? (
            <Badge tone="red">API unreachable</Badge>
          ) : health.data.model_loaded ? (
            <Badge tone="green">API connected</Badge>
          ) : (
            <Badge tone="amber">Model not loaded</Badge>
          )}
        </div>

        <div className="relative shrink-0">
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-2 rounded-xl border border-slate-200 py-1.5 pl-1.5 pr-2 hover:bg-slate-50"
            aria-haspopup="menu"
            aria-expanded={open}
          >
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-navy-800 text-[11px] font-bold uppercase text-white">
              {user?.name.slice(0, 2)}
            </span>
            <span className="hidden text-left leading-tight md:block">
              <span className="block max-w-[9rem] truncate text-xs font-semibold capitalize text-navy-900">
                {user?.name}
              </span>
              <span className="block text-[10px] text-slate-500">{user && roleLabels[user.role]}</span>
            </span>
            <ChevronDown size={14} className="text-slate-400" />
          </button>

          {open && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
              <div className="absolute right-0 z-20 mt-2 w-56 rounded-xl border border-slate-200 bg-white p-1.5 shadow-panel">
                <div className="border-b border-slate-100 px-3 py-2">
                  <p className="truncate text-xs font-semibold text-navy-900">{user?.email}</p>
                  <p className="mt-0.5 text-[10px] text-slate-500">Signed in with a JWT from the API</p>
                </div>
                <button
                  onClick={() => {
                    logout();
                    navigate("/", { replace: true });
                  }}
                  className="mt-1 flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-navy-700 hover:bg-red-50 hover:text-red-700"
                >
                  <LogOut size={16} />
                  Log out
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
