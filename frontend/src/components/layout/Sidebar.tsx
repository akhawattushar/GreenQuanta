import { NavLink } from "react-router-dom";
import { LogOut, X } from "lucide-react";
import { navItems } from "./navItems";
import { Logo } from "../ui/Logo";
import { useAuth, roleLabels } from "../../auth/AuthContext";

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  const { user } = useAuth();
  const items = navItems.filter((i) => !i.roles || (user && i.roles.includes(user.role)));

  return (
    <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
      {items.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${
              isActive
                ? "bg-teal-50 text-teal-800 ring-1 ring-inset ring-teal-200"
                : "text-navy-600 hover:bg-slate-100 hover:text-navy-900"
            }`
          }
        >
          <Icon size={18} className="shrink-0" />
          <span className="truncate">{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}

function Brand({ onClose }: { onClose?: () => void }) {
  return (
    <div className="flex items-center justify-between gap-2 border-b border-slate-200 px-4 py-4">
      <div className="flex min-w-0 items-center gap-2.5">
        <Logo size={32} className="shrink-0" />
        <div className="min-w-0 leading-none">
          <p className="truncate text-sm font-extrabold text-navy-900">
            Green<span className="text-emerald-600">Quanta</span>
          </p>
          <p className="mt-1 text-[10px] font-medium text-slate-500">QuantaFleet</p>
        </div>
      </div>
      {onClose && (
        <button onClick={onClose} className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-100" aria-label="Close menu">
          <X size={18} />
        </button>
      )}
    </div>
  );
}

function Footer() {
  const { user, logout } = useAuth();
  return (
    <div className="border-t border-slate-200 px-3 py-3">
      {user && (
        <div className="mb-2 flex items-center gap-2.5 rounded-xl bg-slate-50 px-3 py-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-navy-800 text-[11px] font-bold uppercase text-white">
            {user.name.slice(0, 2)}
          </span>
          <div className="min-w-0 leading-tight">
            <p className="truncate text-xs font-semibold capitalize text-navy-900">{user.name}</p>
            <p className="truncate text-[10px] text-slate-500">{roleLabels[user.role]}</p>
          </div>
        </div>
      )}
      <button
        onClick={logout}
        className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium text-navy-600 transition-colors hover:bg-red-50 hover:text-red-700"
      >
        <LogOut size={17} />
        Log out
      </button>
    </div>
  );
}

/** Fixed sidebar for desktop (lg and up). */
export function Sidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-slate-200 bg-white lg:flex">
      <Brand />
      <NavList />
      <Footer />
    </aside>
  );
}

/** Slide-over drawer for tablet and mobile. */
export function SidebarDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <div className={`fixed inset-0 z-50 lg:hidden ${open ? "" : "pointer-events-none"}`} aria-hidden={!open}>
      <div
        className={`absolute inset-0 bg-navy-950/40 transition-opacity duration-200 ${open ? "opacity-100" : "opacity-0"}`}
        onClick={onClose}
      />
      <div
        className={`absolute left-0 top-0 flex h-full w-[min(17rem,85vw)] flex-col bg-white shadow-panel transition-transform duration-200 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Brand onClose={onClose} />
        <NavList onNavigate={onClose} />
        <Footer />
      </div>
    </div>
  );
}
