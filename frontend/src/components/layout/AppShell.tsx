import { useState, type ReactNode } from "react";
import { Sidebar, SidebarDrawer } from "./Sidebar";
import { Topbar } from "./Topbar";

export function AppShell({
  title,
  subtitle,
  actions,
  children,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="flex min-h-screen w-full bg-slate-50">
      <Sidebar />
      <SidebarDrawer open={menuOpen} onClose={() => setMenuOpen(false)} />

      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar title={title} subtitle={subtitle} onOpenMenu={() => setMenuOpen(true)} />
        <main className="w-full max-w-full flex-1 px-4 py-5 sm:px-6 sm:py-6">
          <div className="mx-auto w-full max-w-7xl space-y-5">
            {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
