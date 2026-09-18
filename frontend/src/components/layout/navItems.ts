import {
  BarChart3,
  FileText,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  ShieldCheck,
  Ship,
  Sliders,
  type LucideIcon,
} from "lucide-react";
import type { Role } from "../../auth/AuthContext";

export interface NavItem {
  to: string;
  label: string;
  short: string;
  icon: LucideIcon;
  roles?: Role[];
}

export const navItems: NavItem[] = [
  { to: "/dashboard", label: "Dashboard", short: "Fleet overview", icon: LayoutDashboard },
  { to: "/prediction", label: "Fuel Prediction", short: "Estimate consumption", icon: Gauge },
  { to: "/optimization", label: "Fleet Optimization", short: "Plan the voyage", icon: Sliders },
  { to: "/results", label: "Results & Comparison", short: "Compare algorithms", icon: BarChart3 },
  { to: "/scenarios", label: "Scenario Analysis", short: "Stress-test plans", icon: FlaskConical },
  { to: "/voyages", label: "Voyage Monitoring", short: "Track live voyages", icon: Ship },
  { to: "/reports", label: "Reports & Export", short: "Download & share", icon: FileText },
  { to: "/admin", label: "Admin Panel", short: "Users, data, models", icon: ShieldCheck, roles: ["admin"] },
];
