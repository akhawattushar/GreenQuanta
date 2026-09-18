import { type ButtonHTMLAttributes, type ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  icon?: ReactNode;
  fullWidth?: boolean;
}

const variants: Record<string, string> = {
  primary: "bg-teal-600 text-white hover:bg-teal-700 active:bg-teal-800 shadow-card",
  secondary: "bg-navy-800 text-white hover:bg-navy-700 active:bg-navy-900 shadow-card",
  outline: "bg-white text-navy-800 border border-slate-300 hover:bg-slate-50 hover:border-slate-400",
  ghost: "bg-transparent text-navy-600 hover:bg-slate-100 hover:text-navy-900",
  danger: "bg-red-600 text-white hover:bg-red-700",
};

const sizes: Record<string, string> = {
  sm: "text-xs px-3 py-1.5 gap-1.5",
  md: "text-sm px-4 py-2.5 gap-2",
  lg: "text-[15px] px-6 py-3 gap-2",
};

export function Button({
  children,
  variant = "primary",
  size = "md",
  icon,
  fullWidth = false,
  className = "",
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-xl font-semibold transition-colors duration-150 disabled:opacity-55 disabled:cursor-not-allowed ${
        variants[variant]
      } ${sizes[size]} ${fullWidth ? "w-full" : ""} ${className}`}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}
