import { Link } from "react-router-dom";
import { Compass } from "lucide-react";
import { Button } from "../components/ui/Button";
import { LogoLockup } from "../components/ui/Logo";
import { useAuth } from "../auth/AuthContext";

export default function NotFound() {
  const { isAuthenticated } = useAuth();
  return (
    <div className="flex min-h-screen w-full flex-col bg-slate-50">
      <div className="px-5 py-5 sm:px-8">
        <LogoLockup />
      </div>
      <div className="flex flex-1 items-center justify-center px-5 py-10">
        <div className="w-full max-w-md text-center">
          <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-teal-50 text-teal-700">
            <Compass size={26} />
          </span>
          <p className="mt-5 text-5xl font-extrabold tracking-tight text-navy-900">404</p>
          <h1 className="mt-2 text-lg font-bold text-navy-900">This page is off the chart</h1>
          <p className="mt-2 text-sm text-slate-500">
            The route you followed does not exist. Check the address or head back to a known port.
          </p>
          <div className="mt-6 flex flex-col justify-center gap-3 sm:flex-row">
            <Link to={isAuthenticated ? "/dashboard" : "/"}>
              <Button fullWidth>{isAuthenticated ? "Back to dashboard" : "Back to home"}</Button>
            </Link>
            {!isAuthenticated && (
              <Link to="/login">
                <Button variant="outline" fullWidth>
                  Login
                </Button>
              </Link>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
