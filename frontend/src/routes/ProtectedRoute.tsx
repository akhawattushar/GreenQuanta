import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth, type Role } from "../auth/AuthContext";

/** Blocks every internal page until a verified API session exists. */
export function ProtectedRoute({ roles }: { roles?: Role[] }) {
  const { isAuthenticated, ready, user } = useAuth();
  const location = useLocation();

  // Wait for the stored token to be re-validated against GET /auth/me so a
  // refresh does not flash a redirect (or let a revoked token through).
  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-200 border-t-teal-500" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  }

  if (roles && user && !roles.includes(user.role)) {
    return <Navigate to="/dashboard" replace />;
  }

  return <Outlet />;
}

/** Keeps signed-in users away from the login/signup screens. */
export function PublicOnlyRoute() {
  const { isAuthenticated, ready } = useAuth();
  if (!ready) return null;
  if (isAuthenticated) return <Navigate to="/dashboard" replace />;
  return <Outlet />;
}
