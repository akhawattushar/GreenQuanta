import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  api,
  errorMessage,
  getAuthToken,
  setAuthToken,
  setUnauthorizedHandler,
  type Role,
  type User,
} from "../services/api";

/**
 * Real JWT authentication against the FastAPI backend.
 *
 * The token lives in localStorage; on boot it is re-validated with GET
 * /auth/me, so a stale or revoked token logs the user out instead of letting
 * them sit on a protected page.
 */

interface AuthState {
  user: User | null;
  ready: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (name: string, email: string, password: string, role?: Role) => Promise<void>;
  logout: () => void;
}

export const roleLabels: Record<Role, string> = {
  operator: "Fleet Operator",
  admin: "Administrator",
  regulator: "Regulator / Viewer",
  researcher: "Researcher",
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  const logout = useCallback(() => {
    setAuthToken(null);
    setUser(null);
  }, []);

  // Any 401 from the API clears the session immediately.
  useEffect(() => {
    setUnauthorizedHandler(() => setUser(null));
    return () => setUnauthorizedHandler(null);
  }, []);

  // Validate the stored token before any protected route renders.
  useEffect(() => {
    let alive = true;
    if (!getAuthToken()) {
      setReady(true);
      return () => {
        alive = false;
      };
    }
    api
      .me()
      .then((me) => alive && setUser(me))
      .catch(() => {
        if (alive) setAuthToken(null);
      })
      .finally(() => alive && setReady(true));
    return () => {
      alive = false;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    try {
      const res = await api.login(email.trim().toLowerCase(), password);
      setAuthToken(res.access_token);
      setUser(res.user);
    } catch (err) {
      throw new Error(errorMessage(err));
    }
  }, []);

  const signup = useCallback(
    async (name: string, email: string, password: string, role: Role = "operator") => {
      try {
        const res = await api.register(name.trim(), email.trim().toLowerCase(), password, role);
        setAuthToken(res.access_token);
        setUser(res.user);
      } catch (err) {
        throw new Error(errorMessage(err));
      }
    },
    []
  );

  const value = useMemo<AuthState>(
    () => ({ user, ready, isAuthenticated: !!user, login, signup, logout }),
    [user, ready, login, signup, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>.");
  return ctx;
}

export type { Role, User };
