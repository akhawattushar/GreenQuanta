import { Route, Routes } from "react-router-dom";
import { ProtectedRoute, PublicOnlyRoute } from "./routes/ProtectedRoute";
import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import Prediction from "./pages/Prediction";
import Optimization from "./pages/Optimization";
import ResultsPage from "./pages/Results";
import Scenarios from "./pages/Scenarios";
import Voyages from "./pages/Voyages";
import Reports from "./pages/Reports";
import Admin from "./pages/Admin";
import NotFound from "./pages/NotFound";

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/" element={<Landing />} />

      {/* Public only — redirect signed-in users to the dashboard */}
      <Route element={<PublicOnlyRoute />}>
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
      </Route>

      {/* Protected — every internal page */}
      <Route element={<ProtectedRoute />}>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/optimization" element={<Optimization />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/scenarios" element={<Scenarios />} />
        <Route path="/voyages" element={<Voyages />} />
        <Route path="/reports" element={<Reports />} />
      </Route>

      {/* Protected + admin role only */}
      <Route element={<ProtectedRoute roles={["admin"]} />}>
        <Route path="/admin" element={<Admin />} />
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
