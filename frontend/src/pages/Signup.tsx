import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Lock, Mail, User } from "lucide-react";
import { AuthLayout } from "../components/layout/AuthLayout";
import { Button } from "../components/ui/Button";
import { Input, Select } from "../components/ui/Input";
import { ErrorNote } from "../components/ui/States";
import { roleLabels, useAuth, type Role } from "../auth/AuthContext";

export default function Signup() {
  const { signup } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>("operator");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await signup(name, email, password, role);
      navigate("/dashboard", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the account.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Set up a workspace for your fleet."
      footer={
        <>
          Already registered?{" "}
          <Link to="/login" className="font-semibold text-teal-700 hover:underline">
            Login
          </Link>
        </>
      }
    >
      <form onSubmit={onSubmit} className="space-y-4" noValidate>
        <Input
          label="Full name"
          placeholder="Aarav Sharma"
          autoComplete="name"
          icon={<User size={15} />}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <Input
          label="Work email"
          type="email"
          autoComplete="email"
          placeholder="you@company.com"
          icon={<Mail size={15} />}
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <Input
          label="Password"
          type="password"
          autoComplete="new-password"
          placeholder="At least 8 characters"
          icon={<Lock size={15} />}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <Select label="Role" value={role} onChange={(e) => setRole(e.target.value as Role)}>
          {(Object.keys(roleLabels) as Role[]).map((r) => (
            <option key={r} value={r}>
              {roleLabels[r]}
            </option>
          ))}
        </Select>

        {error && <ErrorNote message={error} />}

        <Button type="submit" fullWidth size="lg" disabled={busy}>
          {busy ? "Creating account…" : "Sign Up"}
        </Button>
      </form>
    </AuthLayout>
  );
}
