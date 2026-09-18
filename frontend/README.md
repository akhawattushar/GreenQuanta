# GreenQuanta · QuantaFleet — Frontend

React 19 + TypeScript + Vite + Tailwind CSS. Talks to the FastAPI backend in
`../backend`.

## Run

```bash
npm install
cp .env.example .env          # VITE_API_BASE_URL=http://localhost:8000
npm run dev                   # http://localhost:5173
npm run build                 # type-check + production build
npm run lint
```

The backend must be running, and its `CORS_ORIGINS` must include
`http://localhost:5173`.

## Authentication

Real JWT auth against the API. `src/auth/AuthContext.tsx` stores the token in
localStorage and re-validates it on boot with `GET /auth/me`, so a stale or
revoked token logs the user out instead of leaving them on a protected page.
Any 401 from any call clears the session immediately.

Route guards live in `src/routes/ProtectedRoute.tsx`:

- `/`, `/login`, `/signup` are public.
- Everything else redirects to `/login` when there is no verified session,
  including after a browser refresh.
- `/admin` additionally requires the `admin` role.
- Signed-in users visiting `/login` or `/signup` bounce to `/dashboard`.
- Unknown routes render the 404 page.

Flow: **Landing → Login / Sign Up → Dashboard → protected pages.** The first
account created on an empty database is promoted to administrator.

## No demo data

There is no local fallback and no synthetic result. Every screen calls the API;
when a call fails the page shows the server's reason. Values that are derived
from configured factors rather than learned by the model (cost, emissions) are
labelled, and voyage data is marked simulated because no telemetry feed exists.

## Forms match the model

Prediction, Optimization and Scenario Analysis collect the eleven features the
model was actually trained on, shared via `src/components/forms/VoyageInputs.tsx`.
The vessel-type dropdown is populated from `/prediction/model`, so it always
matches the fitted `OneHotEncoder` vocabulary. Nothing is prefilled.

## Responsive behaviour

- `lg` and up: fixed 256px sidebar. Below `lg`: hamburger → slide-over drawer.
- Cards stack, forms collapse to one column, tables scroll horizontally inside
  their own container so the page body never scrolls sideways.
- 16px base font with no global scale transform, so 100% and 110% browser zoom
  both render normally.

## Structure

```
src/
  assets/ocean-hero.jpg     landing + auth artwork
  auth/AuthContext.tsx      JWT session
  routes/ProtectedRoute.tsx guards
  services/api.ts           typed client; mirrors app/schemas/models.py
  lib/useApi.ts             loading / error / success hook
  lib/runStore.ts           remembers the last optimisation run id
  components/
    forms/VoyageInputs.tsx  shared vessel + sea-state fields
    layout/                 AppShell, Sidebar, Topbar, AuthLayout
    ui/                     Button, Card, Input, Table, Notices, …
  pages/                    one file per screen
```

## Verification status

Type-checked with a standalone `tsc` pass; all relative imports resolve. The
build machine had no network, so `npm install` could not run and
`npm run build` / `npm run lint` were **not** executed. Run them before relying
on this.
