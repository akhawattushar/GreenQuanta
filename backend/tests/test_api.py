"""HTTP-level tests. Skipped automatically when FastAPI/httpx are absent."""

from __future__ import annotations

import pytest

API = "/api/v1"

VESSEL = {"vessel_type": "Tanker Ship", "displacement": 12.0, "trim": 0.0}
ENVIRONMENT = {
    "wind_speed": 14.0,
    "wind_direction_relative": 90.0,
    "combined_wave_height": 3.0,
    "combined_wave_period": 5.5,
    "sea_current_speed": 0.5,
    "sea_current_direction_relative": 90.0,
    "sea_water_temperature": 17.0,
}
PREDICTION_BODY = {
    "sailing_speed": 15.5,
    "vessel": VESSEL,
    "environment": ENVIRONMENT,
    "distance_nm": 1450.0,
}


# --- health ------------------------------------------------------------
def test_health_reports_dependencies(api_client):
    body = api_client.get(f"{API}/health").json()
    assert body["status"] in {"ok", "degraded"}
    assert body["model_loaded"] is True
    assert isinstance(body["pdf_export"], bool)


def test_root_banner(api_client):
    assert api_client.get("/").json()["api_prefix"] == API


def test_openapi_schema_is_served(api_client):
    assert api_client.get("/openapi.json").status_code == 200


# --- auth --------------------------------------------------------------
def test_register_returns_a_token(api_client):
    response = api_client.post(
        f"{API}/auth/register",
        json={"name": "First Admin", "email": "first@example.com", "password": "password-1234"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    # The first account on an empty database bootstraps the admin role.
    assert body["user"]["role"] == "admin"


def test_duplicate_registration_conflicts(api_client, auth_headers):
    response = api_client.post(
        f"{API}/auth/register",
        json={"name": "Again", "email": "operator@example.com", "password": "password-1234"},
    )
    assert response.status_code == 409


def test_login_and_me(api_client, auth_headers):
    login = api_client.post(
        f"{API}/auth/login",
        json={"email": "operator@example.com", "password": "test-password-123"},
    )
    assert login.status_code == 200
    token = login.json()["access_token"]
    me = api_client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["email"] == "operator@example.com"


def test_wrong_password_is_rejected(api_client, auth_headers):
    response = api_client.post(
        f"{API}/auth/login", json={"email": "operator@example.com", "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_short_password_is_rejected(api_client):
    response = api_client.post(
        f"{API}/auth/register", json={"name": "X", "email": "x@example.com", "password": "short"}
    )
    assert response.status_code == 422


# --- route protection --------------------------------------------------
PROTECTED = [
    ("get", f"{API}/prediction/model"),
    ("post", f"{API}/prediction/fuel"),
    ("post", f"{API}/optimization/run"),
    ("get", f"{API}/optimization/algorithms"),
    ("post", f"{API}/scenario/run"),
    ("get", f"{API}/voyage/active"),
    ("get", f"{API}/voyage/summary"),
    ("get", f"{API}/report/csv"),
    ("get", f"{API}/admin/users"),
]


def test_every_protected_route_rejects_anonymous_access(api_client):
    for method, path in PROTECTED:
        response = api_client.request(method.upper(), path, json={})
        assert response.status_code == 401, f"{method.upper()} {path} returned {response.status_code}"


def test_invalid_token_is_rejected(api_client):
    response = api_client.get(f"{API}/voyage/active", headers={"Authorization": "Bearer not.a.token"})
    assert response.status_code == 401


def test_operator_cannot_reach_admin_routes(api_client, auth_headers):
    assert api_client.get(f"{API}/admin/users", headers=auth_headers).status_code == 403


# --- prediction --------------------------------------------------------
def test_prediction_returns_real_model_output(api_client, auth_headers):
    response = api_client.post(f"{API}/prediction/fuel", json=PREDICTION_BODY, headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["fuel_rate"] > 0
    assert body["model_metadata"]["model_class"]
    assert body["model_metadata"]["target_column"] == "fuel_consumption_rate"
    # No fabricated confidence is ever returned.
    assert "confidence" not in body
    assert body["unit_verified"] is False


def test_prediction_rejects_unknown_vessel_type(api_client, auth_headers):
    body = {**PREDICTION_BODY, "vessel": {**VESSEL, "vessel_type": "Container"}}
    response = api_client.post(f"{API}/prediction/fuel", json=body, headers=auth_headers)
    assert response.status_code == 422
    assert "category" in response.json()["detail"].lower()


def test_prediction_rejects_out_of_range_values(api_client, auth_headers):
    body = {**PREDICTION_BODY, "sailing_speed": -5}
    assert api_client.post(f"{API}/prediction/fuel", json=body, headers=auth_headers).status_code == 422


def test_model_metadata_includes_measured_metrics(api_client, auth_headers):
    body = api_client.get(f"{API}/prediction/model", headers=auth_headers).json()
    assert body["loaded"] is True
    assert body["metrics"]["available"] is True
    assert body["metrics"]["n_samples"] > 0


# --- optimisation ------------------------------------------------------
@pytest.fixture()
def optimization_body():
    return {
        "origin": "Mumbai",
        "destination": "Singapore",
        "distance_nm": 1450.0,
        "vessel": VESSEL,
        "environment": ENVIRONMENT,
        "available_fuels": ["Marine Diesel", "LNG"],
        "max_eta_hours": 140.0,
        "algorithm": "both",
        "population_size": 12,
        "generations": 8,
    }


def test_optimization_run_and_fetch(api_client, auth_headers, optimization_body):
    response = api_client.post(f"{API}/optimization/run", json=optimization_body, headers=auth_headers)
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["best_plan"]["feasible"] is True
    assert "quantum advantage" in body["result"]["disclaimer"].lower()
    assert body["result"]["assumptions"]["co2e_tonnes_per_tonne_fuel"]

    latest = api_client.get(f"{API}/optimization/runs/latest", headers=auth_headers)
    assert latest.json()["id"] == body["run_id"]


def test_optimization_rejects_identical_ports(api_client, auth_headers, optimization_body):
    body = {**optimization_body, "destination": "Mumbai"}
    assert api_client.post(f"{API}/optimization/run", json=body, headers=auth_headers).status_code == 422


def test_latest_run_404_when_none(api_client, auth_headers):
    assert api_client.get(f"{API}/optimization/runs/latest", headers=auth_headers).status_code == 404


# --- scenarios ---------------------------------------------------------
def test_scenario_run(api_client, auth_headers):
    body = {
        "scenarios": ["base", "severe_weather"],
        "distance_nm": 1450.0,
        "vessel": VESSEL,
        "environment": ENVIRONMENT,
        "available_fuels": ["Marine Diesel", "LNG"],
        "max_eta_hours": 140.0,
        "population_size": 10,
        "generations": 6,
    }
    response = api_client.post(f"{API}/scenario/run", json=body, headers=auth_headers)
    assert response.status_code == 200
    rows = response.json()["result"]["rows"]
    assert {r["key"] for r in rows} == {"base", "severe_weather"}


def test_scenario_catalog(api_client, auth_headers):
    keys = {s["key"] for s in api_client.get(f"{API}/scenario/catalog", headers=auth_headers).json()["scenarios"]}
    assert "base" in keys


def _scenario_body():
    return {
        "scenarios": ["base", "severe_weather"],
        "distance_nm": 1450.0,
        "vessel": VESSEL,
        "environment": ENVIRONMENT,
        "available_fuels": ["Marine Diesel", "LNG"],
        "max_eta_hours": 140.0,
        "population_size": 10,
        "generations": 6,
    }


def test_scenario_history_and_fetch_by_id(api_client, auth_headers):
    """Regression test: the history/detail endpoints used to be missing
    entirely, so Scenario Analysis could never reload a saved run."""
    response = api_client.post(f"{API}/scenario/run", json=_scenario_body(), headers=auth_headers)
    run_id = response.json()["run_id"]

    latest = api_client.get(f"{API}/scenario/runs/latest", headers=auth_headers)
    assert latest.status_code == 200
    assert latest.json()["id"] == run_id

    history = api_client.get(f"{API}/scenario/runs", headers=auth_headers)
    assert history.status_code == 200
    assert any(item["id"] == run_id for item in history.json()["items"])

    detail = api_client.get(f"{API}/scenario/runs/{run_id}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["response"]["rows"]
    assert detail.json()["request"]["scenarios"] == ["base", "severe_weather"]


def test_scenario_run_not_visible_to_another_user(api_client, auth_headers):
    response = api_client.post(f"{API}/scenario/run", json=_scenario_body(), headers=auth_headers)
    run_id = response.json()["run_id"]

    other = api_client.post(
        f"{API}/auth/register",
        json={"name": "Other Operator", "email": "other-scenario-user@example.com", "password": "test-password-123"},
    )
    other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}

    assert api_client.get(f"{API}/scenario/runs/{run_id}", headers=other_headers).status_code == 404
    other_history = api_client.get(f"{API}/scenario/runs", headers=other_headers).json()
    assert all(item["id"] != run_id for item in other_history["items"])


# --- prediction history --------------------------------------------------
def test_prediction_history_scoped_to_user(api_client, auth_headers):
    """Regression test: Fuel Prediction had no way to reload saved runs."""
    response = api_client.post(f"{API}/prediction/fuel", json=PREDICTION_BODY, headers=auth_headers)
    prediction_id = response.json()["prediction_id"]

    history = api_client.get(f"{API}/prediction/history", headers=auth_headers)
    assert history.status_code == 200
    items = history.json()["items"]
    assert any(item["id"] == prediction_id for item in items)
    match = next(item for item in items if item["id"] == prediction_id)
    assert match["request"]["vessel"]["vessel_type"] == VESSEL["vessel_type"]
    assert match["response"]["fuel_rate"] > 0

    other = api_client.post(
        f"{API}/auth/register",
        json={"name": "Other Predictor", "email": "other-prediction-user@example.com", "password": "test-password-123"},
    )
    other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
    other_history = api_client.get(f"{API}/prediction/history", headers=other_headers).json()
    assert all(item["id"] != prediction_id for item in other_history["items"])


# --- voyages -----------------------------------------------------------
def test_create_list_and_delete_voyage(api_client, auth_headers):
    create = api_client.post(
        f"{API}/voyage",
        headers=auth_headers,
        json={
            "vessel": "MV Test",
            "vessel_type": "Tanker Ship",
            "origin": "Mumbai",
            "destination": "Singapore",
            "distance_nm": 1200.0,
            "speed_knots": 14.0,
            "fuel_loaded_tonnes": 300.0,
        },
    )
    assert create.status_code == 201, create.text
    voyage_id = create.json()["id"]

    body = api_client.get(f"{API}/voyage/active", headers=auth_headers).json()
    assert body["data_source"] == "user_entered"
    assert body["live_telemetry"] is False
    assert any(v["id"] == voyage_id for v in body["voyages"])

    delete = api_client.delete(f"{API}/voyage/{voyage_id}", headers=auth_headers)
    assert delete.status_code == 204


def test_fleet_summary(api_client, auth_headers):
    api_client.post(
        f"{API}/voyage",
        headers=auth_headers,
        json={
            "vessel": "MV Test",
            "vessel_type": "Tanker Ship",
            "origin": "Mumbai",
            "destination": "Singapore",
            "distance_nm": 1200.0,
            "speed_knots": 14.0,
            "fuel_loaded_tonnes": 300.0,
        },
    )
    body = api_client.get(f"{API}/voyage/summary", headers=auth_headers).json()
    assert body["total_voyages"] > 0
    assert body["live_telemetry"] is False


def test_unknown_voyage_404(api_client, auth_headers):
    assert api_client.get(f"{API}/voyage/VY-0000", headers=auth_headers).status_code == 404


def _second_operator_headers(api_client):
    """A distinct operator account, separate from the `auth_headers` fixture's user."""
    response = api_client.post(
        f"{API}/auth/register",
        json={"name": "Second Operator", "email": "second-operator@example.com", "password": "test-password-123"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_user_cannot_see_or_fetch_another_users_voyage(api_client, auth_headers):
    """Ownership, not just authentication: a normal user must not see another
    user's voyages in the list, the summary, or by direct id."""
    create = api_client.post(
        f"{API}/voyage",
        headers=auth_headers,
        json={
            "vessel": "MV Owner Only",
            "vessel_type": "Tanker Ship",
            "origin": "Mumbai",
            "destination": "Singapore",
            "distance_nm": 1200.0,
            "speed_knots": 14.0,
            "fuel_loaded_tonnes": 300.0,
        },
    )
    assert create.status_code == 201, create.text
    voyage_id = create.json()["id"]

    other = _second_operator_headers(api_client)

    # Not visible in the other user's list.
    other_active = api_client.get(f"{API}/voyage/active", headers=other).json()
    assert all(v["id"] != voyage_id for v in other_active["voyages"])

    # Not counted in the other user's summary.
    other_summary = api_client.get(f"{API}/voyage/summary", headers=other).json()
    assert other_summary["total_voyages"] == 0

    # Direct fetch by id from the other account is a 404, not a 403 (avoids
    # confirming the id exists on someone else's account).
    direct = api_client.get(f"{API}/voyage/{voyage_id}", headers=other)
    assert direct.status_code == 404

    # The other user cannot delete it either.
    other_delete = api_client.delete(f"{API}/voyage/{voyage_id}", headers=other)
    assert other_delete.status_code == 403

    # The owner can still see and fetch their own voyage.
    own_direct = api_client.get(f"{API}/voyage/{voyage_id}", headers=auth_headers)
    assert own_direct.status_code == 200


def test_admin_sees_all_voyages(api_client, auth_headers):
    """Admins get platform-level visibility; the bootstrap account in
    `auth_headers` is promoted to admin before the operator account is made."""
    create = api_client.post(
        f"{API}/voyage",
        headers=auth_headers,
        json={
            "vessel": "MV Operator Owned",
            "vessel_type": "Tanker Ship",
            "origin": "Mumbai",
            "destination": "Singapore",
            "distance_nm": 800.0,
            "speed_knots": 12.0,
            "fuel_loaded_tonnes": 150.0,
        },
    )
    voyage_id = create.json()["id"]

    admin_login = api_client.post(
        f"{API}/auth/login", json={"email": "bootstrap@example.com", "password": "password-1234"}
    )
    admin_headers = {"Authorization": f"Bearer {admin_login.json()['access_token']}"}

    admin_active = api_client.get(f"{API}/voyage/active", headers=admin_headers).json()
    assert any(v["id"] == voyage_id for v in admin_active["voyages"])
    assert api_client.get(f"{API}/voyage/{voyage_id}", headers=admin_headers).status_code == 200


def test_non_admin_cannot_list_all_users(api_client, auth_headers):
    """Role check, not just a valid token: an operator hitting an admin-only
    endpoint is rejected."""
    response = api_client.get(f"{API}/admin/users", headers=auth_headers)
    assert response.status_code == 403


# --- reports -----------------------------------------------------------
def test_csv_export_after_a_run(api_client, auth_headers, optimization_body):
    api_client.post(f"{API}/optimization/run", json=optimization_body, headers=auth_headers)
    response = api_client.get(f"{API}/report/csv?kind=optimization", headers=auth_headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.text.startswith("# GreenQuanta")


def test_csv_export_404_without_a_run(api_client, auth_headers):
    assert api_client.get(f"{API}/report/csv?kind=optimization", headers=auth_headers).status_code == 404


def test_unknown_report_kind_rejected(api_client, auth_headers):
    assert api_client.get(f"{API}/report/csv?kind=bogus", headers=auth_headers).status_code == 422


# --- admin -------------------------------------------------------------
@pytest.fixture()
def admin_headers(api_client):
    response = api_client.post(
        f"{API}/auth/register",
        json={"name": "Root", "email": "root@example.com", "password": "password-1234", "role": "admin"},
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_admin_can_list_users_and_read_model(api_client, admin_headers):
    assert api_client.get(f"{API}/admin/users", headers=admin_headers).status_code == 200
    body = api_client.get(f"{API}/admin/model", headers=admin_headers).json()
    assert body["registry"]["loaded"] is True


def test_admin_system_hides_secrets(api_client, admin_headers):
    body = api_client.get(f"{API}/admin/system", headers=admin_headers).json()
    assert "secret" not in str(body).lower() or "SECRET_KEY" not in str(body)
