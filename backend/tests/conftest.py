"""Shared fixtures.

`api_client` is skipped automatically when FastAPI/httpx are not installed, so
the service-level tests still run in a minimal environment.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests-only")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")
os.environ.setdefault("MONGODB_DB", f"quantafleet_test_{uuid.uuid4().hex[:8]}")


@pytest.fixture()
def db():
    from app.db.database import Database, DatabaseError

    try:
        database = Database()
    except DatabaseError as exc:
        pytest.skip(f"MongoDB is not reachable for tests: {exc}")
    yield database
    database._client.drop_database(database.db.name)
    database.close()


@pytest.fixture()
def environment():
    from app.services.evaluator import Environment

    return Environment(
        wind_speed=14.0,
        wind_direction_relative=90.0,
        combined_wave_height=3.0,
        combined_wave_period=5.5,
        sea_current_speed=0.5,
        sea_current_direction_relative=90.0,
        sea_water_temperature=17.0,
    )


@pytest.fixture()
def vessel():
    from app.services.evaluator import VesselState

    return VesselState(vessel_type="Tanker Ship", displacement=12.0, trim=0.0)


@pytest.fixture()
def api_client(db):
    """FastAPI TestClient wired to an in-memory database."""
    pytest.importorskip("fastapi", reason="FastAPI is not installed")
    pytest.importorskip("httpx", reason="httpx is required by TestClient")
    from fastapi.testclient import TestClient

    from app.api import deps
    from app.main import create_app

    deps.init_database(db)

    app = create_app()
    # Replace the lifespan-created database with the test one.
    with TestClient(app) as client:
        deps.init_database(db)
        yield client


@pytest.fixture()
def auth_headers(api_client):
    """A plain operator account.

    The first account created on an empty database is promoted to admin so the
    admin panel is reachable out of the box, so a throwaway account is
    registered first to make sure this one really is an operator.
    """
    api_client.post(
        "/api/v1/auth/register",
        json={"name": "Bootstrap", "email": "bootstrap@example.com", "password": "password-1234"},
    )
    response = api_client.post(
        "/api/v1/auth/register",
        json={
            "name": "Test Operator",
            "email": "operator@example.com",
            "password": "test-password-123",
            "role": "operator",
        },
    )
    assert response.status_code == 201, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
