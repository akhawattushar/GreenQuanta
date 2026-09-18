"""Repository layer."""

import pytest

from app.db.database import AuditRepository, DatabaseError, RunRepository, UserRepository, VoyageRepository


def test_user_create_and_lookup(db):
    repo = UserRepository(db)
    user = repo.create(name="A", email="Person@Example.com", password_hash="h", role="operator")
    assert repo.get_by_email("person@example.com")["id"] == user["id"]
    assert repo.get(user["id"])["role"] == "operator"
    assert repo.count() == 1


def test_duplicate_email_rejected(db):
    repo = UserRepository(db)
    repo.create(name="A", email="a@b.com", password_hash="h", role="operator")
    with pytest.raises(DatabaseError):
        repo.create(name="B", email="a@b.com", password_hash="h", role="operator")


def test_set_role(db):
    repo = UserRepository(db)
    user = repo.create(name="A", email="a@b.com", password_hash="h", role="operator")
    assert repo.set_role(user["id"], "admin")
    assert repo.get(user["id"])["role"] == "admin"


def test_run_repository_round_trip(db):
    user = UserRepository(db).create(name="A", email="a@b.com", password_hash="h", role="operator")
    repo = RunRepository(db, "optimization")
    run_id = repo.save(user_id=user["id"], request={"origin": "Mumbai"}, response={"plans": [1, 2]})
    record = repo.get(run_id, user["id"])
    assert record["request"]["origin"] == "Mumbai"
    assert record["response"]["plans"] == [1, 2]
    assert repo.latest(user["id"])["id"] == run_id
    assert repo.get(run_id, "someone-else") is None


def test_voyage_and_audit(db):
    voyages = VoyageRepository(db)
    voyages.create(
        {
            "id": "VY-1",
            "user_id": "usr-test",
            "vessel": "MV Test",
            "vessel_type": "Tanker Ship",
            "origin": "A",
            "destination": "B",
            "distance_nm": 100.0,
            "speed_knots": 10.0,
            "fuel_loaded_tonnes": 50.0,
            "departed_at": "2026-01-01T00:00:00+00:00",
            "data_source": "user_entered",
        }
    )
    assert voyages.count() == 1
    assert voyages.get("VY-1")["vessel"] == "MV Test"

    AuditRepository(db).log(user_id=None, action="test.action", detail="d")
    assert AuditRepository(db).recent()[0]["action"] == "test.action"
