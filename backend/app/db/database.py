"""MongoDB Atlas-backed persistence layer.

Everything the API touches goes through the repository classes below. Documents
are stored as native BSON (no JSON string packing needed), keyed on our own
``id`` field so callers never deal with ObjectId.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.core.config import get_settings


class DatabaseError(RuntimeError):
    """Raised when persistence fails."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _strip(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    return doc


class Database:
    """Owns the MongoDB client, database handle and indexes."""

    def __init__(self) -> None:
        settings = get_settings()
        try:
            self._client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
            self._client.admin.command("ping")
        except PyMongoError as exc:
            raise DatabaseError(f"Could not connect to MongoDB: {exc}") from exc
        self.db = self._client[settings.mongodb_db]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.db.users.create_index([("email", ASCENDING)], unique=True)
        self.db.predictions.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.optimization_runs.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.scenario_runs.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.voyages.create_index([("user_id", ASCENDING), ("departed_at", DESCENDING)])
        self.db.audit_log.create_index([("created_at", DESCENDING)])

    def close(self) -> None:
        self._client.close()

    def healthy(self) -> bool:
        try:
            self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False


# --------------------------------------------------------------------------
# repositories
# --------------------------------------------------------------------------
class UserRepository:
    def __init__(self, db: Database) -> None:
        self.col = db.db.users

    def create(self, *, name: str, email: str, password_hash: str, role: str) -> dict[str, Any]:
        record = {
            "id": new_id("usr"),
            "name": name,
            "email": email.lower(),
            "password_hash": password_hash,
            "role": role,
            "status": "active",
            "created_at": _now(),
        }
        try:
            self.col.insert_one(dict(record))
        except DuplicateKeyError as exc:
            raise DatabaseError("An account with that email already exists.") from exc
        return record

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        return _strip(self.col.find_one({"email": email.lower()}))

    def get(self, user_id: str) -> dict[str, Any] | None:
        return _strip(self.col.find_one({"id": user_id}))

    def list(self) -> list[dict[str, Any]]:
        fields = {"_id": 0, "id": 1, "name": 1, "email": 1, "role": 1, "status": 1, "created_at": 1}
        cursor = self.col.find({}, fields).sort("created_at", ASCENDING)
        return list(cursor)

    def count(self) -> int:
        return self.col.count_documents({})

    def set_role(self, user_id: str, role: str) -> bool:
        result = self.col.update_one({"id": user_id}, {"$set": {"role": role}})
        return result.matched_count > 0


class RunRepository:
    """Shared store for prediction / optimization / scenario history."""

    COLLECTIONS = {
        "prediction": "predictions",
        "optimization": "optimization_runs",
        "scenario": "scenario_runs",
    }

    def __init__(self, db: Database, kind: str) -> None:
        if kind not in self.COLLECTIONS:
            raise ValueError(f"Unknown run kind {kind!r}")
        self.col = db.db[self.COLLECTIONS[kind]]
        self.prefix = kind[:3]

    def save(self, *, user_id: str, request: dict, response: dict) -> str:
        run_id = new_id(self.prefix)
        self.col.insert_one(
            {
                "id": run_id,
                "user_id": user_id,
                "request": request,
                "response": response,
                "created_at": _now(),
            }
        )
        return run_id

    def latest(self, user_id: str) -> dict[str, Any] | None:
        return _strip(self.col.find_one({"user_id": user_id}, sort=[("created_at", DESCENDING)]))

    def get(self, run_id: str, user_id: str) -> dict[str, Any] | None:
        return _strip(self.col.find_one({"id": run_id, "user_id": user_id}))

    def history(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.col.find({"user_id": user_id}).sort("created_at", DESCENDING).limit(limit)
        return [_strip(d) for d in cursor]


class VoyageRepository:
    def __init__(self, db: Database) -> None:
        self.col = db.db.voyages

    def create(self, voyage: dict[str, Any]) -> None:
        self.col.insert_one(dict(voyage))

    def list(self, user_id: str | None = None) -> list[dict[str, Any]]:
        query = {"user_id": user_id} if user_id else {}
        cursor = self.col.find(query).sort("departed_at", DESCENDING)
        return [_strip(d) for d in cursor]

    def get(self, voyage_id: str) -> dict[str, Any] | None:
        return _strip(self.col.find_one({"id": voyage_id}))

    def delete(self, voyage_id: str) -> bool:
        result = self.col.delete_one({"id": voyage_id})
        return result.deleted_count > 0

    def count(self) -> int:
        return self.col.count_documents({})


class AuditRepository:
    def __init__(self, db: Database) -> None:
        self.col = db.db.audit_log

    def log(self, *, user_id: str | None, action: str, detail: str = "") -> None:
        self.col.insert_one(
            {
                "id": new_id("log"),
                "user_id": user_id,
                "action": action,
                "detail": detail,
                "created_at": _now(),
            }
        )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.col.find({}).sort("created_at", DESCENDING).limit(limit)
        return [_strip(d) for d in cursor]
