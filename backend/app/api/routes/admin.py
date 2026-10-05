"""Administrator-only endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import AdminUser, DbDep
from app.core.config import get_settings
from app.db.database import AuditRepository, UserRepository
from app.schemas.models import RoleUpdateRequest
from app.services import model_registry, reporting

router = APIRouter(prefix="/admin", tags=["admin"])
logger = logging.getLogger(__name__)


@router.get("/users", summary="List all accounts")
def users(_: AdminUser, db: DbDep) -> dict:
    return {"users": UserRepository(db).list()}


@router.patch("/users/{user_id}/role", summary="Change an account's role")
def set_role(user_id: str, payload: RoleUpdateRequest, admin: AdminUser, db: DbDep) -> dict:
    repo = UserRepository(db)
    if repo.get(user_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    if user_id == admin["id"] and payload.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot remove your own administrator role.",
        )
    repo.set_role(user_id, payload.role)
    AuditRepository(db).log(user_id=admin["id"], action="admin.set_role", detail=f"{user_id}->{payload.role}")
    return {"id": user_id, "role": payload.role}


@router.get("/model", summary="Model registry status and measured holdout metrics")
def model(_: AdminUser) -> dict:
    info = model_registry.status()
    metrics = None
    if info["loaded"]:
        try:
            metrics = model_registry.evaluate_holdout()
        except model_registry.ModelUnavailableError:
            logger.exception("Admin model holdout evaluation failed")
            metrics = {"available": False, "reason": model_registry.MODEL_UNAVAILABLE_MESSAGE}
        except Exception:
            logger.exception("Admin model holdout evaluation failed")
            metrics = {"available": False, "reason": "Model evaluation could not be completed."}
    return {"registry": info, "metrics": metrics}


@router.post("/model/reload", summary="Re-read the artifacts from disk")
def reload_model(_: AdminUser) -> dict:
    try:
        bundle = model_registry.get_bundle(force_reload=True)
    except model_registry.ModelUnavailableError as exc:
        logger.exception("Admin model reload failed")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=model_registry.MODEL_UNAVAILABLE_MESSAGE) from exc
    return {"reloaded": True, "metadata": bundle.metadata()}


@router.get("/logs", summary="Recent audit entries")
def logs(_: AdminUser, db: DbDep, limit: int = 50) -> dict:
    return {"entries": AuditRepository(db).recent(limit=min(max(limit, 1), 500))}


@router.get("/system", summary="Runtime configuration summary (no secrets)")
def system(_: AdminUser) -> dict:
    settings = get_settings()
    return {
        "environment": settings.app_env,
        "database": f"mongodb:{settings.mongodb_db}",
        "artifacts_dir": str(settings.artifacts_dir),
        "model_path": settings.model_path,
        "preprocessor_path": settings.preprocessor_path,
        "target_unit": settings.target_unit,
        "target_unit_verified": settings.target_unit_verified,
        "pdf_export": reporting.pdf_available(),
        "cors_origins": list(settings.cors_origins),
        "dataset_management": "Not implemented — datasets are produced by the offline training pipeline.",
        "api_keys": "Not implemented — no key issuance in this build.",
    }
