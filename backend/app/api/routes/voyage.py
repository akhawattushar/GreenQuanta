"""Voyage monitoring endpoints. Voyages are created by users and persisted."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.api.deps import CurrentUser, DbDep, WriteUser
from app.db.database import AuditRepository
from app.schemas.models import FleetSummaryResponse, VoyageCreateRequest, VoyageListResponse
from app.services.model_registry import MODEL_UNAVAILABLE_MESSAGE, FeatureValidationError, ModelUnavailableError
from app.services.voyage import (
    DATA_SOURCE, VoyageCalculationError, create_voyage, delete_voyage, fleet_summary, get_voyage, list_voyages,
)

router = APIRouter(prefix="/voyage", tags=["voyage"])
logger = logging.getLogger(__name__)


@router.get("/active", response_model=VoyageListResponse, summary="Voyages in progress")
def active(user: CurrentUser, db: DbDep) -> VoyageListResponse:
    # Normal users only ever see voyages they created; admins see the whole fleet.
    scope = None if user["role"] == "admin" else user["id"]
    return VoyageListResponse(voyages=list_voyages(db, user_id=scope), data_source=DATA_SOURCE, live_telemetry=False)


@router.get("/summary", response_model=FleetSummaryResponse, summary="Fleet-level aggregates")
def summary(user: CurrentUser, db: DbDep) -> FleetSummaryResponse:
    scope = None if user["role"] == "admin" else user["id"]
    return FleetSummaryResponse(**fleet_summary(db, user_id=scope))


@router.post("", status_code=status.HTTP_201_CREATED, summary="Record a new voyage")
def create(payload: VoyageCreateRequest, user: WriteUser, db: DbDep) -> dict:
    try:
        voyage = create_voyage(db, user_id=user["id"], user_name=user["name"], payload=payload.model_dump())
    except (FeatureValidationError, VoyageCalculationError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        logger.exception("Voyage prediction model unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=MODEL_UNAVAILABLE_MESSAGE,
        ) from exc
    AuditRepository(db).log(user_id=user["id"], action="voyage.create", detail=voyage["id"])
    return voyage


@router.delete("/{voyage_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Remove a voyage record")
def remove(voyage_id: str, user: WriteUser, db: DbDep) -> None:
    try:
        deleted = delete_voyage(db, voyage_id, user_id=user["id"], is_admin=user["role"] == "admin")
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voyage not found.")
    AuditRepository(db).log(user_id=user["id"], action="voyage.delete", detail=voyage_id)


@router.get("/{voyage_id}", summary="One voyage")
def detail(voyage_id: str, user: CurrentUser, db: DbDep) -> dict:
    voyage = get_voyage(db, voyage_id)
    if voyage is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voyage not found.")
    if user["role"] != "admin" and voyage.get("user_id") != user["id"]:
        # Same response as "not found" so this endpoint cannot be used to probe
        # which voyage ids exist on other accounts.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voyage not found.")
    return voyage
