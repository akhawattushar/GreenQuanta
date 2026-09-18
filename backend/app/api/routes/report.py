"""CSV / PDF export of stored runs."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from app.api.deps import CurrentUser, DbDep
from app.db.database import RunRepository
from app.services import reporting

router = APIRouter(prefix="/report", tags=["report"])

KINDS = ("optimization", "scenario", "prediction")


def _load(db, kind: str, run_id: str | None, user_id: str) -> dict:
    if kind not in KINDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown report kind {kind!r}. Expected one of: {', '.join(KINDS)}.",
        )
    repo = RunRepository(db, kind)
    record = repo.get(run_id, user_id) if run_id else repo.latest(user_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {kind} run found to export. Run one first.",
        )
    return record


def _rows_and_notes(kind: str, record: dict) -> tuple:
    response = record["response"]
    if kind == "optimization":
        rows = response.get("plans", [])
        notes = [response.get("disclaimer", "")]
        assumptions = response.get("assumptions", {})
    elif kind == "scenario":
        rows = response.get("rows", [])
        notes = [response.get("note", "")]
        assumptions = response.get("assumptions", {})
    else:
        rows = [
            {
                "fuel_rate": response.get("fuel_rate"),
                "unit": response.get("fuel_rate_unit"),
                "unit_verified": response.get("unit_verified"),
                "voyage_fuel_tonnes": response.get("voyage_fuel_tonnes"),
                "duration_hours": response.get("duration_hours"),
                **(response.get("features_used") or {}),
            }
        ]
        notes = [response.get("note", "")]
        assumptions = {}

    notes = [n for n in notes if n]
    notes.append(f"Run id: {record['id']} · generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if assumptions:
        notes.append(
            "Cost and emission figures use configured conversion factors, not learned values: "
            + "; ".join(f"{k}={v}" for k, v in assumptions.items() if not isinstance(v, dict))
        )
    return rows, notes


@router.get("/csv", summary="Export a run as CSV")
def export_csv(
    user: CurrentUser,
    db: DbDep,
    kind: str = Query("optimization", description="optimization | scenario | prediction"),
    run_id: str | None = Query(None, description="Defaults to the most recent run."),
) -> Response:
    record = _load(db, kind, run_id, user["id"])
    rows, notes = _rows_and_notes(kind, record)
    payload = reporting.build_csv_report(title=f"{kind.title()} run {record['id']}", rows=rows, notes=notes)
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="quantafleet-{kind}-{record["id"]}.csv"'},
    )


@router.get("/pdf", summary="Export a run as PDF")
def export_pdf(
    user: CurrentUser,
    db: DbDep,
    kind: str = Query("optimization", description="optimization | scenario | prediction"),
    run_id: str | None = Query(None, description="Defaults to the most recent run."),
) -> Response:
    if not reporting.pdf_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF export is unavailable because ReportLab is not installed. Use /report/csv instead.",
        )
    record = _load(db, kind, run_id, user["id"])
    rows, notes = _rows_and_notes(kind, record)
    payload = reporting.build_pdf_report(
        title="GreenQuanta QuantaFleet report",
        subtitle=f"{kind.title()} run {record['id']}",
        sections=[(kind.title(), rows)],
        notes=notes,
    )
    return Response(
        content=payload,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="quantafleet-{kind}-{record["id"]}.pdf"'},
    )


@router.get("/history", summary="Exportable runs for the current user")
def history(user: CurrentUser, db: DbDep) -> dict:
    items = []
    for kind in KINDS:
        for record in RunRepository(db, kind).history(user["id"], limit=10):
            items.append({"id": record["id"], "kind": kind, "created_at": record["created_at"]})
    items.sort(key=lambda r: r["created_at"], reverse=True)
    return {"items": items, "pdf_available": reporting.pdf_available()}
