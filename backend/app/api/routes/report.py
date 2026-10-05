"""CSV / PDF export of stored runs."""

from __future__ import annotations

from datetime import datetime, timezone
import logging

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from app.api.deps import CurrentUser, DbDep
from app.db.database import RunRepository, VoyageRepository
from app.services import reporting

router = APIRouter(prefix="/report", tags=["report"])
logger = logging.getLogger(__name__)

KINDS = ("optimization", "scenario", "prediction", "voyage")


def _load(db, kind: str, run_id: str | None, user_id: str) -> dict:
    if kind not in KINDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown report kind {kind!r}. Expected one of: {', '.join(KINDS)}.",
        )
    if kind == "voyage":
        repo = VoyageRepository(db)
        record = repo.get(run_id) if run_id else next(iter(repo.list(user_id)), None)
        if record is not None and record.get("user_id") != user_id:
            record = None
    else:
        repo = RunRepository(db, kind)
        record = repo.get(run_id, user_id) if run_id else repo.latest(user_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No {kind} run found to export. Run one first.",
        )
    return record


def _history_date(value) -> datetime | None:
    """Return a comparable UTC timestamp for supported stored date shapes."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rows_and_notes(kind: str, record: dict) -> tuple:
    response = record if kind == "voyage" else record.get("response")
    response = response if isinstance(response, dict) else {}
    if kind == "optimization":
        notes = [response.get("disclaimer", "")]
        assumptions = response.get("assumptions", {})
    elif kind == "scenario":
        notes = [response.get("note", "")]
        assumptions = response.get("assumptions", {})
    elif kind == "voyage":
        notes = []
        assumptions = {}
    else:
        notes = [response.get("note", "")]
        assumptions = {}
    rows, quality_notes = reporting.normalize_record(kind, record)
    notes.extend(quality_notes)
    if kind == "prediction" and rows:
        features = response.get("features_used")
        if isinstance(features, dict):
            rows[0].update(features)
        rows[0]["unit"] = response.get("fuel_rate_unit")
    notes = [n for n in notes if n]
    notes.append(f"Run id: {record.get('id', 'unknown')} · generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    stored_rows = response.get("plans") if kind == "optimization" else response.get("rows") if kind == "scenario" else None
    excluded = max(0, len(stored_rows) - len(rows)) if isinstance(stored_rows, list) else 0
    notes.append(f"Stored result rows included: {len(rows)}; excluded: {excluded}.")
    if isinstance(assumptions, dict) and assumptions:
        notes.append(
            "Cost and emission figures use configured conversion factors, not learned values: "
            + "; ".join(f"{k}={v}" for k, v in assumptions.items() if not isinstance(v, dict))
        )
    return rows, notes


def _filtered(rows: list[dict], *, model_id: str | None, optimization_algorithm: str | None,
              source_type: str | None) -> list[dict]:
    return [row for row in rows if (model_id is None or row["model_id"] == model_id)
            and (optimization_algorithm is None or row["optimization_algorithm"] == optimization_algorithm)
            and (source_type is None or row["source_type"] == source_type)]


@router.get("/csv", summary="Export a run as CSV")
def export_csv(
    user: CurrentUser,
    db: DbDep,
    kind: str = Query("optimization", description="optimization | scenario | prediction | voyage"),
    run_id: str | None = Query(None, description="Defaults to the most recent run."),
    model_id: str | None = Query(None, description="Filter stored result rows by prediction model."),
    optimization_algorithm: str | None = Query(None, description="Filter separately by optimizer algorithm."),
    source_type: str | None = Query(None, description="Filter by stored source type."),
) -> Response:
    record = _load(db, kind, run_id, user["id"])
    rows, notes = _rows_and_notes(kind, record)
    rows = _filtered(rows, model_id=model_id, optimization_algorithm=optimization_algorithm, source_type=source_type)
    payload = reporting.build_csv_report(title=f"{kind.title()} run {record['id']}", rows=rows, notes=notes,
                                         columns=reporting.report_columns(rows))
    return Response(
        content=payload,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="quantafleet-{kind}-{record["id"]}.csv"'},
    )


@router.get("/pdf", summary="Export a run as PDF")
def export_pdf(
    user: CurrentUser,
    db: DbDep,
    kind: str = Query("optimization", description="optimization | scenario | prediction | voyage"),
    run_id: str | None = Query(None, description="Defaults to the most recent run."),
    model_id: str | None = Query(None, description="Filter stored result rows by prediction model."),
    optimization_algorithm: str | None = Query(None, description="Filter separately by optimizer algorithm."),
    source_type: str | None = Query(None, description="Filter by stored source type."),
) -> Response:
    if not reporting.pdf_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF export is unavailable because ReportLab is not installed. Use /report/csv instead.",
        )
    record = _load(db, kind, run_id, user["id"])
    rows, notes = _rows_and_notes(kind, record)
    rows = _filtered(rows, model_id=model_id, optimization_algorithm=optimization_algorithm, source_type=source_type)
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
def history(user: CurrentUser, db: DbDep, model_id: str | None = None,
            optimization_algorithm: str | None = None, source_type: str | None = None) -> dict:
    items = []
    warnings = []

    def add_record(kind: str, record: dict) -> None:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str) or not record["id"]:
            logger.warning("Skipping report history record without a valid ID")
            warnings.append("A malformed historical record was excluded.")
            return
        created = _history_date(record.get("departed_at") if kind == "voyage" else record.get("created_at"))
        if created is None:
            logger.warning("Skipping report history record with invalid date: %s", record["id"])
            warnings.append("A historical record with an invalid date was excluded.")
            return
        try:
            rows, quality_notes = reporting.normalize_record(kind, record)
            matching = _filtered(rows, model_id=model_id, optimization_algorithm=optimization_algorithm,
                                 source_type=source_type)
        except Exception:
            logger.exception("Skipping malformed report history record: %s", record["id"])
            warnings.append("A malformed historical record was excluded.")
            return
        if any("malformed" in note or "unavailable" in note for note in quality_notes):
            warnings.append("Some stored result rows were unavailable or malformed.")
        if matching:
            items.append({"id": record["id"], "kind": kind,
                          "created_at": created.isoformat(timespec="seconds"),
                          "model_ids": sorted({row["model_id"] for row in matching}),
                          "optimization_algorithms": sorted({row["optimization_algorithm"] for row in matching
                                                             if row["optimization_algorithm"]}),
                          "_sort_at": created})

    for kind in KINDS:
        if kind == "voyage":
            continue
        for record in RunRepository(db, kind).history(user["id"], limit=10):
            add_record(kind, record)
    for record in VoyageRepository(db).list(user["id"]):
        add_record("voyage", record)
    items.sort(key=lambda item: (-item["_sort_at"].timestamp(), item["kind"], item["id"]))
    for item in items:
        item.pop("_sort_at")
    result = {"items": items, "pdf_available": reporting.pdf_available()}
    if warnings:
        result["warnings"] = list(dict.fromkeys(warnings))
    return result
