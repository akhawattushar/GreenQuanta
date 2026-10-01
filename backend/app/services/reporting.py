"""Report generation.

CSV is produced with the standard library. PDF is produced with ReportLab when
it is installed; when it is not, `pdf_available()` returns False and the API
returns 503 with the install hint rather than a broken file./
"""

from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone
from html import escape
from typing import Any, Iterable, Sequence

try:  # pragma: no cover - exercised by pdf_available()
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    _REPORTLAB = True
except ImportError:  # pragma: no cover
    _REPORTLAB = False


def pdf_available() -> bool:
    return _REPORTLAB


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _flatten(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    if value is None:
        return ""
    return str(value)


def _csv_safe(value: Any) -> str:
    text = _flatten(value)
    # Spreadsheet applications may evaluate cells even when a CSV writer quotes them.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) else text


REPORT_COLUMNS = (
    "source_type", "source_id", "created_at", "vessel", "model_id", "model_run_id",
    "optimization_algorithm", "scenario_type", "raw_prediction", "raw_prediction_unit",
    "conversion_duration_hours", "normalized_fuel_tonnes", "normalized_fuel_unit",
    "cost", "cost_currency", "emissions", "emissions_unit", "fuel_type", "shore_power",
    "fuelcast_inputs", "scenario_overrides", "post_prediction_assumptions", "data_quality_warnings",
)

SNAPSHOT_NOTE = (
    "One supplied environmental snapshot represents this FuelCast calculation; it is not a "
    "time-series route forecast or measured voyage telemetry, and its physical representativeness "
    "has not been established."
)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def normalize_record(kind: str, record: dict) -> tuple[list[dict], list[str]]:
    """Read stored results only; preserve original export columns and add provenance."""
    if not isinstance(record, dict):
        return [], ["Malformed stored record excluded."]
    response = record if kind == "voyage" else record.get("response")
    request = {} if kind == "voyage" else record.get("request")
    if not isinstance(response, dict):
        return [], ["Stored response unavailable; record excluded."]
    if not isinstance(request, dict):
        request = {}
    if kind == "optimization":
        source_rows = response.get("plans")
    elif kind == "scenario":
        source_rows = response.get("rows")
    else:
        source_rows = [response]
    if not isinstance(source_rows, list):
        return [], ["Stored result rows unavailable; record excluded."]

    rows: list[dict] = []
    report_warnings: list[str] = []
    for index, source in enumerate(source_rows):
        if not isinstance(source, dict):
            report_warnings.append(f"Stored result row {index + 1} malformed; excluded.")
            continue
        model_id = source.get("model_id") or response.get("model_id") or request.get("model_id") or "unknown"
        if not isinstance(model_id, str):
            model_id = "unknown"
        raw = source.get("raw_prediction", source.get("fuel_rate"))
        raw_unit = source.get("raw_prediction_unit", source.get("fuel_rate_unit"))
        duration = source.get("conversion_duration_hours", source.get("duration_hours", source.get("eta_hours")))
        fuel = source.get("normalized_voyage_fuel_tonnes", source.get("normalized_fuel_tonnes"))
        fuel_unit = source.get("normalized_voyage_fuel_unit", source.get("normalized_fuel_unit"))
        if fuel is None and kind != "voyage":
            fuel = source.get("voyage_fuel_tonnes", source.get("fuel_tonnes"))
            if fuel is not None:
                fuel_unit = "tonnes"  # The producing services define these stored fields in tonnes.
        if fuel_unit != "tonnes" or not _finite(fuel) or fuel < 0:
            fuel = None
            fuel_unit = None
        if not _finite(raw):
            raw = None
        if not _finite(duration) or duration <= 0:
            duration = None
        if not isinstance(raw_unit, str) or not raw_unit:
            raw_unit = None
        if model_id == "fuelcast_xgboost" and raw_unit != "kg/s":
            raw_unit = None  # A malformed historical row cannot establish the rate's unit.
        warnings = []
        if model_id == "unknown":
            warnings.append("model provenance unavailable")
        if raw is None:
            warnings.append("raw prediction unavailable")
        if not raw_unit:
            warnings.append("prediction unit unavailable")
        if duration is None:
            warnings.append("conversion duration unavailable")
        if fuel is None:
            warnings.append("normalized voyage fuel unavailable")
        snapshot = source.get("fuelcast_input_snapshot") or source.get("fixed_environment_snapshot") or source.get("fuelcast_inputs") or response.get("fixed_environment_snapshot") or record.get("fuelcast_inputs")
        if model_id == "fuelcast_xgboost" and not snapshot:
            warnings.append("environmental snapshot unavailable")
        algorithm = source.get("optimization_algorithm_id") or source.get("algorithm_id")
        if not isinstance(algorithm, str):
            algorithm = None
        if kind == "optimization" and not algorithm:
            label = str(source.get("label", "")).lower()
            algorithm = "quantum" if "qiea" in label else "nsga2" if "nsga" in label else "baseline" if "baseline" in label else None
        cost = source.get("cost_usd")
        currency = "USD" if _finite(cost) else None
        if currency is None:
            cost = source.get("cost_inr")
            currency = "INR" if _finite(cost) else None
        if currency is None:
            cost = None
        emissions = source.get("ghg_tonnes_co2e")
        if not _finite(emissions):
            emissions = None
        metadata = response.get("model_metadata")
        model_run_id = source.get("model_run_id") or response.get("model_run_id")
        if not model_run_id and isinstance(metadata, dict):
            model_run_id = metadata.get("run_id")
        normalized = {
            "source_type": kind, "source_id": record.get("id"),
            "created_at": record.get("created_at") or record.get("departed_at"),
            "vessel": source.get("vessel") or source.get("vessel_type") or request.get("vessel"),
            "model_id": model_id, "model_run_id": model_run_id,
            "optimization_algorithm": algorithm, "scenario_type": source.get("key") or source.get("scenario"),
            "raw_prediction": raw, "raw_prediction_unit": raw_unit,
            "conversion_duration_hours": duration, "normalized_fuel_tonnes": fuel,
            "normalized_fuel_unit": fuel_unit, "cost": cost, "cost_currency": currency,
            "emissions": emissions, "emissions_unit": "tonnes CO2e" if emissions is not None else None,
            "fuel_type": source.get("fuel_type"), "shore_power": source.get("shore_power"),
            "fuelcast_inputs": snapshot, "scenario_overrides": source.get("applied_overrides"),
            "post_prediction_assumptions": source.get("post_prediction_assumptions") or (
                response.get("assumptions") if kind == "optimization" else None
            ),
            "data_quality_warnings": warnings,
        }
        rows.append({**source, **normalized})
    if any(row["model_id"] == "fuelcast_xgboost" for row in rows):
        report_warnings.append(SNAPSHOT_NOTE)
    if len({row["model_id"] for row in rows}) > 1:
        report_warnings.append("Different prediction models or legacy assumptions appear in this report; operational outputs do not establish model accuracy.")
    if len({row["raw_prediction_unit"] for row in rows}) > 1:
        report_warnings.append("Mixed or unavailable raw prediction units; raw rates cannot be aggregated or directly compared.")
    if len({row["conversion_duration_hours"] for row in rows}) > 1:
        report_warnings.append("Different or unavailable voyage durations limit direct comparison.")
    snapshots = {repr(row["fuelcast_inputs"]) for row in rows if row["fuelcast_inputs"] is not None}
    if len(snapshots) > 1:
        report_warnings.append("Different environmental snapshots limit direct comparison.")
    return rows, report_warnings


def rows_to_csv(rows: Sequence[dict], columns: Iterable[str] | None = None) -> str:
    """Render dict rows as CSV text."""
    if not rows:
        return ""
    fields = list(columns) if columns else list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writerow({f: _csv_safe(f) for f in fields})
    for row in rows:
        writer.writerow({f: _csv_safe(row.get(f)) for f in fields})
    return buffer.getvalue()


def build_csv_report(*, title: str, rows: Sequence[dict], notes: Sequence[str] = (), columns: Iterable[str] | None = None) -> bytes:
    """CSV with a short provenance preamble so the file is self-describing."""
    header = io.StringIO()
    writer = csv.writer(header, lineterminator="\n")
    writer.writerow(["# GreenQuanta QuantaFleet export"])
    writer.writerow([f"# {title}"])
    writer.writerow([f"# Generated: {_stamp()}"])
    for note in notes:
        writer.writerow([f"# {note}"])
    writer.writerow([])
    return (header.getvalue() + rows_to_csv(rows, columns=columns)).encode("utf-8")


def report_columns(rows: Sequence[dict]) -> list[str]:
    """Stable provenance first, followed by retained producer-specific fields."""
    extras = sorted({key for row in rows for key in row} - set(REPORT_COLUMNS))
    return list(REPORT_COLUMNS) + extras


def build_pdf_report(
    *,
    title: str,
    subtitle: str = "",
    sections: Sequence[tuple] = (),
    notes: Sequence[str] = (),
) -> bytes:
    """`sections` is a sequence of (heading, rows) where rows is a list of dicts."""
    if not _REPORTLAB:
        raise RuntimeError(
            "PDF export needs ReportLab. Install it with `pip install reportlab`, "
            "or use the CSV export instead."
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=title,
        author="GreenQuanta QuantaFleet",
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("gqTitle", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#0E1B33"))
    h2 = ParagraphStyle(
        "gqHeading", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#0B7175"), spaceBefore=10
    )
    small = ParagraphStyle("gqSmall", parent=styles["BodyText"], fontSize=8, textColor=colors.HexColor("#64748B"))

    story: list = [Paragraph(escape(title), h1)]
    if subtitle:
        story.append(Paragraph(escape(subtitle), styles["BodyText"]))
    story.append(Paragraph(f"Generated {_stamp()}", small))
    story.append(Spacer(1, 6 * mm))

    for heading, rows in sections:
        story.append(Paragraph(escape(heading), h2))
        if not rows:
            story.append(Paragraph("No rows.", small))
            continue
        fields = list(dict.fromkeys(k for r in rows for k in r))
        data = [fields] + [[_flatten(r.get(f)) for f in fields] for r in rows]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F5F6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0A5B5F")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 4 * mm))

    if notes:
        story.append(Paragraph("Notes and assumptions", h2))
        for note in notes:
            story.append(Paragraph(f"• {escape(str(note))}", small))

    doc.build(story)
    return buffer.getvalue()
