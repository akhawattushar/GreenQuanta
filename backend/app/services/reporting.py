"""Report generation.

CSV is produced with the standard library. PDF is produced with ReportLab when
it is installed; when it is not, `pdf_available()` returns False and the API
returns 503 with the install hint rather than a broken file.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
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


def rows_to_csv(rows: Sequence[dict], columns: Iterable[str] | None = None) -> str:
    """Render dict rows as CSV text."""
    if not rows:
        return ""
    fields = list(columns) if columns else list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({f: _flatten(row.get(f)) for f in fields})
    return buffer.getvalue()


def build_csv_report(*, title: str, rows: Sequence[dict], notes: Sequence[str] = ()) -> bytes:
    """CSV with a short provenance preamble so the file is self-describing."""
    header = io.StringIO()
    writer = csv.writer(header, lineterminator="\n")
    writer.writerow(["# GreenQuanta QuantaFleet export"])
    writer.writerow([f"# {title}"])
    writer.writerow([f"# Generated: {_stamp()}"])
    for note in notes:
        writer.writerow([f"# {note}"])
    writer.writerow([])
    return (header.getvalue() + rows_to_csv(rows)).encode("utf-8")


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

    story: list = [Paragraph(title, h1)]
    if subtitle:
        story.append(Paragraph(subtitle, styles["BodyText"]))
    story.append(Paragraph(f"Generated {_stamp()}", small))
    story.append(Spacer(1, 6 * mm))

    for heading, rows in sections:
        story.append(Paragraph(heading, h2))
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
            story.append(Paragraph(f"• {note}", small))

    doc.build(story)
    return buffer.getvalue()
