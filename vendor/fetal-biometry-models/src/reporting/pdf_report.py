from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}{unit}"
    return f"{value}{unit}"


def _normalize_ga_dict(ga: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize GA payloads from either old chart lookup or HC-polynomial lookup."""
    if not isinstance(ga, dict) or not ga:
        return {}

    out = dict(ga)
    display = (
        out.get("ga_display")
        or out.get("ga_text")
        or out.get("estimate_text")
        or out.get("gestational_age")
    )

    weeks = out.get("ga_weeks")
    days = out.get("ga_days")
    decimal_weeks = out.get("ga_weeks_decimal")

    if display is None and weeks is not None:
        try:
            display = f"{int(weeks)}w {int(days or 0)}d"
        except Exception:
            display = str(weeks)

    if decimal_weeks is None and weeks is not None:
        try:
            decimal_weeks = float(weeks) + (float(days or 0) / 7.0)
        except Exception:
            decimal_weeks = None

    trimester = out.get("trimester")
    if trimester is None and decimal_weeks is not None:
        try:
            ga_days_total = float(decimal_weeks) * 7.0
            if ga_days_total < 98:
                trimester = "First trimester (<14w)"
            elif ga_days_total < 196:
                trimester = "Second trimester (14w-27w6d)"
            else:
                trimester = "Third trimester (>=28w)"
        except Exception:
            trimester = None

    out["ga_display"] = display or "N/A"
    out["ga_weeks_decimal"] = decimal_weeks
    out["trimester"] = trimester or "N/A"
    out.setdefault("method", "HC_TO_GA_ESTIMATION")
    out.setdefault(
        "source_note",
        "Gestational age is estimated from head circumference only and must be reviewed by a licensed doctor.",
    )
    return out


def _measurement_value(m: Dict[str, Any], mm_key: str, px_key: str, mm_unit: str = "mm", px_unit: str = "px") -> str:
    if m.get(mm_key) is not None:
        return _fmt(m.get(mm_key), 2, f" {mm_unit}")
    return _fmt(m.get(px_key), 2, f" {px_unit}")


def _table_style(colors):
    from reportlab.platypus import TableStyle

    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ])


def build_simple_pdf_report(payload: Dict[str, Any], report_type: str = "doctor") -> bytes:
    """Generate a simple MVP PDF in memory.

    Supported report_type values:
      - patient
      - doctor
      - technical

    This version includes HC-derived GA fields in the measurement table and
    adds a clear clinical-review disclaimer.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image as RLImage
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PDF export requires reportlab. Install with: pip install reportlab") from exc

    report_type = (report_type or "doctor").lower().strip()
    if report_type not in {"patient", "doctor", "technical"}:
        report_type = "doctor"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title = "Fetal Head Biometry MVP Report"
    if report_type == "patient":
        title = "Patient Summary - Fetal Head Biometry"
    elif report_type == "technical":
        title = "Technical Audit - Fetal Head Biometry"
    else:
        title = "Doctor Review - Fetal Head Biometry"

    m = payload.get("measurements", {}) or {}
    q = payload.get("quality", {}) or {}
    c = payload.get("confidence", {}) or {}
    d = payload.get("decision", {}) or {}
    e = payload.get("ellipse_parameters", {}) or {}
    model = payload.get("model_info", {}) or {}
    audit = payload.get("audit", {}) or {}
    inp = payload.get("input_info", {}) or {}
    preprocessing = payload.get("preprocessing", {}) or {}
    ga = _normalize_ga_dict(m.get("ga_estimate"))

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("AI result must be reviewed by a licensed doctor.", styles["Normal"]))
    story.append(Paragraph("This MVP output is not a standalone diagnosis.", styles["Italic"]))
    story.append(Spacer(1, 0.2 * inch))

    if report_type == "patient":
        story.append(Paragraph("Patient-Friendly Summary", styles["Heading2"]))
        story.append(Paragraph(
            "The uploaded fetal head ultrasound image was processed to estimate fetal head measurements. "
            "The gestational age shown below is estimated from head circumference only.",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.15 * inch))

    rows = [
        ["Field", "Value"],
        ["Decision", str(d.get("decision", "N/A")).upper()],
        ["HC", _measurement_value(m, "hc_mm", "hc_px", "mm", "px")],
        ["BPD", _measurement_value(m, "bpd_mm", "bpd_px", "mm", "px")],
        ["OFD", _measurement_value(m, "ofd_mm", "ofd_px", "mm", "px")],
        ["Head Area", _measurement_value(m, "head_area_mm2", "head_area_px2", "mm²", "px²")],
        ["Estimated GA", ga.get("ga_display", "N/A")],
        ["GA Decimal Weeks", _fmt(ga.get("ga_weeks_decimal"), 2, " weeks")],
        ["Trimester", ga.get("trimester", "N/A")],
        ["GA Method", ga.get("method", "N/A")],
        ["Quality", f"{_fmt(q.get('quality_score'))} / 100 ({q.get('quality_label', 'N/A')})"],
        ["Confidence", f"{_fmt(c.get('confidence_score'))} / 100 ({c.get('confidence_band', 'N/A')})"],
    ]
    table = Table(rows, colWidths=[2.2 * inch, 4.6 * inch])
    table.setStyle(_table_style(colors))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "GA note: Gestational age is estimated from head circumference only and must be reviewed by a licensed doctor.",
        styles["Italic"],
    ))
    story.append(Spacer(1, 0.2 * inch))

    overlay_path = (payload.get("artifacts", {}) or {}).get("overlay_path")
    if overlay_path and Path(overlay_path).exists():
        story.append(Paragraph("AI Overlay", styles["Heading2"]))
        try:
            story.append(RLImage(str(overlay_path), width=5.8 * inch, height=3.7 * inch))
            story.append(Spacer(1, 0.2 * inch))
        except Exception:
            story.append(Paragraph("Overlay image could not be embedded in the PDF.", styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))

    if report_type in {"doctor", "technical"}:
        story.append(Paragraph("Ellipse Geometry", styles["Heading2"]))
        erows = [["Field", "Value"]] + [[k, str(v)] for k, v in (e or {}).items()]
        etable = Table(erows, colWidths=[2.2 * inch, 4.6 * inch])
        etable.setStyle(_table_style(colors))
        story.append(etable)
        story.append(Spacer(1, 0.2 * inch))

        meta = inp.get("metadata_summary") or {}
        if meta:
            story.append(Paragraph("DICOM Metadata Summary", styles["Heading2"]))
            mrows = [["Field", "Value"]] + [[str(k), str(v)] for k, v in meta.items()]
            mtable = Table(mrows, colWidths=[2.2 * inch, 4.6 * inch])
            mtable.setStyle(_table_style(colors))
            story.append(mtable)
            story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Doctor Review", styles["Heading2"]))
        story.append(Paragraph("Doctor signature / approval: ______________________________", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

    if report_type == "technical":
        story.append(Paragraph("Technical Audit Metadata", styles["Heading2"]))
        arows = [
            ["Field", "Value"],
            ["Model variant", model.get("model_variant")],
            ["Checkpoint", model.get("checkpoint_name")],
            ["Device", model.get("device")],
            ["Runtime ms", audit.get("runtime_ms")],
            ["Payload SHA256", audit.get("payload_sha256")],
            ["Input SHA256", inp.get("input_sha256")],
            ["Pipeline status", audit.get("pipeline_status")],
        ]
        atable = Table(arows, colWidths=[2.2 * inch, 4.6 * inch])
        atable.setStyle(_table_style(colors))
        story.append(atable)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Preprocessing Summary", styles["Heading2"]))
        prows = [["Field", "Value"]] + [[str(k), str(v)] for k, v in preprocessing.items()]
        ptable = Table(prows, colWidths=[2.2 * inch, 4.6 * inch])
        ptable.setStyle(_table_style(colors))
        story.append(ptable)

    doc.build(story)
    return buf.getvalue()
