from __future__ import annotations

from typing import Any, Dict, Optional


def normalize_ga_dict(ga: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize GA output so Streamlit, PDFs, JSON, and CSV always see the same keys."""
    if not isinstance(ga, dict) or not ga:
        return {
            "ga_weeks_decimal": None,
            "ga_weeks": None,
            "ga_days": None,
            "ga_display": "N/A",
            "trimester": "N/A",
            "method": "unavailable",
            "source_note": "GA estimate unavailable.",
        }

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
            total_days = float(decimal_weeks) * 7.0
            if total_days < 98:
                trimester = "First trimester (<14w)"
            elif total_days < 196:
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


def build_payload(
    input_meta: Dict[str, Any],
    preprocessing_meta: Dict[str, Any],
    input_gate,
    post_gate,
    seg_meta: Dict[str, Any],
    post_meta: Dict[str, Any],
    ellipse: Optional[Dict[str, Any]],
    measurements: Dict[str, Any],
    ga: Optional[Dict[str, Any]],
    quality: Dict[str, Any],
    confidence: Dict[str, Any],
    decision: Dict[str, Any],
    artifacts: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
    audit: Optional[Dict[str, Any]] = None,
    report_previews: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonical report-ready payload used by Streamlit, API, and PDFs."""
    measurements = dict(measurements or {})
    incoming_ga = ga or measurements.get("ga_estimate")
    measurements["ga_estimate"] = normalize_ga_dict(incoming_ga)
    measurements.setdefault(
        "ga_note",
        "GA is estimated from HC only when HC in millimeters is available.",
    )

    return {
        "schema_version": "1.1.0",
        "input_info": input_meta or {},
        "preprocessing": preprocessing_meta or {},
        "validity_gate": {
            "input_passed": bool(getattr(input_gate, "passed", False)),
            "input_reason_codes": list(getattr(input_gate, "reason_codes", []) or []),
            "input_details": getattr(input_gate, "details", {}) or {},
            "post_segmentation_passed": bool(getattr(post_gate, "passed", False)),
            "post_reason_codes": list(getattr(post_gate, "reason_codes", []) or []),
            "post_details": getattr(post_gate, "details", {}) or {},
        },
        "segmentation": {**(seg_meta or {}), "postprocess": post_meta or {}},
        "ellipse_parameters": ellipse,
        "measurements": measurements,
        "quality": quality or {},
        "confidence": confidence or {},
        "decision": decision or {},
        "model_info": model_info or {},
        "audit": audit or {},
        "report_previews": report_previews or {},
        "artifacts": artifacts or {},
    }


def build_report_previews(payload_like: Dict[str, Any]) -> Dict[str, Any]:
    """Small patient/doctor/technical blocks for dashboard preview and PDF assembly."""
    measurements = dict(payload_like.get("measurements", {}) or {})
    ga = normalize_ga_dict(measurements.get("ga_estimate"))
    measurements["ga_estimate"] = ga

    quality = payload_like.get("quality", {}) or {}
    confidence = payload_like.get("confidence", {}) or {}
    decision = payload_like.get("decision", {}) or {}
    ellipse = payload_like.get("ellipse_parameters", {}) or {}

    disclaimer = "AI result must be reviewed by a licensed doctor."
    patient = {
        "title": "Patient Summary",
        "summary": (
            "The fetal head was analyzed from the uploaded ultrasound scan using "
            "an AI-assisted measurement workflow. Gestational age is estimated from HC only."
        ),
        "hc": measurements.get("hc_mm") or measurements.get("hc_px"),
        "hc_unit": "mm" if measurements.get("hc_mm") is not None else "px",
        "estimated_ga": ga.get("ga_display", "N/A"),
        "trimester": ga.get("trimester", "N/A"),
        "ga_method": ga.get("method", "N/A"),
        "quality_label": quality.get("quality_label"),
        "decision": decision.get("decision"),
        "disclaimer": disclaimer,
    }
    doctor = {
        "title": "Doctor Review",
        "measurements": measurements,
        "ellipse_parameters": ellipse,
        "quality": quality,
        "confidence": confidence,
        "decision": decision,
        "review_note": "Review the original scan, mask, ellipse overlay, image-quality flags, confidence score, and HC-derived GA before clinical use.",
        "disclaimer": disclaimer,
    }
    technical = {
        "title": "Technical Audit",
        "model_info": payload_like.get("model_info", {}),
        "audit": payload_like.get("audit", {}),
        "preprocessing": payload_like.get("preprocessing", {}),
        "validity_gate": payload_like.get("validity_gate", {}),
        "ga_calculation": ga,
    }
    return {"patient": patient, "doctor": doctor, "technical": technical}
