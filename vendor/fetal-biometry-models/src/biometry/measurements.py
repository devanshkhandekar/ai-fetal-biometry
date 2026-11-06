from __future__ import annotations

from typing import Any, Dict, Optional
import math

from src.biometry.ga_lookup import estimate_ga_from_hc


def ramanujan_ellipse_circumference(a: float, b: float) -> float:
    """Ramanujan II approximation for ellipse circumference.

    Args:
        a: semi-major axis in any unit
        b: semi-minor axis in the same unit
    """
    a = float(a)
    b = float(b)
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / (((a + b) ** 2) + 1e-12)
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(max(4.0 - 3.0 * h, 1e-12))))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _empty_measurements(pixel_spacing_mm: Optional[float], note: str, include_ga: bool) -> Dict[str, Any]:
    return {
        "bpd_px": None,
        "ofd_px": None,
        "hc_px": None,
        "head_area_px2": None,
        "major_diameter_px": None,
        "minor_diameter_px": None,
        "semi_major_axis_px": None,
        "semi_minor_axis_px": None,
        "pixel_spacing_mm": pixel_spacing_mm,
        "scale_available": False,
        "bpd_mm": None,
        "ofd_mm": None,
        "hc_mm": None,
        "head_area_mm2": None,
        "ga_estimate": estimate_ga_from_hc(None) if include_ga else None,
        "ga_note": note,
        "measurement_status": "unavailable",
    }


def compute_biometry_from_ellipse(
    ellipse: Dict[str, Any],
    pixel_spacing_mm: Optional[float] = None,
    include_ga: bool = True,
) -> Dict[str, Any]:
    """Compute deterministic fetal head biometry from a fitted ellipse.

    Robust convention:
      - major_diameter_px / minor_diameter_px are full diameters in pixels.
      - semi_major_axis_px / semi_minor_axis_px are half diameters in pixels.
      - OFD = larger full diameter.
      - BPD = smaller full diameter.
      - HC = Ramanujan circumference using semi-axes.
      - GA is calculated only when HC_mm is available.
    """
    if not ellipse:
        return _empty_measurements(pixel_spacing_mm, "GA unavailable because ellipse fitting failed.", include_ga)

    major_diam_px = _safe_float(ellipse.get("major_diameter_px", ellipse.get("major_axis_px")))
    minor_diam_px = _safe_float(ellipse.get("minor_diameter_px", ellipse.get("minor_axis_px")))
    if major_diam_px is None or minor_diam_px is None or major_diam_px <= 0 or minor_diam_px <= 0:
        return _empty_measurements(pixel_spacing_mm, "GA unavailable because ellipse axes are invalid.", include_ga)

    ofd_px = max(major_diam_px, minor_diam_px)
    bpd_px = min(major_diam_px, minor_diam_px)
    a_px = ofd_px / 2.0
    b_px = bpd_px / 2.0
    hc_px = ramanujan_ellipse_circumference(a_px, b_px)
    head_area_px2 = math.pi * a_px * b_px

    ps = _safe_float(pixel_spacing_mm)
    scale_available = ps is not None and ps > 0

    out: Dict[str, Any] = {
        "bpd_px": round(bpd_px, 3),
        "ofd_px": round(ofd_px, 3),
        "hc_px": round(hc_px, 3),
        "head_area_px2": round(head_area_px2, 3),
        "major_diameter_px": round(ofd_px, 3),
        "minor_diameter_px": round(bpd_px, 3),
        "semi_major_axis_px": round(a_px, 3),
        "semi_minor_axis_px": round(b_px, 3),
        "pixel_spacing_mm": ps,
        "scale_available": bool(scale_available),
        "bpd_mm": None,
        "ofd_mm": None,
        "hc_mm": None,
        "head_area_mm2": None,
        "ga_estimate": None,
        "ga_note": "GA requires HC in millimeters. Provide valid DICOM/manual pixel spacing.",
        "measurement_status": "calculated_pixels_only",
        "measurement_method": "ellipse_fit_major_minor_diameters_ramanujan_hc",
    }

    if scale_available:
        bpd_mm = bpd_px * ps
        ofd_mm = ofd_px * ps
        hc_mm = hc_px * ps
        head_area_mm2 = head_area_px2 * (ps ** 2)
        out.update(
            {
                "bpd_mm": round(bpd_mm, 3),
                "ofd_mm": round(ofd_mm, 3),
                "hc_mm": round(hc_mm, 3),
                "head_area_mm2": round(head_area_mm2, 3),
                "ga_estimate": estimate_ga_from_hc(hc_mm) if include_ga else None,
                "ga_note": (
                    "GA estimated from HC only using a Hadlock-style polynomial. "
                    "AI result must be reviewed by a licensed doctor."
                ),
                "measurement_status": "calculated_mm_and_pixels",
            }
        )
    elif include_ga:
        out["ga_estimate"] = estimate_ga_from_hc(None)

    return out
