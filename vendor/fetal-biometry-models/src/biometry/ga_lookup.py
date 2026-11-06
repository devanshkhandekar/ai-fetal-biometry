from __future__ import annotations

from typing import Any, Dict, Optional

from src.biometry.ga_hadlock import estimate_ga_from_hc_hadlock, ga_estimate_to_dict


def estimate_ga_from_hc(hc_mm: Optional[float], chart_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a JSON-safe GA estimate from HC in mm.

    This function intentionally accepts an optional chart_cfg argument for backward
    compatibility with older pipeline code, but the MVP method now follows the
    HC-only Hadlock-style polynomial calculation.
    """
    ga = estimate_ga_from_hc_hadlock(hc_mm)
    if ga is None:
        return {
            "ga_weeks_decimal": None,
            "ga_weeks": None,
            "ga_days": None,
            "ga_display": "N/A",
            "trimester": "N/A",
            "method": "unavailable_no_hc_mm",
            "source_note": "GA requires HC in millimeters. Provide valid DICOM/manual pixel spacing.",
        }
    return ga_estimate_to_dict(ga) or {}


def lookup_ga_from_hc(hc_mm: Optional[float], chart_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Alias retained for older imports."""
    return estimate_ga_from_hc(hc_mm, chart_cfg=chart_cfg)
