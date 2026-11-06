from __future__ import annotations

import math
from typing import Any, Dict, Optional

FORMULA_ID = "hadlock_hc_ac_fl"
FORMULA_NAME = "Hadlock HC-AC-FL"
FORMULA_TEXT = (
    "log10(EFW_g) = 1.326 - 0.00326*(AC_cm*FL_cm) "
    "+ 0.0107*HC_cm + 0.0438*AC_cm + 0.158*FL_cm"
)


def _require_positive_finite(name: str, value: float) -> float:
    if value is None:
        raise ValueError(f"{name} is required.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}.")
    if value <= 0:
        raise ValueError(f"{name} must be > 0 mm, got {value}.")
    return value


def estimate_fetal_weight_hadlock_hc_ac_fl(
    hc_mm: float,
    ac_mm: float,
    fl_mm: float,
) -> Dict[str, Any]:
    """
    Estimate fetal weight from head circumference (HC), abdominal circumference
    (AC), and femur length (FL).

    Inputs
    ------
    hc_mm, ac_mm, fl_mm:
        Measurements in millimeters.

    Internally the Hadlock HC-AC-FL equation uses centimeters and returns grams.

    Notes
    -----
    This is a deterministic formula, not a learned model and not a clinical
    diagnosis. The caller should only invoke it when its upstream measurements
    have passed the application's validity/quality checks.
    """
    hc_mm = _require_positive_finite("hc_mm", hc_mm)
    ac_mm = _require_positive_finite("ac_mm", ac_mm)
    fl_mm = _require_positive_finite("fl_mm", fl_mm)

    hc_cm = hc_mm / 10.0
    ac_cm = ac_mm / 10.0
    fl_cm = fl_mm / 10.0

    log10_efw = (
        1.326
        - 0.00326 * ac_cm * fl_cm
        + 0.0107 * hc_cm
        + 0.0438 * ac_cm
        + 0.158 * fl_cm
    )
    efw_g = 10.0 ** log10_efw

    return {
        "status": "calculated",
        "formula_id": FORMULA_ID,
        "formula_name": FORMULA_NAME,
        "formula": FORMULA_TEXT,
        "inputs_mm": {
            "hc_mm": hc_mm,
            "ac_mm": ac_mm,
            "fl_mm": fl_mm,
        },
        "inputs_cm": {
            "hc_cm": hc_cm,
            "ac_cm": ac_cm,
            "fl_cm": fl_cm,
        },
        "log10_efw": float(log10_efw),
        "efw_g": float(efw_g),
        "efw_kg": float(efw_g / 1000.0),
    }


def estimate_fetal_weight_safe(
    hc_mm: Optional[float],
    ac_mm: Optional[float],
    fl_mm: Optional[float],
) -> Dict[str, Any]:
    """
    App-safe wrapper. Returns an unavailable payload rather than raising when a
    required input is absent/invalid.
    """
    try:
        return estimate_fetal_weight_hadlock_hc_ac_fl(
            hc_mm=hc_mm,
            ac_mm=ac_mm,
            fl_mm=fl_mm,
        )
    except (TypeError, ValueError) as exc:
        return {
            "status": "unavailable",
            "formula_id": FORMULA_ID,
            "formula_name": FORMULA_NAME,
            "reason": str(exc),
            "efw_g": None,
            "efw_kg": None,
        }
