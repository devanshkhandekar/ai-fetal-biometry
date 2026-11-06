from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class GAEstimate:
    ga_weeks_decimal: float
    ga_weeks: int
    ga_days: int
    ga_display: str
    trimester: str
    method: str
    source_note: str


def classify_trimester(ga_weeks_decimal: float) -> str:
    """Classify trimester from GA in decimal weeks."""
    total_days = float(ga_weeks_decimal) * 7.0
    if total_days < 98:
        return "First trimester (<14w)"
    if total_days < 196:
        return "Second trimester (14w-27w6d)"
    return "Third trimester (>=28w)"


def estimate_ga_from_hc_hadlock(hc_mm: Optional[float]) -> Optional[GAEstimate]:
    """Estimate gestational age from HC in mm using an HC-only Hadlock-style polynomial.

    Formula uses HC in cm:
        GA_weeks = 8.96 + 0.540*HCcm - 0.0040*HCcm^2 + 0.000399*HCcm^3

    This is an estimate from head circumference only. It is not a substitute for
    clinical dating or multi-parameter fetal biometry review.
    """
    if hc_mm is None:
        return None
    try:
        hc_mm = float(hc_mm)
    except (TypeError, ValueError):
        return None
    if hc_mm <= 0:
        return None

    hc_cm = hc_mm / 10.0
    ga = 8.96 + 0.540 * hc_cm - 0.0040 * (hc_cm ** 2) + 0.000399 * (hc_cm ** 3)

    # Practical display bounds for obstetric GA. This also prevents absurd values
    # when spacing is missing/wrong or the mask is invalid.
    ga = max(10.0, min(float(ga), 42.0))

    weeks = int(ga)
    days = int(round((ga - weeks) * 7.0))
    if days == 7:
        weeks += 1
        days = 0

    return GAEstimate(
        ga_weeks_decimal=round(ga, 3),
        ga_weeks=weeks,
        ga_days=days,
        ga_display=f"{weeks}w {days}d",
        trimester=classify_trimester(ga),
        method="HC_TO_GA_HADLOCK_POLYNOMIAL",
        source_note=(
            "Gestational age is estimated from head circumference only using "
            "a Hadlock-style polynomial. AI result must be reviewed by a licensed doctor."
        ),
    )


def ga_estimate_to_dict(ga: Optional[GAEstimate]) -> Optional[Dict[str, Any]]:
    if ga is None:
        return None
    return asdict(ga)
