#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.biometry.estimated_fetal_weight import (
    estimate_fetal_weight_hadlock_hc_ac_fl,
    estimate_fetal_weight_safe,
)


def main():
    # Deterministic regression example.
    # HC=320 mm, AC=300 mm, FL=60 mm -> ~2205.972 g.
    result = estimate_fetal_weight_hadlock_hc_ac_fl(
        hc_mm=320.0,
        ac_mm=300.0,
        fl_mm=60.0,
    )

    expected_g = 2205.9720217473264
    assert abs(result["efw_g"] - expected_g) < 1e-6, result

    invalid = estimate_fetal_weight_safe(
        hc_mm=320.0,
        ac_mm=None,
        fl_mm=60.0,
    )
    assert invalid["status"] == "unavailable"
    assert invalid["efw_g"] is None

    print("PASS: Phase-7 EFW regression test")
    print(json.dumps(result, indent=2))
    print("\nInvalid-input behavior:")
    print(json.dumps(invalid, indent=2))


if __name__ == "__main__":
    main()
