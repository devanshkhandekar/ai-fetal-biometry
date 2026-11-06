#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.landmarks.biometrynet_subprocess import BiometryNetSubprocess


REPO = ROOT / "external" / "Multicentre-Fetal-Biometry"


def first_valid_image(csv_rel, image_dir_rel):
    csv_path = REPO / csv_rel
    df = pd.read_csv(csv_path)
    junk = [c for c in df.columns if str(c).lower()=="index" or str(c).startswith("Unnamed")]
    if junk:
        df = df.drop(columns=junk)
    image_name = str(df.iloc[0, 0])
    image_path = REPO / image_dir_rel / image_name
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    return image_path


def main():
    bridge = BiometryNetSubprocess()

    abdomen_image = first_valid_image(
        "data/annotations/FP/Abdomen_Test.csv",
        "data/images/FP/Abdomen",
    )
    femur_image = first_valid_image(
        "data/annotations/FP/Femur_Test.csv",
        "data/images/FP/Femur",
    )

    print("Testing abdomen:", abdomen_image.name)
    abdomen = bridge.measure_abdomen(abdomen_image)
    print(json.dumps(abdomen, indent=2))

    assert abdomen["status"] == "valid", abdomen
    assert abdomen["tad_mm"] and abdomen["tad_mm"] > 0
    assert abdomen["apad_mm"] and abdomen["apad_mm"] > 0
    assert abdomen["ac_mm"] and abdomen["ac_mm"] > 0
    assert abdomen["benchmark_preprocessing"] is True

    print("\nTesting femur:", femur_image.name)
    femur = bridge.measure_femur(femur_image)
    print(json.dumps(femur, indent=2))

    assert femur["status"] == "valid", femur
    assert femur["fl_mm"] and femur["fl_mm"] > 0
    assert femur["benchmark_preprocessing"] is True

    print("\nPASS: Phase 4/5 single-upload subprocess bridge works.")


if __name__ == "__main__":
    main()
