#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plane_detection.dataset_utils import (
    CANONICAL_LABELS,
    add_canonical_columns,
    load_metadata,
    resolve_image_path,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/fetal_planes_db/FETAL_PLANES_DB_data.csv")
    p.add_argument("--images-dir", default="data/fetal_planes_db/Images")
    p.add_argument("--output-dir", default="outputs/phase1_plane_detection/metrics")
    p.add_argument("--skip-image-check", action="store_true")
    args = p.parse_args()

    csv_path = PROJECT_ROOT / args.csv
    images_dir = PROJECT_ROOT / args.images_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_canonical_columns(load_metadata(csv_path))

    print("Columns:", list(df.columns))
    print("Rows:", len(df))
    print("Patients:", df["_patient_id"].nunique())
    print("\nPlane counts:")
    print(df["_plane"].value_counts(dropna=False))
    print("\nSplit counts:")
    print(df["_split"].value_counts(dropna=False))

    invalid_labels = sorted(set(df["_plane"]) - set(CANONICAL_LABELS))
    duplicate_names = int(df["_image_name"].duplicated().sum())

    missing = []
    if not args.skip_image_check:
        for i, name in enumerate(df["_image_name"], start=1):
            try:
                resolve_image_path(images_dir, name)
            except Exception as exc:
                missing.append({"image_name": name, "error": str(exc)})
            if i % 2000 == 0:
                print(f"Checked {i}/{len(df)} image paths...")

    audit = {
        "rows": int(len(df)),
        "unique_patients": int(df["_patient_id"].nunique()),
        "plane_counts": {str(k): int(v) for k, v in df["_plane"].value_counts().items()},
        "split_counts": {str(k): int(v) for k, v in df["_split"].value_counts().items()},
        "invalid_labels": invalid_labels,
        "duplicate_image_names": duplicate_names,
        "missing_image_count": len(missing),
    }

    (output_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [{"plane": k, "count": v} for k, v in audit["plane_counts"].items()]
    ).to_csv(output_dir / "class_distribution.csv", index=False)

    if missing:
        pd.DataFrame(missing).to_csv(output_dir / "missing_images.csv", index=False)

    print("\nSaved:", output_dir / "dataset_audit.json")
    print(json.dumps(audit, indent=2))

    if invalid_labels:
        raise SystemExit(f"FAIL: unexpected labels: {invalid_labels}")
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} images could not be resolved.")
    if "unknown" in audit["split_counts"]:
        print(
            "\nWARNING: Some Train/split values could not be interpreted. "
            "Inspect the raw unique values before continuing."
        )


if __name__ == "__main__":
    main()
