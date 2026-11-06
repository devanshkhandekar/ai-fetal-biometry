#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing original HC18/standard-plane images.",
    )
    p.add_argument(
        "--output",
        default="outputs/phase2_image_quality/quality_manifest.csv",
    )
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac", type=float, default=0.15)
    args = p.parse_args()

    if args.train_frac <= 0 or args.val_frac <= 0:
        raise SystemExit("train-frac and val-frac must be > 0.")
    test_frac = 1.0 - args.train_frac - args.val_frac
    if test_frac <= 0:
        raise SystemExit("train-frac + val-frac must be < 1.")

    root = Path(args.images_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    # images = sorted(
    #     p for p in root.rglob("*")
    #     if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    # )

    images = sorted(
    p for p in root.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.stem.lower().endswith("hc")
        and "annotation" not in p.stem.lower()
        and "mask" not in p.stem.lower()
    ))
    if not images:
        raise SystemExit(f"No supported images found under {root}")

    # Split source images BEFORE creating variants. This prevents the original
    # image appearing in train while its degraded derivative appears in test.
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)

    n_train = int(round(len(images) * args.train_frac))
    n_val = int(round(len(images) * args.val_frac))

    split_by_idx = {}
    for rank, idx in enumerate(indices):
        if rank < n_train:
            split_by_idx[int(idx)] = "train"
        elif rank < n_train + n_val:
            split_by_idx[int(idx)] = "val"
        else:
            split_by_idx[int(idx)] = "test"

    rows = []
    label_offsets = {"good": 1, "review": 2, "poor": 3}
    for idx, path in enumerate(images):
        source_id = path.stem
        split = split_by_idx[idx]
        for label in ["good", "review", "poor"]:
            rows.append({
                "source_id": source_id,
                "source_path": str(path),
                "image_name": path.name,
                "split": split,
                "quality_label": label,
                "variant_seed": int(args.seed + idx * 10 + label_offsets[label]),
            })

    df = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Source images: {len(images)}")
    print(f"Proxy samples: {len(df)}")
    print("\nSplit x label:")
    print(df.groupby(["split", "quality_label"]).size())
    print(f"\nSaved: {output}")

    # Sanity: every source image must belong to only one split.
    leaks = df.groupby("source_id")["split"].nunique()
    assert int((leaks > 1).sum()) == 0, "Source-image leakage detected."


if __name__ == "__main__":
    main()
