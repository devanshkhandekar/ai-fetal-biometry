#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plane_detection.dataset_utils import (
    add_canonical_columns,
    load_metadata,
    resolve_image_path,
)
from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/fetal_planes_db/FETAL_PLANES_DB_data.csv")
    p.add_argument("--images-dir", default="data/fetal_planes_db/Images")
    p.add_argument("--config", default="models/checkpoints/fetalclip/FetalCLIP_config.json")
    p.add_argument("--weights", default="models/checkpoints/fetalclip/FetalCLIP_weights.pt")
    p.add_argument(
        "--output",
        default="outputs/phase1_plane_detection/embeddings/fetal_planes_fetalclip.npz",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    df = add_canonical_columns(load_metadata(PROJECT_ROOT / args.csv))
    df = df[df["_plane"].isin(["abdomen", "brain", "femur", "thorax", "cervix", "other"])].copy()

    paths = [
        resolve_image_path(PROJECT_ROOT / args.images_dir, n)
        for n in df["_image_name"]
    ]

    detector = FetalCLIPPlaneDetector(
        config_path=PROJECT_ROOT / args.config,
        weights_path=PROJECT_ROOT / args.weights,
        device=args.device,
    )

    embeddings = detector.encode_images(paths, batch_size=args.batch_size)

    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=embeddings.astype(np.float32),
        labels=df["_plane"].astype(str).to_numpy(),
        patient_ids=df["_patient_id"].astype(str).to_numpy(),
        splits=df["_split"].astype(str).to_numpy(),
        image_names=df["_image_name"].astype(str).to_numpy(),
    )
    print(f"Saved {embeddings.shape} embeddings to {output}")


if __name__ == "__main__":
    main()
