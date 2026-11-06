#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.image_quality.degradations import apply_quality_variant
from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="outputs/phase2_image_quality/quality_manifest.csv",
    )
    p.add_argument(
        "--config",
        default="models/checkpoints/fetalclip/FetalCLIP_config.json",
    )
    p.add_argument(
        "--weights",
        default="models/checkpoints/fetalclip/FetalCLIP_weights.pt",
    )
    p.add_argument(
        "--output",
        default="outputs/phase2_image_quality/quality_embeddings.npz",
    )
    p.add_argument(
        "--sample-dir",
        default="outputs/phase2_image_quality/sample_variants",
    )
    p.add_argument("--sample-per-class", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    df = pd.read_csv(ROOT / args.manifest)
    required = {
        "source_id", "source_path", "image_name", "split",
        "quality_label", "variant_seed",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {sorted(missing)}")

    detector = FetalCLIPPlaneDetector(
        config_path=ROOT / args.config,
        weights_path=ROOT / args.weights,
        device=args.device,
    )

    sample_dir = ROOT / args.sample_dir
    sample_counts = {"good": 0, "review": 0, "poor": 0}
    all_embeddings = []

    for start in range(0, len(df), args.batch_size):
        chunk = df.iloc[start:start + args.batch_size]
        pil_images = []

        for _, row in chunk.iterrows():
            source_path = Path(str(row["source_path"]))
            if not source_path.exists():
                raise FileNotFoundError(source_path)

            with Image.open(source_path) as im:
                variant, recipe = apply_quality_variant(
                    im,
                    quality_label=str(row["quality_label"]),
                    seed=int(row["variant_seed"]),
                )

            label = str(row["quality_label"])
            if sample_counts[label] < args.sample_per_class:
                class_dir = sample_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)
                save_name = (
                    f"{sample_counts[label]:02d}_"
                    f"{Path(str(row['image_name'])).stem}_"
                    f"{recipe.replace('+', '_')}.png"
                )
                variant.save(class_dir / save_name)
                sample_counts[label] += 1

            pil_images.append(variant)

        emb = detector.encode_images(
            pil_images,
            batch_size=max(1, len(pil_images)),
        )
        all_embeddings.append(emb.astype(np.float32))

        done = min(start + args.batch_size, len(df))
        if done % 200 == 0 or done == len(df):
            print(f"Embedded {done}/{len(df)} samples")

    embeddings = np.concatenate(all_embeddings, axis=0)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        embeddings=embeddings,
        labels=df["quality_label"].astype(str).to_numpy(),
        splits=df["split"].astype(str).to_numpy(),
        source_ids=df["source_id"].astype(str).to_numpy(),
        image_names=df["image_name"].astype(str).to_numpy(),
        source_paths=df["source_path"].astype(str).to_numpy(),
        variant_seeds=df["variant_seed"].astype(np.int64).to_numpy(),
    )

    print("\nSaved:", output)
    print("Embedding shape:", embeddings.shape)
    print("Sample gallery:", sample_dir)


if __name__ == "__main__":
    main()
