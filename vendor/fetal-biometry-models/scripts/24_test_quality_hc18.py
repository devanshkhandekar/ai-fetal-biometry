#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/sample_cases")
    p.add_argument(
        "--probe",
        default="outputs/phase2_image_quality/model/fetalclip_quality_probe.joblib",
    )
    p.add_argument(
        "--metrics",
        default="outputs/phase2_image_quality/model/metrics.json",
    )
    p.add_argument(
        "--config",
        default="models/checkpoints/fetalclip/FetalCLIP_config.json",
    )
    p.add_argument(
        "--weights",
        default="models/checkpoints/fetalclip/FetalCLIP_weights.pt",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    image_dir = ROOT / args.images_dir
    images = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"No images found under {image_dir}")

    meta = json.loads((ROOT / args.metrics).read_text(encoding="utf-8"))
    good_t = float(meta["good_threshold"])
    margin_t = float(meta["margin_threshold"])

    encoder = FetalCLIPPlaneDetector(
        config_path=ROOT / args.config,
        weights_path=ROOT / args.weights,
        device=args.device,
    )
    X = encoder.encode_images(images, batch_size=args.batch_size)

    probe = joblib.load(ROOT / args.probe)
    probs = probe.predict_proba(X)
    classes = [str(c) for c in probe.classes_]

    rows = []
    for path, pr in zip(images, probs):
        order = np.argsort(pr)[::-1]
        a, b = int(order[0]), int(order[1])
        raw = classes[a]
        top = float(pr[a])
        second = float(pr[b])
        margin = top - second

        policy = raw
        if raw == "good":
            good_score = float(pr[classes.index("good")])
            if good_score < good_t or margin < margin_t:
                policy = "review"

        rows.append({
            "image_name": path.name,
            "raw_prediction": raw,
            "policy_label": policy,
            "top_score": top,
            "second_label": classes[b],
            "second_score": second,
            "margin": margin,
            **{f"score_{classes[i]}": float(pr[i]) for i in range(len(classes))},
        })

    df = pd.DataFrame(rows)
    out_dir = ROOT / "outputs/phase2_image_quality/hc18_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "hc18_quality_results.csv", index=False)

    summary = {
        "total_images": int(len(df)),
        "policy_counts": {
            str(k): int(v)
            for k, v in df["policy_label"].value_counts().items()
        },
        "policy_rates": {
            str(k): float(v)
            for k, v in df["policy_label"].value_counts(normalize=True).items()
        },
        "good_threshold": good_t,
        "margin_threshold": margin_t,
        "mode": "advisory",
        "note": (
            "These are sanity results on original ultrasound images. "
            "They are not clinician quality labels."
        ),
    }
    (out_dir / "hc18_quality_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
