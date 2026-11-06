#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector


def load_thresholds(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("score_threshold"), data.get("margin_threshold"), data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", default="data/sample_cases")
    p.add_argument("--config", default="models/checkpoints/fetalclip/FetalCLIP_config.json")
    p.add_argument("--weights", default="models/checkpoints/fetalclip/FetalCLIP_weights.pt")
    p.add_argument(
        "--thresholds",
        default="outputs/phase1_plane_detection/metrics/selected_thresholds.json",
    )
    p.add_argument("--output-dir", default="outputs/phase1_plane_detection")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    image_dir = PROJECT_ROOT / args.images_dir
    paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
    )
    if not paths:
        raise SystemExit(f"No images found under {image_dir}")

    score_t, margin_t, threshold_meta = load_thresholds(PROJECT_ROOT / args.thresholds)

    detector = FetalCLIPPlaneDetector(
        config_path=PROJECT_ROOT / args.config,
        weights_path=PROJECT_ROOT / args.weights,
        device=args.device,
        score_threshold=score_t,
        margin_threshold=margin_t,
    )
    results = detector.predict_batch(paths, batch_size=args.batch_size)

    rows = []
    for path, result in zip(paths, results):
        rows.append({
            "image_name": path.name,
            "ground_truth": "brain",
            "predicted_plane": result["predicted_plane"],
            "top_score": result["top_score"],
            "second_plane": result["second_plane"],
            "second_score": result["second_score"],
            "margin": result["margin"],
            "status": result["status"],
            "application_plane": result["application_plane"],
        })
    df = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / args.output_dir
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(pred_dir / "hc18_plane_results.csv", index=False)

    metrics = {
        "total_hc18_images": int(len(df)),
        "raw_brain_prediction_rate": float((df["predicted_plane"] == "brain").mean()),
        "accepted_brain_rate": float(
            ((df["predicted_plane"] == "brain") & (df["status"] == "accept")).mean()
        ),
        "review_rate": float((df["status"] == "review").mean()),
        "misclassification_counts": {
            str(k): int(v)
            for k, v in df.loc[df["predicted_plane"] != "brain", "predicted_plane"]
            .value_counts()
            .items()
        },
        "thresholds": {
            "score_threshold": score_t,
            "margin_threshold": margin_t,
            "thresholds_found": threshold_meta.get("thresholds_found"),
        },
        "routing_enabled": False,
    }
    (metrics_dir / "hc18_plane_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
