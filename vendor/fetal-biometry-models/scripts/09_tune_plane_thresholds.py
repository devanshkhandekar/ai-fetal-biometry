#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plane_detection.dataset_utils import (
    load_metadata,
    official_train_test,
    resolve_image_path,
)
from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector
from src.plane_detection.prompts import STANDARD_CLASSES


def policy_metrics(df: pd.DataFrame, score_t: float, margin_t: float) -> dict:
    accepted = (df["top_score"] >= score_t) & (df["margin"] >= margin_t)

    standard_mask = df["ground_truth"].isin(STANDARD_CLASSES)
    other_mask = df["ground_truth"] == "other"
    brain_mask = df["ground_truth"] == "brain"
    nonbrain_mask = df["ground_truth"] != "brain"

    correct = df["predicted_plane"] == df["ground_truth"]

    standard_accepted = accepted & standard_mask
    accepted_accuracy = (
        float(correct[standard_accepted].mean())
        if standard_accepted.any() else 0.0
    )
    standard_coverage = (
        float(standard_accepted.sum() / standard_mask.sum())
        if standard_mask.any() else 0.0
    )
    overall_coverage = float(accepted.mean()) if len(df) else 0.0

    false_head_accepted = accepted & nonbrain_mask & (df["predicted_plane"] == "brain")
    false_head_rate = (
        float(false_head_accepted.sum() / nonbrain_mask.sum())
        if nonbrain_mask.any() else 0.0
    )

    brain_accepted_correct = accepted & brain_mask & (df["predicted_plane"] == "brain")
    brain_accept_recall = (
        float(brain_accepted_correct.sum() / brain_mask.sum())
        if brain_mask.any() else 0.0
    )

    other_accepted = accepted & other_mask
    other_accept_rate = (
        float(other_accepted.sum() / other_mask.sum())
        if other_mask.any() else 0.0
    )

    return {
        "score_threshold": float(score_t),
        "margin_threshold": float(margin_t),
        "overall_coverage": overall_coverage,
        "standard_coverage": standard_coverage,
        "accepted_standard_accuracy": accepted_accuracy,
        "brain_accept_recall": brain_accept_recall,
        "false_head_accept_rate": false_head_rate,
        "false_head_accept_count": int(false_head_accepted.sum()),
        "other_accept_rate": other_accept_rate,
        "other_accept_count": int(other_accepted.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/fetal_planes_db/FETAL_PLANES_DB_data.csv")
    p.add_argument("--images-dir", default="data/fetal_planes_db/Images")
    p.add_argument("--config", default="models/checkpoints/fetalclip/FetalCLIP_config.json")
    p.add_argument("--weights", default="models/checkpoints/fetalclip/FetalCLIP_weights.pt")
    p.add_argument("--output-dir", default="outputs/phase1_plane_detection")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--max-false-head-rate", type=float, default=0.005)
    p.add_argument(
        "--max-other-accept-rate",
        type=float,
        default=0.10,
        help="Engineering target for open-set Other rejection. Adjust if justified.",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    raw = load_metadata(PROJECT_ROOT / args.csv)
    train_df, _ = official_train_test(raw)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.val_fraction,
        random_state=args.seed,
    )
    _, val_idx = next(
        splitter.split(train_df, groups=train_df["_patient_id"])
    )
    val_df = train_df.iloc[val_idx].copy()

    if args.limit:
        val_df = val_df.head(args.limit).copy()

    print(
        f"Validation rows={len(val_df)}, "
        f"patients={val_df['_patient_id'].nunique()}"
    )
    print(val_df["_plane"].value_counts())

    image_paths = [
        resolve_image_path(PROJECT_ROOT / args.images_dir, name)
        for name in val_df["_image_name"]
    ]

    detector = FetalCLIPPlaneDetector(
        config_path=PROJECT_ROOT / args.config,
        weights_path=PROJECT_ROOT / args.weights,
        device=args.device,
    )
    results = detector.predict_batch(image_paths, batch_size=args.batch_size)

    rows = []
    for (_, src), path, result in zip(val_df.iterrows(), image_paths, results):
        rows.append({
            "image_name": src["_image_name"],
            "image_path": str(path),
            "patient_id": src["_patient_id"],
            "ground_truth": src["_plane"],
            "predicted_plane": result["predicted_plane"],
            "top_score": result["top_score"],
            "second_plane": result["second_plane"],
            "second_score": result["second_score"],
            "margin": result["margin"],
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_dir / "threshold_validation_predictions.csv", index=False)

    score_grid = np.round(np.arange(0.50, 0.951, 0.05), 2)
    margin_grid = np.round(np.arange(0.00, 0.501, 0.05), 2)

    sweep_rows = []
    for score_t in score_grid:
        for margin_t in margin_grid:
            sweep_rows.append(policy_metrics(pred_df, score_t, margin_t))

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(metrics_dir / "threshold_sweep.csv", index=False)

    eligible = sweep[
        (sweep["false_head_accept_rate"] <= args.max_false_head_rate)
        & (sweep["other_accept_rate"] <= args.max_other_accept_rate)
    ].copy()

    if not eligible.empty:
        chosen = eligible.sort_values(
            ["standard_coverage", "brain_accept_recall", "accepted_standard_accuracy"],
            ascending=False,
        ).iloc[0]
        thresholds_found = True
        reason = "Meets configured false-head and Other-rejection engineering targets."
    else:
        # Do not pretend a safe threshold was found. Pick the least-bad row for
        # diagnostics only and explicitly mark failure.
        chosen = sweep.sort_values(
            ["false_head_accept_rate", "other_accept_rate", "standard_coverage"],
            ascending=[True, True, False],
        ).iloc[0]
        thresholds_found = False
        reason = (
            "No score/margin pair met the configured safety targets. "
            "Do not enable hard routing; consider linear probing with an explicit Other class."
        )

    selected = {
        "thresholds_found": thresholds_found,
        "score_threshold": float(chosen["score_threshold"]),
        "margin_threshold": float(chosen["margin_threshold"]),
        "validation_metrics": {
            key: float(chosen[key]) if key not in {"false_head_accept_count", "other_accept_count"}
            else int(chosen[key])
            for key in [
                "overall_coverage",
                "standard_coverage",
                "accepted_standard_accuracy",
                "brain_accept_recall",
                "false_head_accept_rate",
                "false_head_accept_count",
                "other_accept_rate",
                "other_accept_count",
            ]
        },
        "targets": {
            "max_false_head_rate": args.max_false_head_rate,
            "max_other_accept_rate": args.max_other_accept_rate,
        },
        "reason": reason,
        "routing_enabled": False,
    }
    (metrics_dir / "selected_thresholds.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )

    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
