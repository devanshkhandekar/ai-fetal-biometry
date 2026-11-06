#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plane_detection.dataset_utils import (
    add_canonical_columns,
    load_metadata,
    official_train_test,
    resolve_image_path,
)
from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector
from src.plane_detection.prompts import STANDARD_CLASSES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/fetal_planes_db/FETAL_PLANES_DB_data.csv")
    p.add_argument("--images-dir", default="data/fetal_planes_db/Images")
    p.add_argument("--config", default="models/checkpoints/fetalclip/FetalCLIP_config.json")
    p.add_argument("--weights", default="models/checkpoints/fetalclip/FetalCLIP_weights.pt")
    p.add_argument("--output-dir", default="outputs/phase1_plane_detection")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="Smoke-test only.")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    pred_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    raw = load_metadata(PROJECT_ROOT / args.csv)
    _, test_df = official_train_test(raw)
    if args.limit:
        test_df = test_df.head(args.limit).copy()

    image_paths = [
        resolve_image_path(PROJECT_ROOT / args.images_dir, name)
        for name in test_df["_image_name"]
    ]

    detector = FetalCLIPPlaneDetector(
        config_path=PROJECT_ROOT / args.config,
        weights_path=PROJECT_ROOT / args.weights,
        device=args.device,
    )

    results = detector.predict_batch(image_paths, batch_size=args.batch_size)

    rows = []
    for (_, src), image_path, result in zip(test_df.iterrows(), image_paths, results):
        row = {
            "image_name": src["_image_name"],
            "image_path": str(image_path),
            "patient_id": src["_patient_id"],
            "ground_truth": src["_plane"],
            "predicted_plane": result["predicted_plane"],
            "top_score": result["top_score"],
            "second_plane": result["second_plane"],
            "second_score": result["second_score"],
            "margin": result["margin"],
        }
        for cls, score in result["scores"].items():
            row[f"score_{cls}"] = score
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_csv = pred_dir / "fetal_planes_test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    standard = pred_df[pred_df["ground_truth"].isin(STANDARD_CLASSES)].copy()
    y_true = standard["ground_truth"].tolist()
    y_pred = standard["predicted_plane"].tolist()

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(STANDARD_CLASSES), average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(STANDARD_CLASSES),
        zero_division=0,
    )

    per_class = pd.DataFrame({
        "class": STANDARD_CLASSES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    per_class.to_csv(metrics_dir / "per_class_metrics_raw.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(STANDARD_CLASSES))
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(cm, display_labels=list(STANDARD_CLASSES)).plot(
        ax=ax, values_format="d", colorbar=False
    )
    ax.set_title("FetalCLIP zero-shot: 5 standard planes")
    fig.tight_layout()
    fig.savefig(metrics_dir / "confusion_matrix_raw.png", dpi=160)
    plt.close(fig)

    # Safety-oriented raw-error analysis.
    nonbrain = pred_df[pred_df["ground_truth"] != "brain"]
    false_brain = nonbrain[nonbrain["predicted_plane"] == "brain"]

    other = pred_df[pred_df["ground_truth"] == "other"]
    other_as_brain = other[other["predicted_plane"] == "brain"]

    metrics = {
        "test_rows_total": int(len(pred_df)),
        "standard_plane_rows": int(len(standard)),
        "other_rows": int(len(other)),
        "standard_accuracy": float(accuracy),
        "standard_macro_f1": float(macro_f1),
        "raw_false_brain_count": int(len(false_brain)),
        "raw_false_brain_rate_over_nonbrain": (
            float(len(false_brain) / len(nonbrain)) if len(nonbrain) else None
        ),
        "raw_other_as_brain_count": int(len(other_as_brain)),
        "note": (
            "The published zero-shot standard-plane benchmark excludes Other. "
            "Other rows here are intentionally used as open-set stress cases."
        ),
    }
    (metrics_dir / "metrics_raw.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    false_brain.to_csv(pred_dir / "false_brain_raw.csv", index=False)
    other.to_csv(pred_dir / "other_open_set_raw.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nPer-class metrics:")
    print(per_class.to_string(index=False))
    print(f"\nPredictions: {pred_csv}")


if __name__ == "__main__":
    main()
