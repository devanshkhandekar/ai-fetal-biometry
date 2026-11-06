#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["good", "review", "poor"]


def prediction_table(model, X, y, source_ids, image_names):
    probs = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    rows = []

    for i, pr in enumerate(probs):
        order = np.argsort(pr)[::-1]
        a, b = int(order[0]), int(order[1])
        row = {
            "ground_truth": str(y[i]),
            "raw_prediction": classes[a],
            "top_score": float(pr[a]),
            "second_label": classes[b],
            "second_score": float(pr[b]),
            "margin": float(pr[a] - pr[b]),
            "source_id": str(source_ids[i]),
            "image_name": str(image_names[i]),
        }
        for j, cls in enumerate(classes):
            row[f"score_{cls}"] = float(pr[j])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_policy(df: pd.DataFrame, good_threshold: float, margin_threshold: float):
    out = df.copy()
    good_ok = (
        (out["raw_prediction"] == "good")
        & (out["score_good"] >= good_threshold)
        & (out["margin"] >= margin_threshold)
    )

    # REVIEW and POOR raw predictions are retained. Only uncertain GOOD
    # predictions are demoted to REVIEW.
    out["policy_label"] = out["raw_prediction"]
    demote = (out["raw_prediction"] == "good") & (~good_ok)
    out.loc[demote, "policy_label"] = "review"
    return out


def policy_metrics(df: pd.DataFrame):
    truth = df["ground_truth"]
    pred = df["policy_label"]

    non_good = truth.isin(["review", "poor"])
    false_good = non_good & (pred == "good")
    true_good = truth == "good"

    return {
        "policy_accuracy": float(accuracy_score(truth, pred)),
        "policy_macro_f1": float(
            f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "good_accept_recall": (
            float(((true_good) & (pred == "good")).sum() / true_good.sum())
            if true_good.any() else 0.0
        ),
        "false_good_rate": (
            float(false_good.sum() / non_good.sum())
            if non_good.any() else 0.0
        ),
        "false_good_count": int(false_good.sum()),
        "poor_as_good_rate": (
            float(((truth == "poor") & (pred == "good")).sum() / (truth == "poor").sum())
            if (truth == "poor").any() else 0.0
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embeddings",
        default="outputs/phase2_image_quality/quality_embeddings.npz",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/phase2_image_quality/model",
    )
    p.add_argument("--max-false-good-rate", type=float, default=0.05)
    args = p.parse_args()

    data = np.load(ROOT / args.embeddings, allow_pickle=True)
    X = data["embeddings"].astype(np.float32)
    y = data["labels"].astype(str)
    splits = data["splits"].astype(str)
    source_ids = data["source_ids"].astype(str)
    image_names = data["image_names"].astype(str)

    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"
    if not tr.any() or not va.any() or not te.any():
        raise SystemExit("Expected non-empty train, val and test splits.")

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]

    best_model = None
    best_c = None
    best_val_f1 = -1.0
    c_rows = []

    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = LogisticRegression(
            C=c,
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        macro = f1_score(
            yva,
            pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        )
        c_rows.append({"C": c, "validation_macro_f1": float(macro)})
        if macro > best_val_f1:
            best_val_f1 = float(macro)
            best_c = c
            best_model = model

    assert best_model is not None

    val_df = prediction_table(
        best_model,
        Xva,
        yva,
        source_ids[va],
        image_names[va],
    )

    sweep_rows = []
    for good_t in np.round(np.arange(0.50, 0.951, 0.05), 2):
        for margin_t in np.round(np.arange(0.00, 0.401, 0.05), 2):
            policy_df = apply_policy(val_df, good_t, margin_t)
            m = policy_metrics(policy_df)
            sweep_rows.append({
                "good_threshold": float(good_t),
                "margin_threshold": float(margin_t),
                **m,
            })

    sweep = pd.DataFrame(sweep_rows)
    eligible = sweep[
        sweep["false_good_rate"] <= args.max_false_good_rate
    ].copy()

    if eligible.empty:
        chosen = sweep.sort_values(
            ["false_good_rate", "good_accept_recall", "policy_macro_f1"],
            ascending=[True, False, False],
        ).iloc[0]
        thresholds_found = False
        reason = (
            "No validation threshold pair met the configured false-GOOD target. "
            "Keep quality output advisory and inspect failures."
        )
    else:
        chosen = eligible.sort_values(
            ["good_accept_recall", "policy_macro_f1", "false_good_rate"],
            ascending=[False, False, True],
        ).iloc[0]
        thresholds_found = True
        reason = "Validation threshold pair met the configured false-GOOD target."

    good_t = float(chosen["good_threshold"])
    margin_t = float(chosen["margin_threshold"])

    test_df = prediction_table(
        best_model,
        Xte,
        yte,
        source_ids[te],
        image_names[te],
    )
    test_policy = apply_policy(test_df, good_t, margin_t)

    raw_pred = test_df["raw_prediction"].to_numpy()
    policy_pred = test_policy["policy_label"].to_numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        yte,
        policy_pred,
        labels=CLASSES,
        zero_division=0,
    )

    per_class = pd.DataFrame({
        "class": CLASSES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, out / "fetalclip_quality_probe.joblib")
    val_df.to_csv(out / "validation_predictions.csv", index=False)
    sweep.to_csv(out / "threshold_sweep.csv", index=False)
    test_policy.to_csv(out / "test_predictions.csv", index=False)
    per_class.to_csv(out / "per_class_metrics.csv", index=False)

    # Failure slices.
    false_good = test_policy[
        test_policy["ground_truth"].isin(["review", "poor"])
        & (test_policy["policy_label"] == "good")
    ]
    false_poor = test_policy[
        (test_policy["ground_truth"] == "good")
        & (test_policy["policy_label"] == "poor")
    ]
    ambiguous = test_policy[
        test_policy["margin"] < 0.10
    ]

    false_good.to_csv(out / "false_good.csv", index=False)
    false_poor.to_csv(out / "false_poor.csv", index=False)
    ambiguous.to_csv(out / "ambiguous.csv", index=False)

    # Policy confusion matrix.
    cm = confusion_matrix(yte, policy_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Phase-2 FetalCLIP quality probe")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    summary = {
        "model": "Frozen FetalCLIP embeddings + LogisticRegression",
        "labels": CLASSES,
        "label_source": "HC18 originals + deterministic synthetic degradations",
        "clinical_ground_truth": False,
        "best_C": best_c,
        "validation_raw_macro_f1": best_val_f1,
        "candidate_C_metrics": c_rows,
        "thresholds_found": thresholds_found,
        "good_threshold": good_t,
        "margin_threshold": margin_t,
        "validation_selection": {
            key: (
                int(chosen[key])
                if key.endswith("_count")
                else float(chosen[key])
            )
            for key in chosen.index
        },
        "official_proxy_test_raw": {
            "accuracy": float(accuracy_score(yte, raw_pred)),
            "macro_f1": float(
                f1_score(
                    yte,
                    raw_pred,
                    labels=CLASSES,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "official_proxy_test_policy": policy_metrics(test_policy),
        "max_false_good_rate_target": args.max_false_good_rate,
        "reason": reason,
        "mode": "advisory",
    }
    (out / "metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("\nPer-class test metrics:")
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
