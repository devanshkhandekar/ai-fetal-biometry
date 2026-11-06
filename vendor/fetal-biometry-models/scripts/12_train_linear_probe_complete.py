#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["abdomen","brain","femur","thorax","cervix","other"]
ROUTABLE = {"brain","abdomen","femur"}

def make_predictions(model, X, y, groups, names):
    probs = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    rows = []
    for i, pr in enumerate(probs):
        order = np.argsort(pr)[::-1]
        a, b = int(order[0]), int(order[1])
        row = dict(
            ground_truth=str(y[i]),
            predicted_plane=classes[a],
            top_score=float(pr[a]),
            second_plane=classes[b],
            second_score=float(pr[b]),
            margin=float(pr[a]-pr[b]),
            patient_id=str(groups[i]),
            image_name=str(names[i]),
        )
        rows.append(row)
    return pd.DataFrame(rows)

def policy(df, score_t, margin_t):
    accepted = (df.top_score >= score_t) & (df.margin >= margin_t)
    gt = df.ground_truth
    pred = df.predicted_plane
    standard = gt != "other"
    brain = gt == "brain"
    nonbrain = ~brain
    other = gt == "other"
    false_head = accepted & nonbrain & (pred == "brain")
    other_to_routable = accepted & other & pred.isin(ROUTABLE)
    return {
        "score_threshold": float(score_t),
        "margin_threshold": float(margin_t),
        "overall_coverage": float(accepted.mean()),
        "standard_coverage": float((accepted & standard).sum()/standard.sum()),
        "accepted_standard_accuracy": float((pred[accepted & standard] == gt[accepted & standard]).mean()) if (accepted & standard).any() else 0.0,
        "brain_accept_recall": float((accepted & brain & (pred=="brain")).sum()/brain.sum()),
        "false_head_accept_rate": float(false_head.sum()/nonbrain.sum()),
        "false_head_accept_count": int(false_head.sum()),
        "other_raw_recall": float(((pred=="other") & other).sum()/other.sum()),
        "other_to_routable_accept_rate": float(other_to_routable.sum()/other.sum()),
        "other_to_routable_accept_count": int(other_to_routable.sum()),
        "review_rate": float((~accepted).mean()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="outputs/phase1_plane_detection/embeddings/fetal_planes_fetalclip.npz")
    ap.add_argument("--output-dir", default="outputs/phase1_plane_detection/linear_probe")
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--max-false-head-rate", type=float, default=0.005)
    ap.add_argument("--max-other-routable-rate", type=float, default=0.10)
    args = ap.parse_args()

    d = np.load(ROOT/args.embeddings, allow_pickle=True)
    X = d["embeddings"].astype(np.float32)
    y = d["labels"].astype(str)
    groups = d["patient_ids"].astype(str)
    splits = d["splits"].astype(str)
    names = d["image_names"].astype(str)

    train_mask, test_mask = splits=="train", splits=="test"
    Xtr, ytr, gtr, ntr = X[train_mask], y[train_mask], groups[train_mask], names[train_mask]
    Xte, yte, gte, nte = X[test_mask], y[test_mask], groups[test_mask], names[test_mask]

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    dev_idx, val_idx = next(splitter.split(Xtr, ytr, groups=gtr))
    Xdev, ydev = Xtr[dev_idx], ytr[dev_idx]
    Xval, yval, gval, nval = Xtr[val_idx], ytr[val_idx], gtr[val_idx], ntr[val_idx]

    best_c, best_f1, best_model = None, -1, None
    candidates = []
    for c in [0.01,0.1,1.0,10.0,100.0]:
        m = LogisticRegression(C=c, max_iter=5000, class_weight="balanced", solver="lbfgs")
        m.fit(Xdev, ydev)
        pv = m.predict(Xval)
        mf1 = f1_score(yval, pv, labels=CLASSES, average="macro", zero_division=0)
        candidates.append({"C":c,"validation_macro_f1":float(mf1)})
        if mf1 > best_f1:
            best_c, best_f1, best_model = c, float(mf1), m

    val_df = make_predictions(best_model, Xval, yval, gval, nval)
    sweep = []
    for s in np.round(np.arange(0.50,0.951,0.05),2):
        for m in np.round(np.arange(0.00,0.501,0.05),2):
            sweep.append(policy(val_df,s,m))
    sweep_df = pd.DataFrame(sweep)

    eligible = sweep_df[
        (sweep_df.false_head_accept_rate <= args.max_false_head_rate) &
        (sweep_df.other_to_routable_accept_rate <= args.max_other_routable_rate)
    ]
    if len(eligible):
        chosen = eligible.sort_values(
            ["standard_coverage","brain_accept_recall","accepted_standard_accuracy"],
            ascending=False
        ).iloc[0]
        thresholds_found = True
        reason = "Validation thresholds met configured engineering targets."
    else:
        chosen = sweep_df.sort_values(
            ["false_head_accept_rate","other_to_routable_accept_rate","standard_coverage"],
            ascending=[True,True,False]
        ).iloc[0]
        thresholds_found = False
        reason = "No validation thresholds met configured engineering targets; keep routing disabled."

    score_t, margin_t = float(chosen.score_threshold), float(chosen.margin_threshold)

    final = LogisticRegression(C=best_c, max_iter=5000, class_weight="balanced", solver="lbfgs")
    final.fit(Xtr, ytr)
    test_df = make_predictions(final, Xte, yte, gte, nte)

    raw_pred = test_df.predicted_plane.to_numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        yte, raw_pred, labels=CLASSES, zero_division=0
    )
    per_class = pd.DataFrame({"class":CLASSES,"precision":precision,"recall":recall,"f1":f1,"support":support})
    test_policy = policy(test_df, score_t, margin_t)

    accepted = (test_df.top_score >= score_t) & (test_df.margin >= margin_t)
    test_df["status"] = np.where(accepted,"accept","review")

    out = ROOT/args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out/"fetalclip_linear_probe.joblib")
    val_df.to_csv(out/"validation_predictions.csv", index=False)
    test_df.to_csv(out/"test_predictions.csv", index=False)
    sweep_df.to_csv(out/"threshold_sweep.csv", index=False)
    per_class.to_csv(out/"per_class_metrics.csv", index=False)

    test_df[(test_df.ground_truth!="brain") & (test_df.predicted_plane=="brain") & (test_df.status=="accept")].to_csv(out/"accepted_false_head.csv", index=False)
    test_df[(test_df.ground_truth=="other") & (test_df.predicted_plane.isin(ROUTABLE)) & (test_df.status=="accept")].to_csv(out/"accepted_other_to_routable.csv", index=False)

    cm = confusion_matrix(yte, raw_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8,8))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    fig.savefig(out/"confusion_matrix.png", dpi=160)
    plt.close(fig)

    summary = {
        "best_C": best_c,
        "validation_macro_f1": best_f1,
        "candidate_C_metrics": candidates,
        "thresholds_found": thresholds_found,
        "score_threshold": score_t,
        "margin_threshold": margin_t,
        "official_test_raw": {
            "accuracy": float(accuracy_score(yte, raw_pred)),
            "macro_f1": float(f1_score(yte, raw_pred, labels=CLASSES, average="macro", zero_division=0))
        },
        "official_test_policy": test_policy,
        "targets": {
            "max_false_head_rate": args.max_false_head_rate,
            "max_other_to_routable_accept_rate": args.max_other_routable_rate
        },
        "reason": reason,
        "routing_enabled": False
    }
    (out/"metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(per_class.to_string(index=False))

if __name__ == "__main__":
    main()
