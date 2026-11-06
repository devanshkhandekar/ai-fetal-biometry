#!/usr/bin/env python
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="data/sample_cases")
    ap.add_argument("--config", default="models/checkpoints/fetalclip/FetalCLIP_config.json")
    ap.add_argument("--weights", default="models/checkpoints/fetalclip/FetalCLIP_weights.pt")
    ap.add_argument("--probe", default="outputs/phase1_plane_detection/linear_probe/fetalclip_linear_probe.joblib")
    ap.add_argument("--metrics", default="outputs/phase1_plane_detection/linear_probe/metrics.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args=ap.parse_args()

    paths=sorted(p for p in (ROOT/args.images_dir).iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"})
    metrics=json.loads((ROOT/args.metrics).read_text())
    score_t=float(metrics["score_threshold"]); margin_t=float(metrics["margin_threshold"])

    encoder=FetalCLIPPlaneDetector(ROOT/args.config, ROOT/args.weights, device=args.device)
    X=encoder.encode_images(paths,batch_size=args.batch_size)
    probe=joblib.load(ROOT/args.probe)
    probs=probe.predict_proba(X); classes=[str(c) for c in probe.classes_]

    rows=[]
    for path, pr in zip(paths, probs):
        order=np.argsort(pr)[::-1]; a,b=int(order[0]),int(order[1])
        top=float(pr[a]); second=float(pr[b]); margin=top-second
        status="accept" if top>=score_t and margin>=margin_t else "review"
        rows.append({
            "image_name":path.name,
            "ground_truth":"brain",
            "predicted_plane":classes[a],
            "top_score":top,
            "second_plane":classes[b],
            "second_score":second,
            "margin":margin,
            "status":status
        })

    df=pd.DataFrame(rows)
    out=ROOT/"outputs/phase1_plane_detection/linear_probe"
    df.to_csv(out/"hc18_predictions.csv",index=False)
    summary={
        "total_hc18_images":len(df),
        "raw_brain_prediction_rate":float((df.predicted_plane=="brain").mean()),
        "accepted_brain_rate":float(((df.predicted_plane=="brain")&(df.status=="accept")).mean()),
        "review_rate":float((df.status=="review").mean()),
        "nonbrain_prediction_counts":{str(k):int(v) for k,v in df.loc[df.predicted_plane!="brain","predicted_plane"].value_counts().items()},
        "score_threshold":score_t,
        "margin_threshold":margin_t,
        "routing_enabled":False
    }
    (out/"hc18_metrics.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
