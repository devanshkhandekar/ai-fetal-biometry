#!/usr/bin/env python
"""
Single-image inference adapter for surgical-vision/Multicentre-Fetal-Biometry.

This file MUST be executed with the dedicated fetalbiometry_hrnet environment.
It reproduces the repository's benchmark preprocessing:
  - RGB image
  - benchmark center/scale when a matching FP metadata row is available
  - scale *= 1.7 for FP metadata, exactly as FetalLandmarks.__getitem__()
  - crop to 256x256
  - ImageNet mean/std normalization
  - HRNet forward
  - decode heatmaps back to original-image coordinates

For non-benchmark uploaded images:
  - an explicit ROI can be supplied, OR
  - full-image fallback can be enabled.
Physical millimeters are produced only when pixel spacing is known.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--metadata-csv")
    p.add_argument("--metric", required=True, choices=["TAD", "APAD", "FL"])
    p.add_argument("--anatomy", required=True, choices=["abdomen", "femur"])
    p.add_argument("--pixel-spacing-mm", type=float)
    p.add_argument("--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--allow-full-image-fallback", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _drop_index_columns(df):
    junk = [
        c for c in df.columns
        if str(c).lower() == "index" or str(c).startswith("Unnamed")
    ]
    return df.drop(columns=junk) if junk else df


def _metadata_for_image(csv_path: Path | None, image_path: Path):
    if csv_path is None or not csv_path.exists():
        return None

    df = _drop_index_columns(pd.read_csv(csv_path))
    if df.empty:
        return None

    image_col = df.columns[0]
    names = df[image_col].astype(str).map(lambda x: Path(x).name)
    match = df.loc[names == image_path.name]
    if match.empty:
        return None

    row = match.iloc[0]
    # Repository format: image_name, scale, center_w, center_h, ...
    result = {
        "scale_raw": float(row.iloc[1]),
        "center_w": float(row.iloc[2]),
        "center_h": float(row.iloc[3]),
    }

    for key in ("px_to_mm_rate", "px_to_mm"):
        if key in row.index:
            try:
                value = float(row[key])
                if math.isfinite(value) and value > 0:
                    result["pixel_spacing_mm"] = value
                    break
            except Exception:
                pass

    return result


def _resolve_roi(image: Image.Image, metadata, roi, allow_full_image_fallback):
    w, h = image.size

    if metadata is not None:
        # Exact benchmark path: FetalLandmarks multiplies scale by 1.7.
        return {
            "center": [metadata["center_w"], metadata["center_h"]],
            "effective_scale": metadata["scale_raw"] * 1.7,
            "roi_source": "fp_metadata",
            "benchmark_preprocessing": True,
        }

    if roi is not None:
        x1, y1, x2, y2 = [float(v) for v in roi]
        if not (x2 > x1 and y2 > y1):
            raise ValueError(f"Invalid ROI: {roi}")
        x1 = max(0.0, min(float(w), x1))
        x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h), y1))
        y2 = max(0.0, min(float(h), y2))
        side = max(x2 - x1, y2 - y1)
        return {
            "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            "effective_scale": side / 200.0,
            "roi_source": "explicit_roi",
            "benchmark_preprocessing": False,
        }

    if allow_full_image_fallback:
        return {
            "center": [w / 2.0, h / 2.0],
            "effective_scale": max(w, h) / 200.0,
            "roi_source": "full_image_fallback",
            "benchmark_preprocessing": False,
        }

    raise ValueError(
        "No FP metadata row and no ROI supplied. "
        "Use --roi or --allow-full-image-fallback."
    )


def _load_model(repo_root: Path, cfg_path: Path, checkpoint_path: Path, device):
    sys.path.insert(0, str(repo_root))

    import lib.models as models
    from lib.config import config

    config.defrost()
    config.merge_from_file(str(cfg_path))
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    model = models.get_face_alignment_net(config)

    #payload = torch.load(str(checkpoint_path), map_location="cpu")
    try:
        payload = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch releases
        # that do not support the weights_only argument.
        payload = torch.load(
            str(checkpoint_path),
            map_location="cpu",
        )
    if isinstance(payload, dict) and "state_dict" in payload:
        weights = payload["state_dict"]
    else:
        weights = payload

    cleaned = {}
    for k, v in weights.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return config, model


def main():
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    cfg_path = (repo_root / args.cfg).resolve() if not Path(args.cfg).is_absolute() else Path(args.cfg)
    checkpoint = (repo_root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    image_path = Path(args.image).resolve()
    metadata_csv = None
    if args.metadata_csv:
        metadata_csv = (
            (repo_root / args.metadata_csv).resolve()
            if not Path(args.metadata_csv).is_absolute()
            else Path(args.metadata_csv)
        )

    if not repo_root.exists():
        raise FileNotFoundError(repo_root)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    pil = Image.open(image_path).convert("RGB")
    image_np = np.asarray(pil, dtype=np.float32)

    metadata = _metadata_for_image(metadata_csv, image_path)
    roi_info = _resolve_roi(
        pil,
        metadata,
        args.roi,
        args.allow_full_image_fallback,
    )

    pixel_spacing = args.pixel_spacing_mm
    if pixel_spacing is None and metadata is not None:
        pixel_spacing = metadata.get("pixel_spacing_mm")
    if pixel_spacing is not None:
        pixel_spacing = float(pixel_spacing)
        if not math.isfinite(pixel_spacing) or pixel_spacing <= 0:
            pixel_spacing = None

    config, model = _load_model(
        repo_root=repo_root,
        cfg_path=cfg_path,
        checkpoint_path=checkpoint,
        device=device,
    )

    # Imports must happen after repo path is inserted.
    from lib.utils.transforms import crop
    from lib.core.evaluation import decode_preds

    center = torch.tensor(roi_info["center"], dtype=torch.float32)
    scale = float(roi_info["effective_scale"])

    cropped = crop(
        image_np,
        center,
        scale,
        config.MODEL.IMAGE_SIZE,
        rot=0.0,
    ).astype(np.float32)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (cropped / 255.0 - mean) / std
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(inp)
        if isinstance(output, (list, tuple)):
            output = output[-1]

    score_map = output.detach().cpu()
    centers = center.view(1, 2)
    scales = torch.tensor([scale], dtype=torch.float32)
    preds = decode_preds(
        score_map,
        centers,
        scales,
        list(config.MODEL.HEATMAP_SIZE),
    )[0].numpy().astype(float)

    p1 = preds[0]
    p2 = preds[1]
    distance_px = float(np.linalg.norm(p1 - p2))
    distance_mm = (
        float(distance_px * pixel_spacing)
        if pixel_spacing is not None
        else None
    )

    heatmap_peaks = (
        score_map.reshape(score_map.shape[0], score_map.shape[1], -1)
        .amax(dim=2)[0]
        .numpy()
        .astype(float)
        .tolist()
    )

    w, h = pil.size
    finite = bool(np.isfinite(preds).all() and math.isfinite(distance_px))
    inside = bool(
        finite
        and (preds[:, 0] >= 0).all()
        and (preds[:, 0] < w).all()
        and (preds[:, 1] >= 0).all()
        and (preds[:, 1] < h).all()
    )

    if not finite or distance_px <= 0:
        status = "invalid"
        reasons = ["invalid_landmark_prediction"]
    elif not inside:
        status = "invalid"
        reasons = ["landmark_outside_image"]
    elif pixel_spacing is None:
        status = "review"
        reasons = ["pixel_spacing_unavailable"]
    elif roi_info["roi_source"] == "full_image_fallback":
        status = "review"
        reasons = ["full_image_roi_fallback_unvalidated"]
    else:
        status = "valid"
        reasons = []

    result = {
        "status": status,
        "metric": args.metric,
        "anatomy": args.anatomy,
        "image_name": image_path.name,
        "landmarks": [
            [float(p1[0]), float(p1[1])],
            [float(p2[0]), float(p2[1])],
        ],
        "distance_px": distance_px,
        "distance_mm": distance_mm,
        "pixel_spacing_mm": pixel_spacing,
        "center": [float(center[0]), float(center[1])],
        "effective_scale": scale,
        "roi_source": roi_info["roi_source"],
        "benchmark_preprocessing": roi_info["benchmark_preprocessing"],
        "heatmap_peaks": heatmap_peaks,
        "reasons": reasons,
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
    }

    # Keep this as the final stdout line so the parent process can parse it
    # even if a dependency emits informational text before it.
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
