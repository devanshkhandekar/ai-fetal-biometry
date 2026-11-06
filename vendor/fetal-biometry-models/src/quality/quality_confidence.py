from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


def image_quality_features(img: np.ndarray) -> Dict[str, Any]:
    arr = img.astype(np.float32)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = float(cv2.Laplacian(arr, cv2.CV_32F).var())
    contrast = float(np.std(arr))
    p1, p99 = np.percentile(arr, [1, 99])
    dynamic_range = float(p99 - p1)
    base = arr if arr.max() > 1 else arr * 255
    med = cv2.medianBlur(base.astype(np.uint8), 3)
    noise = float(np.std(base.astype(np.float32) - med.astype(np.float32)))
    return {
        "blur_score": blur,
        "contrast_score": contrast,
        "dynamic_range": dynamic_range,
        "noise_estimate": noise,
    }


def mask_quality_features(mask: np.ndarray, ellipse: Dict[str, Any] | None) -> Dict[str, Any]:
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    area = int(binary.sum())
    area_ratio = float(area / max(h * w, 1))
    edge_touch_ratio = float((binary[0, :].sum() + binary[-1, :].sum() + binary[:, 0].sum() + binary[:, -1].sum()) / max(area, 1))
    features: Dict[str, Any] = {
        "mask_area_px": area,
        "mask_area_ratio": area_ratio,
        "edge_touch_ratio": edge_touch_ratio,
    }
    if ellipse:
        major = float(ellipse.get("major_diameter_px", ellipse.get("major_axis_px", 0.0)))
        minor = float(ellipse.get("minor_diameter_px", ellipse.get("minor_axis_px", 0.0)))
        features.update({
            "ellipse_axis_ratio": float(major / max(minor, 1e-6)),
            "ellipse_mask_iou": float(ellipse.get("ellipse_mask_iou", 0.0)),
            "center_offset_norm": float(ellipse.get("center_offset_norm", 0.0)),
            "ellipse_outside_fraction": float((ellipse.get("bounds", {}) or {}).get("outside_fraction", 0.0)),
            "ellipse_area_px2": float(ellipse.get("ellipse_area_px2", 0.0)),
        })
        features["mask_to_ellipse_area_ratio"] = float(area / max(features["ellipse_area_px2"], 1e-6))
    else:
        features.update({
            "ellipse_axis_ratio": None,
            "ellipse_mask_iou": 0.0,
            "center_offset_norm": None,
            "ellipse_outside_fraction": None,
            "mask_to_ellipse_area_ratio": None,
        })
    return features


def score_quality(img: np.ndarray, mask: np.ndarray, ellipse: Dict[str, Any] | None) -> Dict[str, Any]:
    """Heuristic MVP quality score with stronger ellipse-fit validation."""
    img_f = image_quality_features(img)
    mask_f = mask_quality_features(mask, ellipse)
    warnings: List[str] = []
    score = 100.0

    if img_f["dynamic_range"] < 25:
        score -= 20
        warnings.append("low_dynamic_range")
    if img_f["blur_score"] < 20:
        score -= 15
        warnings.append("possible_blur")

    if mask_f["mask_area_ratio"] < 0.02 or mask_f["mask_area_ratio"] > 0.55:
        score -= 25
        warnings.append("abnormal_mask_coverage")
    if mask_f["edge_touch_ratio"] > 0.025:
        score -= 20
        warnings.append("head_mask_touches_image_edge")

    if ellipse is None:
        score -= 45
        warnings.append("missing_ellipse")
    else:
        axis_ratio = mask_f["ellipse_axis_ratio"]
        iou = mask_f["ellipse_mask_iou"]
        center_offset = mask_f["center_offset_norm"] or 0.0
        outside_fraction = mask_f["ellipse_outside_fraction"] or 0.0
        area_ratio = mask_f["mask_to_ellipse_area_ratio"]

        if axis_ratio is None or axis_ratio < 1.02:
            score -= 15
            warnings.append("ellipse_too_circular_or_unstable")
        elif axis_ratio > 2.20:
            score -= 40
            warnings.append("ellipse_axis_ratio_implausible")
        elif axis_ratio > 1.80:
            score -= 25
            warnings.append("ellipse_axis_ratio_review")

        if iou < 0.50:
            score -= 40
            warnings.append("ellipse_mask_mismatch_reject")
        elif iou < 0.65:
            score -= 25
            warnings.append("ellipse_mask_mismatch_review")

        if center_offset > 0.08:
            score -= 15
            warnings.append("ellipse_center_offset_high")

        if outside_fraction > 0.02:
            score -= 30
            warnings.append("ellipse_outside_image_bounds")

        if area_ratio is not None and not (0.60 <= area_ratio <= 1.45):
            score -= 15
            warnings.append("mask_ellipse_area_ratio_unstable")

    score = float(np.clip(score, 0, 100))
    label = "good" if score >= 75 else "review" if score >= 45 else "poor"
    return {"quality_score": score, "quality_label": label, "warnings": warnings, "features": {**img_f, **mask_f}}


def score_confidence(seg_meta: Dict[str, Any], quality: Dict[str, Any], post_gate_passed: bool) -> Dict[str, Any]:
    prob_score = 100.0 * float(seg_meta.get("mean_probability_inside_pred", seg_meta.get("mean_probability_inside_mask", 0.0)) or 0.0)
    quality_score = float(quality.get("quality_score", 0.0))
    gate_score = 100.0 if post_gate_passed else 35.0

    features = (quality.get("features", {}) or {})
    ellipse_iou_score = 100.0 * float(features.get("ellipse_mask_iou", 0.0) or 0.0)

    confidence = 0.35 * prob_score + 0.30 * quality_score + 0.20 * gate_score + 0.15 * ellipse_iou_score
    confidence = float(np.clip(confidence, 0, 100))
    band = "high" if confidence >= 75 else "medium" if confidence >= 50 else "low"
    return {
        "confidence_score": confidence,
        "confidence_band": band,
        "components": {
            "segmentation_probability_score": prob_score,
            "quality_score": quality_score,
            "gate_score": gate_score,
            "ellipse_iou_score": ellipse_iou_score,
        },
    }


def final_decision(input_gate, post_gate, quality: Dict[str, Any], confidence: Dict[str, Any]) -> Dict[str, Any]:
    if not input_gate.passed:
        return {"decision": "reject", "reason_codes": list(input_gate.reason_codes or [])}

    reasons: List[str] = []
    if not post_gate.passed:
        reasons.extend(list(post_gate.reason_codes or []))
    reasons.extend(list(quality.get("warnings", []) or []))
    if quality.get("quality_label") == "poor":
        reasons.append("poor_image_or_mask_quality")
    if confidence.get("confidence_band") == "low":
        reasons.append("low_confidence")

    # De-duplicate while preserving order.
    seen = set()
    reasons = [r for r in reasons if not (r in seen or seen.add(r))]

    reject_codes = {
        "ellipse_fit_failed",
        "mask_too_small",
        "mask_too_large",
        "ellipse_axis_ratio_implausible",
        "ellipse_mask_mismatch_reject",
        "ellipse_outside_image_bounds",
        "ellipse_center_outside_image",
        "poor_image_or_mask_quality",
    }
    if any(r in reject_codes for r in reasons):
        return {"decision": "reject", "reason_codes": reasons}
    if reasons:
        return {"decision": "warn", "reason_codes": reasons}
    return {"decision": "accept", "reason_codes": []}
