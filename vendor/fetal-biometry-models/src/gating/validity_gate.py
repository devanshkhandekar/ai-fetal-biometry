from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class GateResult:
    passed: bool
    reason_codes: List[str]
    details: Dict[str, Any]


def input_validity_gate(img: np.ndarray) -> GateResult:
    reasons: List[str] = []
    details: Dict[str, Any] = {}

    if img is None or img.size == 0:
        return GateResult(False, ["empty_image"], {})

    h, w = img.shape[:2]
    details["shape"] = [int(h), int(w)]
    if h < 64 or w < 64:
        reasons.append("suspiciously_small_dimensions")

    finite = np.isfinite(img).all()
    details["finite"] = bool(finite)
    if not finite:
        reasons.append("non_finite_pixels")

    std = float(np.std(img))
    ptp = float(np.ptp(img))
    details.update({"std": std, "range": ptp})
    if std < 2 or ptp < 10:
        reasons.append("very_low_contrast_or_blank")

    return GateResult(len(reasons) == 0, reasons, details)


def post_segmentation_gate(mask: np.ndarray, ellipse: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> GateResult:
    """Post-segmentation safety gate for accepting/reviewing/rejecting measurements."""
    reasons: List[str] = []
    h, w = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8)
    area_px = int(binary.sum())
    area_ratio = float(area_px / max(h * w, 1))
    edge_touch = int(binary[0, :].sum() + binary[-1, :].sum() + binary[:, 0].sum() + binary[:, -1].sum()) if area_px else 0
    edge_touch_ratio = float(edge_touch / max(area_px, 1))

    details: Dict[str, Any] = {
        "mask_area_px": area_px,
        "mask_area_ratio": area_ratio,
        "edge_touch_pixels": edge_touch,
        "edge_touch_ratio": edge_touch_ratio,
    }

    min_area_ratio = float(cfg.get("min_mask_area_ratio", 0.01))
    max_area_ratio = float(cfg.get("max_mask_area_ratio", 0.65))
    if area_ratio < min_area_ratio:
        reasons.append("mask_too_small")
    if area_ratio > max_area_ratio:
        reasons.append("mask_too_large")
    if edge_touch_ratio > float(cfg.get("edge_touch_ratio_max", 0.025)):
        reasons.append("head_mask_touches_image_edge")

    if ellipse is None:
        reasons.append("ellipse_fit_failed")
    else:
        major = float(ellipse.get("major_diameter_px", ellipse.get("major_axis_px", 0.0)))
        minor = float(ellipse.get("minor_diameter_px", ellipse.get("minor_axis_px", 0.0)))
        ratio = major / max(minor, 1e-6)
        iou = float(ellipse.get("ellipse_mask_iou", 0.0))
        center_offset_norm = float(ellipse.get("center_offset_norm", 0.0))
        bounds = ellipse.get("bounds", {}) or {}
        outside_fraction = float(bounds.get("outside_fraction", 0.0))

        details.update({
            "axis_ratio": ratio,
            "ellipse_mask_iou": iou,
            "center_offset_norm": center_offset_norm,
            "ellipse_outside_fraction": outside_fraction,
            "ellipse_fully_inside_image": bool(bounds.get("fully_inside_image", True)),
        })

        # Review/reject thresholds. These are deliberately stricter than the old
        # MVP because wrong HC/GA should not silently pass.
        if ratio < float(cfg.get("axis_ratio_min", 1.02)):
            reasons.append("ellipse_too_circular_or_unstable")
        if ratio > float(cfg.get("axis_ratio_review_max", 1.80)):
            reasons.append("ellipse_axis_ratio_review")
        if ratio > float(cfg.get("axis_ratio_reject_max", 2.20)):
            reasons.append("ellipse_axis_ratio_implausible")

        if iou < float(cfg.get("ellipse_mask_iou_review_min", 0.65)):
            reasons.append("ellipse_mask_mismatch_review")
        if iou < float(cfg.get("ellipse_mask_iou_reject_min", 0.50)):
            reasons.append("ellipse_mask_mismatch_reject")

        if center_offset_norm > float(cfg.get("ellipse_center_offset_review_max", 0.08)):
            reasons.append("ellipse_center_offset_high")

        if outside_fraction > float(cfg.get("ellipse_outside_fraction_max", 0.02)):
            reasons.append("ellipse_outside_image_bounds")

        cx, cy = float(ellipse.get("center_x", -1)), float(ellipse.get("center_y", -1))
        if not (0 <= cx < w and 0 <= cy < h):
            reasons.append("ellipse_center_outside_image")

    return GateResult(len(reasons) == 0, reasons, details)
