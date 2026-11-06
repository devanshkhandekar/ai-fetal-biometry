from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 180)."""
    angle = float(angle) % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def ellipse_to_mask(ellipse: Dict[str, Any], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize fitted ellipse into a uint8 {0,1} mask."""
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    if not ellipse:
        return out
    center = (int(round(float(ellipse["center_x"]))), int(round(float(ellipse["center_y"]))))
    axes = (
        max(1, int(round(float(ellipse["semi_major_axis_px"])))),
        max(1, int(round(float(ellipse["semi_minor_axis_px"])))),
    )
    cv2.ellipse(out, center, axes, float(ellipse["angle_deg"]), 0, 360, 1, -1)
    return out


def ellipse_iou(mask: np.ndarray, ellipse: Dict[str, Any]) -> float:
    """IoU between final clean segmentation mask and the fitted ellipse area."""
    binary = (mask > 0).astype(np.uint8)
    e_mask = ellipse_to_mask(ellipse, binary.shape)
    inter = int(np.logical_and(binary, e_mask).sum())
    union = int(np.logical_or(binary, e_mask).sum())
    return float(inter / max(union, 1))


def ellipse_bounds(ellipse: Dict[str, Any], shape: Tuple[int, int], sample_points: int = 360) -> Dict[str, Any]:
    pts = ellipse_points(ellipse, n=sample_points)
    h, w = shape[:2]
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    outside = int(((pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)).sum())
    return {
        "min_x": int(min_x),
        "min_y": int(min_y),
        "max_x": int(max_x),
        "max_y": int(max_y),
        "outside_sample_points": outside,
        "outside_fraction": float(outside / max(len(pts), 1)),
        "fully_inside_image": bool(outside == 0),
    }


def _mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    m = cv2.moments((mask > 0).astype(np.uint8))
    if abs(m["m00"]) < 1e-6:
        return None
    return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])


def fit_head_ellipse(mask: np.ndarray, image_shape: Optional[Tuple[int, int]] = None, min_contour_points: int = 20) -> Optional[Dict[str, Any]]:
    """Fit a clinically safer ellipse to the final cleaned head mask.

    Key fixes included:
      - Uses largest external contour only.
      - Correctly handles OpenCV axis/angle convention.
      - Stores both full diameters and semi-axes to avoid measurement confusion.
      - Computes ellipse-mask IoU and center offset for quality gating.
      - Reports whether the ellipse extends outside the image.

    OpenCV returns ((cx, cy), (w, h), angle), where w/h are FULL diameters
    and `angle` is tied to the width axis. If h > w, the major axis is the
    height axis, so the major-axis angle must be angle + 90 degrees.
    """
    if mask is None:
        return None
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.ndim != 2 or binary.sum() == 0:
        return None

    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    if len(contour) < max(5, int(min_contour_points)) or contour_area <= 0:
        return None

    (cx, cy), (raw_w, raw_h), raw_angle = cv2.fitEllipse(contour)
    raw_w = float(raw_w)
    raw_h = float(raw_h)
    raw_angle = float(raw_angle)

    if raw_w <= 0 or raw_h <= 0:
        return None

    # Correct axis-angle handling. This is the key fix for ellipses that looked
    # wrongly vertical/narrow despite reasonable masks.
    if raw_w >= raw_h:
        major_diameter = raw_w
        minor_diameter = raw_h
        major_angle = raw_angle
    else:
        major_diameter = raw_h
        minor_diameter = raw_w
        major_angle = raw_angle + 90.0
    major_angle = _normalize_angle(major_angle)

    semi_major = major_diameter / 2.0
    semi_minor = minor_diameter / 2.0
    axis_ratio = float(major_diameter / max(minor_diameter, 1e-6))

    ellipse: Dict[str, Any] = {
        "center_x": float(cx),
        "center_y": float(cy),
        # New explicit names.
        "major_diameter_px": float(major_diameter),
        "minor_diameter_px": float(minor_diameter),
        "semi_major_axis_px": float(semi_major),
        "semi_minor_axis_px": float(semi_minor),
        # Backward-compatible aliases used by older measurement/dashboard code.
        "major_axis_px": float(major_diameter),
        "minor_axis_px": float(minor_diameter),
        "angle_deg": float(major_angle),
        "axis_ratio": axis_ratio,
        "contour_area_px": contour_area,
        "mask_area_px": int(binary.sum()),
        "raw_cv2_width_px": raw_w,
        "raw_cv2_height_px": raw_h,
        "raw_cv2_angle_deg": raw_angle,
        "angle_convention": "angle_deg is corrected major-axis angle; axes are full diameters unless named semi_*",
        "fit_contour_points": int(len(contour)),
    }

    ellipse["ellipse_area_px2"] = float(np.pi * semi_major * semi_minor)
    ellipse["ellipse_mask_iou"] = ellipse_iou(binary, ellipse)

    mc = _mask_centroid(binary)
    if mc is not None:
        dx = float(cx - mc[0])
        dy = float(cy - mc[1])
        diag = float(np.sqrt(binary.shape[0] ** 2 + binary.shape[1] ** 2))
        ellipse["mask_centroid_x"] = float(mc[0])
        ellipse["mask_centroid_y"] = float(mc[1])
        ellipse["center_offset_px"] = float(np.sqrt(dx * dx + dy * dy))
        ellipse["center_offset_norm"] = float(ellipse["center_offset_px"] / max(diag, 1e-6))

    if image_shape is None:
        image_shape = binary.shape
    ellipse["bounds"] = ellipse_bounds(ellipse, image_shape)
    return ellipse


def ellipse_points(ellipse: Dict[str, Any], n: int = 200) -> np.ndarray:
    """Return sampled ellipse perimeter points based on corrected major-axis angle."""
    a = float(ellipse.get("semi_major_axis_px", ellipse.get("major_axis_px", 0) / 2.0))
    b = float(ellipse.get("semi_minor_axis_px", ellipse.get("minor_axis_px", 0) / 2.0))
    theta = np.deg2rad(float(ellipse.get("angle_deg", 0.0)))
    t = np.linspace(0, 2 * np.pi, int(n))
    x = a * np.cos(t)
    y = b * np.sin(t)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = np.vstack([x, y]).T @ R.T
    pts[:, 0] += float(ellipse["center_x"])
    pts[:, 1] += float(ellipse["center_y"])
    return pts.astype(np.int32)
