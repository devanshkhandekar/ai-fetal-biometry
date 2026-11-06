from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


def to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.shape[2] == 3:
        return arr
    return cv2.cvtColor(arr[..., 0], cv2.COLOR_GRAY2RGB)


def draw_mask_overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Draw the final cleaned mask used for ellipse fitting."""
    rgb = to_rgb_uint8(img)
    overlay = rgb.copy()
    if mask is not None:
        overlay[(mask > 0)] = [255, 0, 0]
    return cv2.addWeighted(overlay, alpha, rgb, 1 - alpha, 0)


def draw_ellipse_overlay(img: np.ndarray, mask: Optional[np.ndarray], ellipse: Optional[Dict], alpha: float = 0.28) -> np.ndarray:
    """Draw cleaned red mask and corrected green fitted ellipse.

    The ellipse fields use corrected major-axis convention from ellipse_fit.py.
    """
    out = draw_mask_overlay(img, mask, alpha=alpha) if mask is not None else to_rgb_uint8(img)
    if ellipse is not None:
        center = (int(round(float(ellipse["center_x"]))), int(round(float(ellipse["center_y"]))))
        axes = (
            max(1, int(round(float(ellipse.get("semi_major_axis_px", ellipse.get("major_axis_px", 0) / 2.0))))),
            max(1, int(round(float(ellipse.get("semi_minor_axis_px", ellipse.get("minor_axis_px", 0) / 2.0))))),
        )
        angle = float(ellipse.get("angle_deg", 0.0))
        cv2.ellipse(out, center, axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.circle(out, center, 3, (255, 255, 0), -1)
    return out


def save_overlay(path: str | Path, overlay_rgb: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return str(path)
