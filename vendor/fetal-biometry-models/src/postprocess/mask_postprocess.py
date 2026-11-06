from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


def _as_binary(mask: np.ndarray) -> np.ndarray:
    """Convert raw prediction/probability/binary mask to uint8 {0,1}."""
    if mask is None:
        raise ValueError("mask cannot be None")
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # If the mask looks like a probability map, use 0.5. If already 0/255 or 0/1,
    # this still safely converts it to binary.
    if arr.dtype.kind in {"f", "c"}:
        binary = (arr > 0.5).astype(np.uint8)
    else:
        binary = (arr > 0).astype(np.uint8)
    return binary


def _largest_component(binary: np.ndarray, min_area_px: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    details: Dict[str, Any] = {
        "num_components_raw": int(max(num_labels - 1, 0)),
        "largest_component_area_px": 0,
        "largest_component_bbox": None,
        "largest_component_centroid": None,
        "removed_components": int(max(num_labels - 2, 0)),
    }
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    if num_labels <= 1:
        return cleaned, details

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1
    largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    details["largest_component_area_px"] = largest_area
    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    details["largest_component_bbox"] = {"x": x, "y": y, "w": w, "h": h}
    details["largest_component_centroid"] = {
        "x": float(centroids[largest_idx][0]),
        "y": float(centroids[largest_idx][1]),
    }

    if largest_area >= int(min_area_px):
        cleaned[labels == largest_idx] = 1
    return cleaned, details


def clean_mask(
    mask: np.ndarray,
    min_area_px: int = 100,
    close_kernel_size: int = 7,
    open_kernel_size: int = 3,
    fill_holes: bool = True,
    keep_largest: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Create the final measurement mask used for BOTH ellipse fitting and overlay.

    This function intentionally returns a single `final_clean_mask`. The same mask
    must be used downstream for:
      1. red segmentation overlay
      2. contour extraction
      3. ellipse fitting
      4. quality/confidence checks

    That prevents the earlier mismatch where the red overlay could represent a raw
    mask while the ellipse was fitted on a different processed mask.
    """
    raw_binary = _as_binary(mask)
    h, w = raw_binary.shape[:2]

    details: Dict[str, Any] = {
        "input_shape": [int(h), int(w)],
        "raw_area_px": int(raw_binary.sum()),
        "raw_area_ratio": float(raw_binary.sum() / max(h * w, 1)),
        "min_area_px": int(min_area_px),
        "close_kernel_size": int(close_kernel_size),
        "open_kernel_size": int(open_kernel_size),
        "same_mask_used_for_overlay_and_ellipse": True,
    }

    if raw_binary.sum() == 0:
        details.update({
            "num_components_raw": 0,
            "final_area_px": 0,
            "final_area_ratio": 0.0,
            "edge_touch_ratio": 0.0,
            "status": "empty_mask_after_threshold",
        })
        return raw_binary.astype(np.uint8), details

    work = raw_binary.astype(np.uint8)

    # Close small skull-boundary gaps first; then keep the largest connected
    # region. This is useful for ultrasound masks that have small holes/gaps.
    if close_kernel_size and close_kernel_size > 1:
        k = np.ones((int(close_kernel_size), int(close_kernel_size)), dtype=np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)

    if keep_largest:
        work, cc_details = _largest_component(work, min_area_px=min_area_px)
        details.update(cc_details)

    if fill_holes and work.sum() > 0:
        before_fill = int(work.sum())
        work = ndi.binary_fill_holes(work > 0).astype(np.uint8)
        details["hole_fill_added_px"] = int(work.sum()) - before_fill
    else:
        details["hole_fill_added_px"] = 0

    # Small opening removes isolated jagged protrusions without destroying the
    # broad fetal-head region. Keep kernel small to avoid eroding the skull.
    if open_kernel_size and open_kernel_size > 1 and work.sum() > 0:
        k2 = np.ones((int(open_kernel_size), int(open_kernel_size)), dtype=np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, k2, iterations=1)

    # Re-keep largest after morphology, in case opening separated small fragments.
    if keep_largest and work.sum() > 0:
        work, cc2 = _largest_component(work, min_area_px=min_area_px)
        details["num_components_after_morphology"] = cc2.get("num_components_raw", 0)
        details["final_component_bbox"] = cc2.get("largest_component_bbox")
        details["final_component_centroid"] = cc2.get("largest_component_centroid")

    work = (work > 0).astype(np.uint8)
    final_area = int(work.sum())
    edge_pixels = int(work[0, :].sum() + work[-1, :].sum() + work[:, 0].sum() + work[:, -1].sum()) if final_area else 0
    details.update({
        "final_area_px": final_area,
        "final_area_ratio": float(final_area / max(h * w, 1)),
        "edge_touch_pixels": edge_pixels,
        "edge_touch_ratio": float(edge_pixels / max(final_area, 1)),
        "status": "ok" if final_area >= int(min_area_px) else "too_small_after_cleanup",
    })
    return work.astype(np.uint8), details
