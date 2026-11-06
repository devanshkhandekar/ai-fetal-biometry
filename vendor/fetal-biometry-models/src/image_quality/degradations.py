from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw


def _resampling():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def _down_up(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    small = image.resize((nw, nh), _resampling())
    return small.resize((w, h), _resampling())


def _center_crop_restore(image: Image.Image, fraction: float) -> Image.Image:
    w, h = image.size
    dx = int(round(w * fraction))
    dy = int(round(h * fraction))
    if 2 * dx >= w - 8 or 2 * dy >= h - 8:
        return image
    cropped = image.crop((dx, dy, w - dx, h - dy))
    return cropped.resize((w, h), _resampling())


def _gaussian_noise(image: Image.Image, rng: np.random.Generator, sigma: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _shadow(image: Image.Image, rng: np.random.Generator, strength: float, width_fraction: float) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    overlay = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(overlay)

    band_w = max(10, int(round(w * width_fraction)))
    x0 = int(rng.integers(0, max(1, w - band_w)))
    x1 = min(w, x0 + band_w)
    draw.rectangle([x0, 0, x1, h], fill=int(round(255 * strength)))

    arr = np.asarray(image, dtype=np.float32)
    mask = np.asarray(overlay, dtype=np.float32) / 255.0
    factor = 1.0 - 0.75 * mask[..., None]
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def apply_quality_variant(
    image: Image.Image,
    quality_label: str,
    seed: int,
) -> Tuple[Image.Image, str]:
    """
    Deterministically generate a GOOD / REVIEW / POOR proxy-quality image.

    GOOD:
        Original image only.

    REVIEW:
        One mild degradation recipe.

    POOR:
        A stronger combination of degradations.

    Returns
    -------
    image, recipe_name
    """
    label = str(quality_label).strip().lower()
    if label not in {"good", "review", "poor"}:
        raise ValueError(f"Unsupported quality label: {quality_label}")

    rng = np.random.default_rng(int(seed))
    out = image.convert("RGB")

    if label == "good":
        return out, "original"

    if label == "review":
        recipes = [
            "mild_blur",
            "mild_contrast",
            "mild_downsample",
            "mild_noise",
            "mild_crop",
            "mild_shadow",
        ]
        recipe = recipes[int(rng.integers(0, len(recipes)))]

        if recipe == "mild_blur":
            out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.8, 1.5))))
        elif recipe == "mild_contrast":
            out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.68, 0.82)))
        elif recipe == "mild_downsample":
            out = _down_up(out, float(rng.uniform(0.60, 0.78)))
        elif recipe == "mild_noise":
            out = _gaussian_noise(out, rng, sigma=float(rng.uniform(4.0, 9.0)))
        elif recipe == "mild_crop":
            out = _center_crop_restore(out, fraction=float(rng.uniform(0.035, 0.075)))
        elif recipe == "mild_shadow":
            out = _shadow(
                out,
                rng,
                strength=float(rng.uniform(0.18, 0.32)),
                width_fraction=float(rng.uniform(0.08, 0.16)),
            )
        return out, recipe

    # POOR: deliberately combine 2-3 stronger degradations so the class is
    # heterogeneous rather than corresponding to one trivial corruption.
    operations = ["blur", "contrast", "downsample", "noise", "crop", "shadow"]
    rng.shuffle(operations)
    selected = operations[: int(rng.integers(2, 4))]

    for op in selected:
        if op == "blur":
            out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.3, 4.5))))
        elif op == "contrast":
            out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.30, 0.55)))
        elif op == "downsample":
            out = _down_up(out, float(rng.uniform(0.28, 0.48)))
        elif op == "noise":
            out = _gaussian_noise(out, rng, sigma=float(rng.uniform(12.0, 28.0)))
        elif op == "crop":
            out = _center_crop_restore(out, fraction=float(rng.uniform(0.12, 0.22)))
        elif op == "shadow":
            out = _shadow(
                out,
                rng,
                strength=float(rng.uniform(0.55, 0.85)),
                width_fraction=float(rng.uniform(0.18, 0.34)),
            )

    return out, "strong_" + "+".join(selected)
