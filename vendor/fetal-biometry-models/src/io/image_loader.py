from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
import numpy as np


def load_image(path: str | Path, pixel_spacing_mm: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Image not found: {path}')

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Could not read image: {path}')

    meta = {
        'source_path': str(path),
        'filename': path.name,
        'format': path.suffix.lower().replace('.', ''),
        'original_shape': tuple(img.shape),
        'pixel_spacing_mm': pixel_spacing_mm,
        'has_pixel_spacing': pixel_spacing_mm is not None,
    }
    return img, meta
