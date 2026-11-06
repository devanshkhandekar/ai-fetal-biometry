from dataclasses import dataclass
from typing import Dict, Tuple
import cv2
import numpy as np
import torch


@dataclass
class PreprocessResult:
    image_resized: np.ndarray
    tensor: torch.Tensor
    metadata: Dict


def normalize_grayscale(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1)
    return img.astype(np.float32)


def preprocess_for_model(img: np.ndarray, image_size: int = 256) -> PreprocessResult:
    original_shape = img.shape[:2]
    img_norm = normalize_grayscale(img)
    resized = cv2.resize(img_norm, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized[None, None, ...]).float()  # [1,1,H,W]
    meta = {
        'original_shape': original_shape,
        'model_shape': (image_size, image_size),
        'scale_y': image_size / original_shape[0],
        'scale_x': image_size / original_shape[1],
    }
    return PreprocessResult(image_resized=resized, tensor=tensor, metadata=meta)


def resize_mask_to_original(mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
