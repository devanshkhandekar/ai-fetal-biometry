from pathlib import Path
from typing import Dict, Tuple
import torch
import numpy as np

from src.segmentation.model import build_model
from src.preprocessing.preprocess import preprocess_for_model, resize_mask_to_original


def load_checkpoint(checkpoint_path: str | Path, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model, image_np: np.ndarray, image_size=256, threshold=0.5, device=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    device = device or next(model.parameters()).device
    prep = preprocess_for_model(image_np, image_size=image_size)
    logits = model(prep.tensor.to(device))
    prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    prob = resize_mask_to_original(prob_small, prep.metadata['original_shape'])
    mask = (prob >= threshold).astype(np.uint8)
    meta = {
        'mean_probability_inside_pred': float(prob[mask > 0].mean()) if mask.sum() else 0.0,
        'max_probability': float(prob.max()),
        'threshold': float(threshold),
    }
    return mask, prob, meta
