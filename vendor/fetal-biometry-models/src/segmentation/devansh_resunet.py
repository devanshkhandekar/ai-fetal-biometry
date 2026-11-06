"""
Adapter for pretrained static fetal-head segmentation checkpoints.

Supported static checkpoints:
- phase0_model.pth: static Residual U-Net + deep supervision; some released checkpoints use custom/pruned channel counts
- 4a_best_pruned_ft_v10.pth: pruned static Residual U-Net

The temporal checkpoints are intentionally not used for the MVP single-image HC18 flow,
because the current MVP input is one ultrasound frame, not a 16-frame cine clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

INPUT_H = 256
INPUT_W = 384
IMG_MEAN = 0.2
IMG_STD = 0.15

_BASE_CHANNELS = {
    "enc1": 32,
    "enc2": 64,
    "enc3": 128,
    "enc4": 256,
    "bottleneck": 512,
    "dec4": 256,
    "dec3": 128,
    "dec2": 64,
    "dec1": 32,
}


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class ResidualUNetDS(nn.Module):
    """Static Residual U-Net used by the phase0 pretrained checkpoint."""

    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.enc1 = ResidualBlock(in_ch, b)
        self.enc2 = ResidualBlock(b, b * 2)
        self.enc3 = ResidualBlock(b * 2, b * 4)
        self.enc4 = ResidualBlock(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(b * 8, b * 16)
        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
        self.dec4 = ResidualBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.dec3 = ResidualBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ResidualBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ResidualBlock(b * 2, b)
        self.final = nn.Conv2d(b, 1, 1)
        self.aux_d3 = nn.Conv2d(b * 4, 1, 1)
        self.aux_d2 = nn.Conv2d(b * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class PrunedResidualBlock(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class PrunedResidualUNetDS(nn.Module):
    """Static pruned Residual U-Net used by 4a_best_pruned_ft_v10.pth."""

    def __init__(self, channel_counts: Dict[str, int], in_ch: int = 1, base_ch: int = 32, aux_d3_ch: Optional[int] = None):
        super().__init__()
        b = base_ch
        cc = {**_BASE_CHANNELS, **channel_counts}
        c3, c4, cbn, cd4, cd3 = cc["enc3"], cc["enc4"], cc["bottleneck"], cc["dec4"], cc["dec3"]
        if aux_d3_ch is None:
            aux_d3_ch = cd3

        self.enc1 = ResidualBlock(in_ch, b)
        self.enc2 = ResidualBlock(b, b * 2)
        self.pool = nn.MaxPool2d(2)
        self.enc3 = PrunedResidualBlock(b * 2, b * 4, c3)
        self.enc4 = PrunedResidualBlock(c3, b * 8, c4)
        self.bottleneck = PrunedResidualBlock(c4, b * 16, cbn)
        self.up4 = nn.ConvTranspose2d(cbn, b * 8, 2, stride=2)
        self.dec4 = PrunedResidualBlock(b * 8 + c4, b * 8, cd4)
        self.up3 = nn.ConvTranspose2d(cd4, b * 4, 2, stride=2)
        self.dec3 = PrunedResidualBlock(b * 4 + c3, b * 4, cd3)
        self.up2 = nn.ConvTranspose2d(cd3, b * 2, 2, stride=2)
        self.dec2 = ResidualBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ResidualBlock(b * 2, b)
        self.final = nn.Conv2d(b, 1, 1)
        self.aux_d3 = nn.Conv2d(aux_d3_ch, 1, 1)
        self.aux_d2 = nn.Conv2d(b * 2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


@dataclass
class DevanshModelBundle:
    model: nn.Module
    model_variant: str
    input_size: Tuple[int, int] = (INPUT_H, INPUT_W)
    mean: float = IMG_MEAN
    std: float = IMG_STD
    device: str = "cpu"


def _state_dict_from_checkpoint(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        # fallback for raw state_dict style checkpoints
        return ckpt
    raise ValueError("Unsupported checkpoint format. Expected dict or {'model_state_dict': ...}.")



def _infer_channel_counts_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Infer custom/pruned channel counts from checkpoint tensor shapes.

    The released static checkpoints are not always a plain [32,64,128,256,512]
    ResidualUNetDS. Some contain odd channel counts such as 71/129/257.
    This function reads those counts directly from the state_dict so the loader
    does not depend on a hardcoded architecture assumption.
    """
    def out_ch(name: str, fallback: int) -> int:
        w = sd.get(name, None)
        if w is None:
            w = sd.get(f"backbone.{name}", None)
        return int(w.shape[0]) if w is not None else fallback

    return {
        "enc3": out_ch("enc3.skip.weight", _BASE_CHANNELS["enc3"]),
        "enc4": out_ch("enc4.skip.weight", _BASE_CHANNELS["enc4"]),
        "bottleneck": out_ch("bottleneck.skip.weight", _BASE_CHANNELS["bottleneck"]),
        "dec4": out_ch("dec4.skip.weight", _BASE_CHANNELS["dec4"]),
        "dec3": out_ch("dec3.skip.weight", _BASE_CHANNELS["dec3"]),
        "dec2": _BASE_CHANNELS["dec2"],
        "dec1": _BASE_CHANNELS["dec1"],
    }


def _looks_like_plain_phase0(sd: Dict[str, torch.Tensor]) -> bool:
    """Return True only when checkpoint matches the full-width base_ch=32 model."""
    checks = {
        "enc3.skip.weight": (128, 64, 1, 1),
        "enc4.skip.weight": (256, 128, 1, 1),
        "bottleneck.skip.weight": (512, 256, 1, 1),
        "dec4.skip.weight": (256, 512, 1, 1),
        "dec3.skip.weight": (128, 256, 1, 1),
        "aux_d3.weight": (1, 128, 1, 1),
    }
    for name, expected_shape in checks.items():
        w = sd.get(name, None)
        if w is None:
            w = sd.get(f"backbone.{name}", None)
        if w is None or tuple(w.shape) != expected_shape:
            return False
    return True


def _get_aux_d3_channels(sd: Dict[str, torch.Tensor], fallback: int) -> int:
    # Do NOT use Python `or` with tensors. PyTorch tensors cannot be
    # evaluated as booleans and will raise: "Boolean value of Tensor...".
    aux_w = sd.get("aux_d3.weight", None)
    if aux_w is None:
        aux_w = sd.get("backbone.aux_d3.weight", None)
    return int(aux_w.shape[1]) if aux_w is not None else fallback

def load_devansh_static_checkpoint(
    checkpoint_path: str | Path,
    variant: Literal["phase0", "phase4a"] = "phase4a",
    device: Optional[str] = None,
) -> DevanshModelBundle:
    """Load a pretrained static checkpoint from the static fetal-head model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    sd = _state_dict_from_checkpoint(ckpt)

    if variant == "phase0":
        # Some downloaded "phase0" files are actually custom-channel/pruned
        # static checkpoints. If tensor shapes are not the plain full-width
        # ResidualUNetDS, instantiate the pruned/custom architecture inferred
        # from the checkpoint itself.
        if _looks_like_plain_phase0(sd):
            model = ResidualUNetDS(in_ch=1, base_ch=32)
        else:
            channel_counts = _infer_channel_counts_from_state_dict(sd)
            aux_d3_ch = _get_aux_d3_channels(sd, channel_counts.get("dec3", 128))
            model = PrunedResidualUNetDS(
                channel_counts=channel_counts,
                in_ch=1,
                base_ch=32,
                aux_d3_ch=aux_d3_ch,
            )
    elif variant == "phase4a":
        channel_counts = ckpt.get("channel_counts", None) if isinstance(ckpt, dict) else None
        if not channel_counts:
            channel_counts = _infer_channel_counts_from_state_dict(sd)
        aux_d3_ch = _get_aux_d3_channels(sd, channel_counts.get("dec3", 128))
        model = PrunedResidualUNetDS(channel_counts=channel_counts, in_ch=1, base_ch=32, aux_d3_ch=aux_d3_ch)
    else:
        raise ValueError(f"Unsupported static variant: {variant}")

    # strict=True is intentional after creating the correct architecture: it catches
    # real checkpoint/model mismatches instead of silently skipping layers.
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return DevanshModelBundle(model=model, model_variant=variant, device=device)


def preprocess_devansh(img_gray: np.ndarray) -> torch.Tensor:
    """static fetal-head model preprocessing: resize to 384x256, scale to [0,1], normalize by mean/std."""
    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray.astype(np.uint8)
    resized = cv2.resize(img_gray, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - IMG_MEAN) / (IMG_STD + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_devansh_mask(
    bundle: DevanshModelBundle,
    image_np: np.ndarray,
    threshold: float = 0.5,
    resize_to_original: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return binary mask, probability map, and segmentation metadata.

    Probability/mask are resized back to the original image size by default so our
    downstream post-processing, ellipse fitting, and measurements remain consistent.
    """
    orig_shape = image_np.shape[:2]
    x = preprocess_devansh(image_np).to(bundle.device)
    logits = bundle.model(x)
    prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

    if resize_to_original:
        prob = cv2.resize(prob_small, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        prob = prob_small

    mask = (prob >= threshold).astype(np.uint8)
    meta = {
        "backend": "devansh_fetal_head_biometry_ai",
        "model_variant": bundle.model_variant,
        "input_size_hw": [INPUT_H, INPUT_W],
        "threshold": float(threshold),
        "max_probability": float(prob.max()),
        "mean_probability_inside_pred": float(prob[mask > 0].mean()) if mask.sum() else 0.0,
        "predicted_area_px": int(mask.sum()),
    }
    return mask, prob, meta
