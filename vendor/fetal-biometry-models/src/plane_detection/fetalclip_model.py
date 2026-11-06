from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import open_clip
from PIL import Image, ImageOps

from .prompts import APPLICATION_MAPPING, PROMPT_ENSEMBLES, STANDARD_CLASSES


def _square_pad(image: Image.Image, fill=0) -> Image.Image:
    """
    Symmetrically zero-pad a rectangular image to square.
    The FetalCLIP paper describes square-padding before resizing to 224x224.
    """
    image = image.convert("RGB")
    w, h = image.size
    if w == h:
        return image
    size = max(w, h)
    left = (size - w) // 2
    right = size - w - left
    top = (size - h) // 2
    bottom = size - h - top
    return ImageOps.expand(image, border=(left, top, right, bottom), fill=fill)


class FetalCLIPPlaneDetector:
    """
    Zero-shot FetalCLIP detector for five standard fetal ultrasound views.

    "Other" is intentionally NOT represented by a text prompt in the default
    zero-shot classifier. Instead, Other images are used to validate open-set
    rejection through score + margin thresholds.
    """

    def __init__(
        self,
        config_path: str | Path,
        weights_path: str | Path,
        device: Optional[str] = None,
        score_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        prompt_ensembles: Optional[Dict[str, Sequence[str]]] = None,
        logit_scale: float = 100.0,
        square_pad: bool = True,
    ):
        self.config_path = Path(config_path)
        self.weights_path = Path(weights_path)
        if not self.config_path.exists():
            raise FileNotFoundError(self.config_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(self.weights_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.score_threshold = score_threshold
        self.margin_threshold = margin_threshold
        self.logit_scale = float(logit_scale)
        self.square_pad = bool(square_pad)

        with self.config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)

        # Matches the loading pattern documented by the official FetalCLIP repo
        # for its pinned OpenCLIP version.
        open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config

        self.model, _, self.preprocess_test = open_clip.create_model_and_transforms(
            "FetalCLIP",
            pretrained=str(self.weights_path),
        )
        self.tokenizer = open_clip.get_tokenizer("FetalCLIP")
        self.model.eval().to(self.device)

        self.prompt_ensembles = prompt_ensembles or PROMPT_ENSEMBLES
        self.classes = tuple(STANDARD_CLASSES)

        missing = [c for c in self.classes if c not in self.prompt_ensembles]
        if missing:
            raise ValueError(f"Missing prompts for classes: {missing}")

        self.text_features = self._build_text_features()

    def _amp_context(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()

    @torch.no_grad()
    def _build_text_features(self) -> torch.Tensor:
        """
        Encode each prompt, average normalized prompt embeddings per class,
        then re-normalize the resulting class embedding.
        """
        class_features = []
        for class_name in self.classes:
            prompts = list(self.prompt_ensembles[class_name])
            tokens = self.tokenizer(prompts).to(self.device)
            with self._amp_context():
                feats = self.model.encode_text(tokens)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            class_feat = feats.mean(dim=0)
            class_feat = class_feat / class_feat.norm().clamp_min(1e-12)
            class_features.append(class_feat)
        return torch.stack(class_features, dim=0).to(self.device)

    def _load_and_preprocess(self, image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                pil = im.convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if self.square_pad:
            pil = _square_pad(pil, fill=0)
        return self.preprocess_test(pil)

    @torch.no_grad()
    def encode_images(
        self,
        images: Sequence[str | Path | Image.Image],
        batch_size: int = 8,
    ) -> np.ndarray:
        all_features = []
        for start in range(0, len(images), batch_size):
            batch_items = images[start:start + batch_size]
            batch = torch.stack([self._load_and_preprocess(x) for x in batch_items]).to(self.device)
            with self._amp_context():
                feats = self.model.encode_image(batch)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_features.append(feats.cpu().numpy())
        if not all_features:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def predict_batch(
        self,
        images: Sequence[str | Path | Image.Image],
        batch_size: int = 8,
    ) -> List[dict]:
        results: List[dict] = []

        for start in range(0, len(images), batch_size):
            batch_items = images[start:start + batch_size]
            batch = torch.stack([self._load_and_preprocess(x) for x in batch_items]).to(self.device)

            with self._amp_context():
                image_features = self.model.encode_image(batch)
            image_features = image_features.float()
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

            logits = self.logit_scale * image_features @ self.text_features.T
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                order = np.argsort(row)[::-1]
                top_idx = int(order[0])
                second_idx = int(order[1])
                top_class = self.classes[top_idx]
                second_class = self.classes[second_idx]
                top_score = float(row[top_idx])
                second_score = float(row[second_idx])
                margin = top_score - second_score

                status = self._policy_status(top_score, margin)
                application_plane = (
                    APPLICATION_MAPPING[top_class] if status == "accept"
                    else "unknown"
                )

                results.append({
                    "predicted_plane": top_class,
                    "application_plane": application_plane,
                    "top_score": top_score,
                    "second_plane": second_class,
                    "second_score": second_score,
                    "margin": float(margin),
                    "status": status,
                    "routing_enabled": False,
                    "scores": {
                        class_name: float(row[i])
                        for i, class_name in enumerate(self.classes)
                    },
                })

        return results

    def predict(self, image) -> dict:
        return self.predict_batch([image], batch_size=1)[0]

    def _policy_status(self, top_score: float, margin: float) -> str:
        # Before thresholds are tuned, raw predictions are marked "unconfigured"
        # instead of silently pretending they are accepted.
        if self.score_threshold is None or self.margin_threshold is None:
            return "unconfigured"
        if top_score < self.score_threshold:
            return "review"
        if margin < self.margin_threshold:
            return "review"
        return "accept"
