from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from .fetalclip_model import FetalCLIPPlaneDetector
from .prompts import APPLICATION_MAPPING


class FetalCLIPLinearProbeDetector:
    """
    Optional fallback if zero-shot + rejection thresholds are not good enough.

    The probe is trained on frozen FetalCLIP embeddings, including the explicit
    FETAL_PLANES_DB "other" class.
    """

    def __init__(
        self,
        fetalclip: FetalCLIPPlaneDetector,
        probe_path: str | Path,
        score_threshold: float | None = None,
    ):
        self.fetalclip = fetalclip
        self.probe = joblib.load(probe_path)
        self.score_threshold = score_threshold

    def predict(self, image):
        x = self.fetalclip.encode_images([image], batch_size=1)
        probs = self.probe.predict_proba(x)[0]
        classes = [str(c) for c in self.probe.classes_]
        order = np.argsort(probs)[::-1]
        top_i, second_i = int(order[0]), int(order[1])
        top_class = classes[top_i]
        top_score = float(probs[top_i])
        second_class = classes[second_i]
        second_score = float(probs[second_i])

        status = "accept"
        if self.score_threshold is not None and top_score < self.score_threshold:
            status = "review"

        if status == "review":
            application_plane = "unknown"
        elif top_class == "other":
            application_plane = "unknown"
        else:
            application_plane = APPLICATION_MAPPING.get(top_class, "unsupported")

        return {
            "mode": "linear_probe",
            "predicted_plane": top_class,
            "application_plane": application_plane,
            "top_score": top_score,
            "second_plane": second_class,
            "second_score": second_score,
            "margin": top_score - second_score,
            "status": status,
            "routing_enabled": False,
            "scores": {classes[i]: float(probs[i]) for i in range(len(classes))},
        }
