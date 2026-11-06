from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.biometry.estimated_fetal_weight import estimate_fetal_weight_safe
from src.orchestration.scan_input import ScanInput
from src.pipeline_devansh import run_case_devansh
from src.segmentation.devansh_resunet import load_devansh_static_checkpoint
from src.plane_detection.fetalclip_model import FetalCLIPPlaneDetector
from src.models.landmarks.biometrynet_subprocess import BiometryNetSubprocess

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_QUALITY_SCOPE = {
    "head": {
        "policy_applicable": True,
        "affects_routing": True,
        "affects_fusion": True,
        "scope": "head_proxy_policy",
    },

    "abdomen": {
        "policy_applicable": False,
        "affects_routing": False,
        "affects_fusion": False,
        "scope": "advisory_only_unvalidated_for_abdomen",
    },

    "femur": {
        "policy_applicable": False,
        "affects_routing": False,
        "affects_fusion": False,
        "scope": "advisory_only_unvalidated_for_femur",
    },

    "unsupported": {
        "policy_applicable": False,
        "affects_routing": False,
        "affects_fusion": False,
        "scope": "unsupported",
    },

    "unknown": {
        "policy_applicable": False,
        "affects_routing": False,
        "affects_fusion": False,
        "scope": "unknown",
    },
}

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json_first(paths) -> Dict[str, Any]:
    checked = []
    for raw in paths:
        p = _resolve(raw)
        checked.append(str(p))
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError("None of the policy files exist:\n  " + "\n  ".join(checked))


def _normalized_confidence(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    if v > 1.0:
        v = v / 100.0
    return float(max(0.0, min(1.0, v)))


class ProjectBackend:
    def __init__(self, config_path: str | Path = "configs/project_backend.json"):
        cfg = json.loads(_resolve(config_path).read_text(encoding="utf-8"))
        self.cfg = cfg

        fcfg = cfg["fetalclip"]
        self.fetalclip = FetalCLIPPlaneDetector(
            config_path=_resolve(fcfg["config"]),
            weights_path=_resolve(fcfg["weights"]),
        )

        self.plane_probe = joblib.load(_resolve(cfg["plane"]["probe"]))
        self.plane_policy = _read_json_first(cfg["plane"]["policy_candidates"])

        self.quality_probe = joblib.load(_resolve(cfg["quality"]["probe"]))
        self.quality_policy = _read_json_first(cfg["quality"]["policy_candidates"])

        hcfg = cfg["head"]
        self.head_checkpoint = _resolve(hcfg["checkpoint"])
        self.head_variant = str(hcfg["variant"])
        self.head_threshold = float(hcfg["mask_threshold"])
        self.head_output_dir = _resolve(hcfg["output_dir"])

        if not self.head_checkpoint.exists():
            raise FileNotFoundError(self.head_checkpoint)

        self.head_model_bundle = load_devansh_static_checkpoint(
            str(self.head_checkpoint),
            variant=self.head_variant,
        )
        self.biometrynet = BiometryNetSubprocess(config_path="configs/biometrynet_bridge.json")

    @staticmethod
    def _path(scan):
        p = scan.path if isinstance(scan, ScanInput) else Path(scan)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    @staticmethod
    def _spacing(scan):
        return scan.pixel_spacing_mm if isinstance(scan, ScanInput) else None

    def quality_scope_for(
        self,
        application_plane: str | None,
    ) -> Dict[str, Any]:

        anatomy = str(
            application_plane or "unknown"
        ).strip().lower()

        scope = dict(
            DEFAULT_QUALITY_SCOPE.get(
                anatomy,
                DEFAULT_QUALITY_SCOPE["unknown"],
            )
        )

        # Optional override from project_backend.json.
        configured = (
            self.cfg.get("quality_scope")
            or {}
        )

        if isinstance(configured, dict):

            override = configured.get(
                anatomy
            )

            if isinstance(override, dict):
                scope.update(override)

        scope["anatomy"] = anatomy

        scope.setdefault(
            "policy_applicable",
            False,
        )

        scope.setdefault(
            "affects_routing",
            False,
        )

        scope.setdefault(
            "affects_fusion",
            False,
        )

        scope.setdefault(
            "scope",
            "advisory_only",
        )

        return scope


    def annotate_quality_scope(
        self,
        quality: Dict[str, Any],
        application_plane: str | None,
    ) -> Dict[str, Any]:

        result = dict(
            quality or {}
        )

        scope = self.quality_scope_for(
            application_plane
        )

        result[
            "policy_applicable"
        ] = bool(
            scope["policy_applicable"]
        )

        result[
            "affects_routing"
        ] = bool(
            scope["affects_routing"]
        )

        result[
            "affects_fusion"
        ] = bool(
            scope["affects_fusion"]
        )

        result["scope"] = scope[
            "scope"
        ]

        result[
            "scope_anatomy"
        ] = scope[
            "anatomy"
        ]

        # Important: current IQA labels are not
        # sonographer clinical ground truth.
        result.setdefault(
            "clinical_ground_truth",
            False,
        )

        if not result[
            "policy_applicable"
        ]:

            result[
                "scope_note"
            ] = (
                "Image-quality prediction is "
                "advisory for this anatomy and "
                "does not block measurement "
                "acceptance in the current release."
            )

        return result

    def _encode_one(self, scan):
        path = self._path(scan)
        emb = self.fetalclip.encode_images([path], batch_size=1)
        if emb.shape[0] != 1:
            raise RuntimeError(f"Expected one FetalCLIP embedding, got {emb.shape}")
        return emb

    def detect_plane(self, scan):
        X = self._encode_one(scan)
        probs = self.plane_probe.predict_proba(X)[0]
        classes = [str(c) for c in self.plane_probe.classes_]
        order = np.argsort(probs)[::-1]
        a, b = int(order[0]), int(order[1])

        pred = classes[a]
        top = float(probs[a])
        second = float(probs[b])
        margin = top - second

        score_t = float(self.plane_policy.get("score_threshold", 0.5))
        margin_t = float(self.plane_policy.get("margin_threshold", 0.0))

        accepted = top >= score_t and margin >= margin_t
        application = {
            "brain": "head",
            "abdomen": "abdomen",
            "femur": "femur",
            "thorax": "unsupported",
            "cervix": "unsupported",
            "other": "unknown",
        }.get(pred, "unknown")

        status = "accept" if accepted and application in {"head","abdomen","femur"} else "review"

        return {
            "predicted_plane": pred,
            "application_plane": application,
            "status": status,
            "top_score": top,
            "second_plane": classes[b],
            "second_score": second,
            "margin": margin,
            "score_threshold": score_t,
            "margin_threshold": margin_t,
            "model": "FetalCLIP + six-class linear probe",
        }

    def assess_quality(self, scan):
        X = self._encode_one(scan)
        probs = self.quality_probe.predict_proba(X)[0]
        classes = [str(c) for c in self.quality_probe.classes_]
        order = np.argsort(probs)[::-1]
        a, b = int(order[0]), int(order[1])

        raw = classes[a]
        top = float(probs[a])
        second = float(probs[b])
        margin = top - second

        good_t = float(self.quality_policy.get("good_threshold", 0.8))
        margin_t = float(self.quality_policy.get("margin_threshold", 0.0))

        policy_label = raw
        if raw == "good":
            good_score = float(probs[classes.index("good")])
            if good_score < good_t or margin < margin_t:
                policy_label = "review"

        return {
            "label": policy_label,
            "raw_prediction": raw,
            "score": top,
            "margin": margin,
            "probabilities": {classes[i]: float(probs[i]) for i in range(len(classes))},
            "good_threshold": good_t,
            "margin_threshold": margin_t,
            "mode": "advisory",
            "model": "FetalCLIP + quality linear probe",
            "clinical_ground_truth": False,
        }

    def measure_head(self, scan):
        path = self._path(scan)
        spacing = self._spacing(scan)

        payload = run_case_devansh(
            image_path=str(path),
            checkpoint_path=str(self.head_checkpoint),
            model_variant=self.head_variant,
            output_dir=str(self.head_output_dir),
            pixel_spacing_mm=spacing,
            model_bundle=self.head_model_bundle,
            threshold=self.head_threshold,
        )

        measurements = payload.get("measurements", {}) or {}
        decision = payload.get("decision", {}) or {}
        confidence = payload.get("confidence", {}) or {}
        quality = payload.get("quality", {}) or {}

        raw_decision = str(decision.get("decision") or "").strip().lower()
        status = {
            "accept": "valid",
            "warn": "review",
            "review": "review",
            "reject": "invalid",
        }.get(raw_decision, "review")

        conf = _normalized_confidence(
            confidence.get("confidence_score") or quality.get("quality_score")
        )

        reasons = decision.get("reason_codes", []) or []
        if isinstance(reasons, str):
            reasons = [reasons]

        return {
            "status": status,
            "confidence": conf,
            "hc_mm": measurements.get("hc_mm"),
            "bpd_mm": measurements.get("bpd_mm"),
            "ofd_mm": measurements.get("ofd_mm"),
            "head_area_mm2": measurements.get("head_area_mm2"),
            "hc_px": measurements.get("hc_px"),
            "bpd_px": measurements.get("bpd_px"),
            "ofd_px": measurements.get("ofd_px"),
            "scale_available": measurements.get("scale_available"),
            "ga_estimate": measurements.get("ga_estimate"),
            "reasons": list(reasons),
            "artifacts": payload.get("artifacts", {}) or {},
            "phase0_payload": payload,
        }

    def measure_abdomen(self, scan):
        path = self._path(scan)
        spacing = self._spacing(scan)
        roi = scan.roi_xyxy if isinstance(scan, ScanInput) else None
        return self.biometrynet.measure_abdomen(
            image_path=path,
            pixel_spacing_mm=spacing,
            roi_xyxy=roi,
        )

    def measure_femur(self, scan):
        path = self._path(scan)
        spacing = self._spacing(scan)
        roi = scan.roi_xyxy if isinstance(scan, ScanInput) else None
        return self.biometrynet.measure_femur(
            image_path=path,
            pixel_spacing_mm=spacing,
            roi_xyxy=roi,
        )

    def calculate_efw(self, hc_mm, ac_mm, fl_mm):
        return estimate_fetal_weight_safe(hc_mm=hc_mm, ac_mm=ac_mm, fl_mm=fl_mm)
