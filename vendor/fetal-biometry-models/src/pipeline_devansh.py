from __future__ import annotations

from pathlib import Path
import json
from typing import Optional, Dict, Any

import cv2
import numpy as np

from src.io.image_loader import load_image
from src.io.dicom_loader import load_dicom
from src.preprocessing.preprocess import preprocess_for_model
from src.gating.validity_gate import input_validity_gate, post_segmentation_gate
from src.segmentation.devansh_resunet import load_devansh_static_checkpoint, predict_devansh_mask
from src.postprocess.mask_postprocess import clean_mask
from src.postprocess.ellipse_fit import fit_head_ellipse
from src.biometry.measurements import compute_biometry_from_ellipse
from src.biometry.ga_lookup import estimate_ga_from_hc
from src.quality.quality_confidence import score_quality, score_confidence, final_decision
from src.visualization.overlays import draw_ellipse_overlay, save_overlay
from src.reporting.payload_builder import build_payload, build_report_previews, normalize_ga_dict
from src.utils.config import load_yaml
from src.utils.audit import sha256_file, sha256_dict, runtime_environment, now_ms


def load_any_input(path, pixel_spacing_mm=None):
    suffix = Path(path).suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        img, meta = load_dicom(path)
        if pixel_spacing_mm is not None and meta.get("pixel_spacing_mm") is None:
            meta["pixel_spacing_mm"] = pixel_spacing_mm
            meta["pixel_spacing_source"] = "manual_override"
        return img, meta
    return load_image(path, pixel_spacing_mm=pixel_spacing_mm)


def _artifact_paths(output_dir: Path, image_path: str | Path, model_variant: str):
    stem = Path(image_path).stem
    mask_path = output_dir / "predictions" / f"{stem}_devansh_{model_variant}_mask.png"
    overlay_path = output_dir / "overlays" / f"{stem}_devansh_{model_variant}_overlay.png"
    report_path = output_dir / "reports" / f"{stem}_devansh_{model_variant}_report.json"
    for p in [mask_path, overlay_path, report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    return mask_path, overlay_path, report_path


def _empty_mask_like(img):
    h, w = img.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def _model_info(checkpoint_path, model_variant, model_bundle=None) -> Dict[str, Any]:
    return {
        "model_family": "devansh_static_fetal_head_segmentation",
        "model_variant": model_variant,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": Path(checkpoint_path).name,
        "checkpoint_sha256": sha256_file(checkpoint_path) if Path(checkpoint_path).exists() else None,
        "device": getattr(model_bundle, "device", "cpu") if model_bundle is not None else "cpu",
        "input_size_hw": list(getattr(model_bundle, "input_size", (256, 384))) if model_bundle is not None else [256, 384],
        "ga_method": "HC_TO_GA_HADLOCK_POLYNOMIAL",
    }


def run_case_devansh(
    image_path,
    checkpoint_path,
    model_variant="phase4a",
    output_dir="outputs",
    pixel_spacing_mm: Optional[float] = None,
    model_bundle=None,
    threshold=0.5,
    model_cfg_path="/media/devanshot/Storage/ubuntu_universe/Devansh_Research/AI_Fetal_Ultrasound_Biometry/mvp_tested_versions/fetal_biometry_devansh_mvp_ellipse_ga_robust_complete/fetal_biometry_devansh_mvp_ellipse_ga_robust/configs/model_config.yaml",
    chart_cfg_path="/media/devanshot/Storage/ubuntu_universe/Devansh_Research/AI_Fetal_Ultrasound_Biometry/mvp_tested_versions/fetal_biometry_devansh_mvp_ellipse_ga_robust_complete/fetal_biometry_devansh_mvp_ellipse_ga_robust/configs/chart_config.yaml",
):
    """Run complete Core Biometry MVP pipeline with HC-derived GA.

    Flow:
        input → validity gate → segmentation → mask cleanup → ellipse fitting
        → HC/BPD/OFD/head-area → GA from HC → quality/confidence
        → accept/warn/reject → overlay + JSON payload.
    """
    start_ms = now_ms()
    cfg = load_yaml(model_cfg_path)
    _ = load_yaml(chart_cfg_path)  # retained for backward compatibility/config validation
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img, input_meta = load_any_input(image_path, pixel_spacing_mm=pixel_spacing_mm)
    input_meta = dict(input_meta or {})
    input_meta["filename"] = input_meta.get("filename") or Path(image_path).name
    input_meta["input_sha256"] = sha256_file(image_path)
    input_meta["pixel_spacing_used_for_measurements"] = input_meta.get("pixel_spacing_mm")

    input_gate = input_validity_gate(img)
    prep = preprocess_for_model(img, image_size=int(cfg.get("image_size", 256)))

    if model_bundle is None and input_gate.passed:
        model_bundle = load_devansh_static_checkpoint(checkpoint_path, variant=model_variant)

    model_info = _model_info(checkpoint_path, model_variant, model_bundle)

    if not input_gate.passed:
        empty_mask = _empty_mask_like(img)
        empty_post_gate = post_segmentation_gate(empty_mask, None, cfg)
        measurements = compute_biometry_from_ellipse({}, input_meta.get("pixel_spacing_mm"))
        ga = normalize_ga_dict(measurements.get("ga_estimate") or estimate_ga_from_hc(None))
        quality = {"quality_score": 0, "quality_label": "poor", "warnings": input_gate.reason_codes, "features": {}}
        confidence = {"confidence_score": 0, "confidence_band": "low", "components": {}}
        decision = {"decision": "reject", "reason_codes": input_gate.reason_codes}
        audit = {
            "runtime_ms": now_ms() - start_ms,
            "environment": runtime_environment(),
            "generated_at_epoch_ms": now_ms(),
            "pipeline_status": "rejected_before_segmentation",
            "ga_status": "not_calculated_input_rejected",
        }
        payload_seed = {
            "model_info": model_info,
            "audit": audit,
            "measurements": {**measurements, "ga_estimate": ga},
            "quality": quality,
            "confidence": confidence,
            "decision": decision,
            "ellipse_parameters": None,
            "preprocessing": prep.metadata,
        }
        payload = build_payload(
            input_meta=input_meta,
            preprocessing_meta=prep.metadata,
            input_gate=input_gate,
            post_gate=empty_post_gate,
            seg_meta={},
            post_meta={},
            ellipse=None,
            measurements=measurements,
            ga=ga,
            quality=quality,
            confidence=confidence,
            decision=decision,
            artifacts={},
            model_info=model_info,
            audit=audit,
            report_previews=build_report_previews(payload_seed),
        )
        payload["audit"]["payload_sha256"] = sha256_dict(payload)
        return payload

    mask_raw, prob, seg_meta = predict_devansh_mask(model_bundle, img, threshold=threshold, resize_to_original=True)
    mask, post_meta = clean_mask(mask_raw, min_area_px=int(cfg.get("min_mask_area_px", 100)))
    ellipse = fit_head_ellipse(mask, image_shape=mask.shape)
    post_gate = post_segmentation_gate(mask, ellipse, cfg)

    measurements = compute_biometry_from_ellipse(ellipse or {}, input_meta.get("pixel_spacing_mm"), include_ga=True)
    ga = normalize_ga_dict(measurements.get("ga_estimate") or estimate_ga_from_hc(measurements.get("hc_mm")))
    measurements["ga_estimate"] = ga

    quality = score_quality(img, mask, ellipse)
    confidence = score_confidence(seg_meta, quality, post_gate.passed)
    decision = final_decision(input_gate, post_gate, quality, confidence)

    mask_path, overlay_path, report_path = _artifact_paths(output_dir, image_path, model_variant)
    cv2.imwrite(str(mask_path), (mask * 255).astype("uint8"))
    overlay = draw_ellipse_overlay(img, mask, ellipse)
    save_overlay(overlay_path, overlay)

    audit = {
        "runtime_ms": now_ms() - start_ms,
        "environment": runtime_environment(),
        "generated_at_epoch_ms": now_ms(),
        "pipeline_status": "completed",
        "mask_sha256": sha256_file(mask_path),
        "overlay_sha256": sha256_file(overlay_path),
        "ga_status": "calculated" if measurements.get("hc_mm") is not None else "not_calculated_no_hc_mm",
        "ga_method": ga.get("method"),
        "ellipse_fit_version": "robust_largest_contour_corrected_angle_iou_v2",
        "preprocessing_steps": [
            "input_loading",
            "input_validity_gate",
            "model_specific_resize_256x384",
            "intensity_scale_0_1",
            "normalization_mean_0.2_std_0.15",
            "segmentation_probability_prediction",
            "probability_thresholding",
            "final_clean_mask_cleanup_same_mask_for_overlay_and_ellipse",
            "ellipse_fit",
            "deterministic_biometry",
            "hc_to_ga_hadlock_polynomial",
            "quality_confidence_triage",
        ],
    }

    payload_seed = {
        "model_info": model_info,
        "audit": audit,
        "measurements": measurements,
        "quality": quality,
        "confidence": confidence,
        "decision": decision,
        "ellipse_parameters": ellipse,
        "preprocessing": prep.metadata,
        "validity_gate": {},
    }

    payload = build_payload(
        input_meta=input_meta,
        preprocessing_meta={**prep.metadata, "segmentation_backend_preprocessing": "devansh_256x384_mean0.2_std0.15"},
        input_gate=input_gate,
        post_gate=post_gate,
        seg_meta=seg_meta,
        post_meta=post_meta,
        ellipse=ellipse,
        measurements=measurements,
        ga=ga,
        quality=quality,
        confidence=confidence,
        decision=decision,
        artifacts={"mask_path": str(mask_path), "overlay_path": str(overlay_path), "report_path": str(report_path)},
        model_info=model_info,
        audit=audit,
        report_previews=build_report_previews(payload_seed),
    )
    payload["audit"]["payload_sha256"] = sha256_dict(payload)
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return payload
