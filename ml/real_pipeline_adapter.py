from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def _load_callable(module_name: str, function_name: str):
    if not module_name or not function_name:
        raise RuntimeError("Module/function is not configured")
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    return fn


def _call_scan_callable(fn, request: dict):
    sig = inspect.signature(fn)
    params = sig.parameters
    input_path = request["input_path"]
    anatomy = request["expected_anatomy"]
    spacing = request.get("pixel_spacing")
    roi = request.get("roi")
    workdir = request.get("workdir")

    # The web/API stores ROI as {x, y, width, height}. Older validated
    # HRNet helpers commonly accept roi_xyxy=[x0, y0, x1, y1]. Keep the
    # original ROI for callables that accept `roi`, and derive xyxy only
    # for callables that explicitly accept `roi_xyxy`.
    roi_xyxy = None
    if isinstance(roi, dict):
        required = {"x", "y", "width", "height"}
        if required.issubset(roi):
            x0 = float(roi["x"])
            y0 = float(roi["y"])
            w = float(roi["width"])
            h = float(roi["height"])
            if w <= 0 or h <= 0:
                raise ValueError("ROI width/height must be positive")
            roi_xyxy = [x0, y0, x0 + w, y0 + h]
    elif isinstance(roi, (list, tuple)) and len(roi) == 4:
        roi_xyxy = list(roi)

    spacing_tuple = None
    spacing_scalar = None
    if spacing:
        x = spacing.get("x_mm") or spacing.get("column_mm")
        y = spacing.get("y_mm") or spacing.get("row_mm")
        if x is not None and y is not None:
            spacing_tuple = (float(x), float(y))
            if abs(float(x) - float(y)) < 1e-9:
                spacing_scalar = float(x)

    aliases = {
        "input_path": input_path,
        "image_path": input_path,
        "path": input_path,
        "anatomy": anatomy,
        "expected_anatomy": anatomy,
        "roi": roi,
        "roi_xyxy": roi_xyxy,
        "workdir": workdir,
        "output_dir": workdir,
        "pixel_spacing": spacing_tuple,
        "pixel_spacing_xy": spacing_tuple,
        "pixel_spacing_mm": spacing_scalar if spacing_scalar is not None else spacing_tuple,
    }

    kwargs = {}
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    for name in params:
        if name in aliases:
            kwargs[name] = aliases[name]

    if accepts_var_kw:
        kwargs.setdefault("input_path", input_path)
        kwargs.setdefault("anatomy", anatomy)
        kwargs.setdefault("pixel_spacing", spacing_tuple)
        kwargs.setdefault("roi", roi)
        kwargs.setdefault("workdir", workdir)

    return fn(**kwargs)


@lru_cache(maxsize=2)
def _load_head_bundle(checkpoint_path: str, variant: str):
    from src.segmentation.devansh_resunet import load_devansh_static_checkpoint

    return load_devansh_static_checkpoint(checkpoint_path, variant=variant)


def _run_known_head_fallback(request: dict) -> dict:
    """Exact fallback for the known head MVP function used by the current Streamlit app.

    Use REAL_SCAN_MODULE/FUNCTION instead when your latest multi-anatomy pipeline already
    exposes one integrated callable. The fallback deliberately covers HEAD only.
    """
    from src.pipeline_devansh import run_case_devansh

    checkpoint = _env("HEAD_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError("HEAD_CHECKPOINT_PATH is required for the known HEAD fallback")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"HEAD checkpoint not found: {checkpoint}")

    variant = _env("HEAD_MODEL_VARIANT", "phase4a")
    threshold = float(_env("HEAD_THRESHOLD", "0.50"))
    spacing = request.get("pixel_spacing")
    pixel_spacing_mm = None
    if spacing:
        x = spacing.get("x_mm") or spacing.get("column_mm")
        y = spacing.get("y_mm") or spacing.get("row_mm")
        if x is not None and y is not None:
            if abs(float(x) - float(y)) > 1e-9:
                raise ValueError(
                    "Known run_case_devansh fallback accepts one scalar pixel spacing; "
                    "manual anisotropic spacing must be handled by your current integrated pipeline instead."
                )
            pixel_spacing_mm = float(x)

    bundle = _load_head_bundle(checkpoint, variant)
    return run_case_devansh(
        image_path=request["input_path"],
        checkpoint_path=checkpoint,
        model_variant=variant,
        output_dir=request["workdir"],
        pixel_spacing_mm=pixel_spacing_mm,
        model_bundle=bundle,
        threshold=threshold,
    )


def _extract_measurements(raw: dict, anatomy: str) -> dict:
    m = raw.get("measurements") or {}
    out: dict[str, dict] = {}

    def add(name: str, keys: list[str], unit: str):
        value = None
        if isinstance(m, dict):
            for k in keys:
                if k in m and m[k] is not None:
                    value = m[k]
                    break
        if value is None:
            for k in keys:
                if k in raw and raw[k] is not None:
                    value = raw[k]
                    break
        if value is None:
            return
        if isinstance(value, dict) and "value" in value:
            item = dict(value)
            item.setdefault("unit", unit)
            out[name] = _json_safe(item)
        else:
            out[name] = {"value": float(value), "unit": unit}

    if anatomy == "HEAD":
        add("hc_mm", ["hc_mm", "HC", "HC_mm"], "mm")
        add("bpd_mm", ["bpd_mm", "BPD", "BPD_mm"], "mm")
        add("ofd_mm", ["ofd_mm", "OFD", "OFD_mm"], "mm")
        add("head_area_mm2", ["head_area_mm2", "HEAD_AREA", "Head Area"], "mm2")
        if isinstance(m, dict) and isinstance(m.get("ga_estimate"), dict):
            ga = m["ga_estimate"]
            val = ga.get("ga_weeks_decimal")
            if val is None and ga.get("ga_weeks") is not None:
                val = float(ga["ga_weeks"]) + float(ga.get("ga_days", 0)) / 7.0
            if val is not None:
                out["ga_weeks"] = {
                    "value": float(val),
                    "unit": "weeks",
                    "method": ga.get("method"),
                    "display": ga.get("ga_display") or ga.get("ga_text") or ga.get("estimate_text"),
                }
    elif anatomy == "ABDOMEN":
        add("ac_mm", ["ac_mm", "AC", "AC_mm"], "mm")
    elif anatomy == "FEMUR":
        add("fl_mm", ["fl_mm", "FL", "FL_mm"], "mm")
    return out


def _normalize_status(value: Any) -> str:
    text = str(value or "REVIEW").upper()
    if text in {"ACCEPT", "PASS", "GOOD"}:
        return "ACCEPT"
    if text in {"REJECT", "FAIL", "FAILED", "POOR"}:
        return "REJECT"
    return "REVIEW"


def _canonicalize(raw: dict, expected_anatomy: str) -> dict:
    raw = _json_safe(raw)
    actual = str(raw.get("anatomy") or raw.get("route") or expected_anatomy).upper()
    measurements = _extract_measurements(raw, actual)

    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    confidence = raw.get("confidence") if isinstance(raw.get("confidence"), dict) else {}
    fusion = raw.get("fusion") if isinstance(raw.get("fusion"), dict) else confidence
    status = fusion.get("status") if isinstance(fusion, dict) else None
    status = status or decision.get("decision") or "REVIEW"

    plane = raw.get("plane") or raw.get("plane_detection") or raw.get("plane_result") or {}
    if not isinstance(plane, dict):
        plane = {"label": str(plane)} if plane else {}
    if plane.get("label"):
        plane["label"] = str(plane["label"]).upper()

    quality = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}
    if "label" not in quality and quality.get("quality_label") is not None:
        quality["label"] = quality.get("quality_label")
    if "score" not in quality and quality.get("quality_score") is not None:
        quality["score"] = quality.get("quality_score")

    artifacts_raw = raw.get("artifacts") or {}
    artifacts: dict[str, Any] = {}
    if isinstance(artifacts_raw, dict):
        for key, value in artifacts_raw.items():
            if key.endswith("_path") and value:
                artifacts[key[:-5]] = str(value)
            elif key in {"overlay", "mask", "preview"} and value:
                artifacts[key] = value
    elif isinstance(artifacts_raw, list):
        for item in artifacts_raw:
            if isinstance(item, dict) and item.get("name") and item.get("path"):
                artifacts[item["name"]] = {"path": item["path"], "content_type": item.get("mime")}

    warnings = list(raw.get("warnings") or [])
    warnings.extend(decision.get("reason_codes") or [])
    warnings.extend(quality.get("warnings") or [])
    warnings = list(dict.fromkeys(str(x) for x in warnings if x))

    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    provenance.setdefault("release", _env("PIPELINE_MODEL_RELEASE", "local-real"))
    provenance.setdefault("git_commit", _git_commit(_env("MODEL_REPO_PATH")))
    provenance.setdefault("device", raw.get("device"))

    return {
        "schema_version": "1.0",
        "source": "real",
        "anatomy": actual,
        "measurements": measurements,
        "plane": plane,
        "quality": quality,
        "fusion": {**(fusion if isinstance(fusion, dict) else {}), "status": _normalize_status(status)},
        "warnings": warnings,
        "artifacts": artifacts,
        "dicom_metadata": raw.get("dicom_metadata")
        or ((raw.get("input_info") or {}).get("metadata_summary") if isinstance(raw.get("input_info"), dict) else {})
        or {},
        "provenance": provenance,
        "device": raw.get("device") or provenance.get("device"),
    }


def _git_commit(repo: str) -> str | None:
    if not repo or not Path(repo).exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=3
        ).strip()
    except Exception:
        return None


def analyze_scan(request: dict) -> dict:
    expected = str(request["expected_anatomy"]).upper()
    module_name = _env("REAL_SCAN_MODULE")
    function_name = _env("REAL_SCAN_FUNCTION")

    if module_name and function_name:
        fn = _load_callable(module_name, function_name)
        raw = _call_scan_callable(fn, request)
    elif expected == "HEAD":
        raw = _run_known_head_fallback(request)
    else:
        raise NotImplementedError(
            f"{expected} requires your latest integrated callable. Set REAL_SCAN_MODULE and REAL_SCAN_FUNCTION "
            "to the function that already runs your FetalCLIP/routing/HRNet pipeline."
        )

    result = _canonicalize(raw, expected)
    actual = result["anatomy"]
    if actual not in {"HEAD", "ABDOMEN", "FEMUR", "UNSUPPORTED"}:
        raise ValueError(f"Unsupported routed anatomy from real pipeline: {actual!r}")
    if actual != expected and actual != "UNSUPPORTED":
        result["warnings"].append(
            f"Uploaded slot was {expected}, but the real pipeline routed the scan as {actual}."
        )
    return result


def _extract_value(result: dict, key: str) -> float:
    item = (result.get("measurements") or {}).get(key)
    if isinstance(item, dict):
        item = item.get("value")
    if item is None:
        raise ValueError(f"Required measurement {key} is missing")
    return float(item)


def summarize_study(request: dict) -> dict:
    scans = request.get("scans") or {}
    for anatomy in ("HEAD", "ABDOMEN", "FEMUR"):
        if anatomy not in scans:
            raise ValueError(f"Missing {anatomy} scan result")
        status = str(((scans[anatomy].get("fusion") or {}).get("status") or "REVIEW")).upper()
        if status != "ACCEPT":
            raise ValueError(f"{anatomy} status is {status}; EFW requires ACCEPT for all three scans")

    hc_mm = _extract_value(scans["HEAD"], "hc_mm")
    ac_mm = _extract_value(scans["ABDOMEN"], "ac_mm")
    fl_mm = _extract_value(scans["FEMUR"], "fl_mm")

    module_name = _env("REAL_EFW_MODULE")
    function_name = _env("REAL_EFW_FUNCTION")
    if not module_name or not function_name:
        raise RuntimeError(
            "REAL_EFW_MODULE and REAL_EFW_FUNCTION must point to your already-validated HC+AC+FL EFW function."
        )
    fn = _load_callable(module_name, function_name)
    sig = inspect.signature(fn)
    names = set(sig.parameters)
    if {"hc_mm", "ac_mm", "fl_mm"}.issubset(names):
        raw = fn(hc_mm=hc_mm, ac_mm=ac_mm, fl_mm=fl_mm)
    elif "measurements" in names:
        raw = fn(measurements={"hc_mm": hc_mm, "ac_mm": ac_mm, "fl_mm": fl_mm})
    else:
        # Last-resort positional call for an existing deterministic helper.
        raw = fn(hc_mm, ac_mm, fl_mm)

    raw = _json_safe(raw)
    if isinstance(raw, dict):
        efw = raw.get("efw_g") or raw.get("EFW_g") or raw.get("EFW") or raw.get("efw")
        formula = raw.get("formula") or raw.get("method") or raw.get("version")
        if isinstance(efw, dict):
            efw_value = efw.get("value")
            formula = formula or efw.get("method")
        else:
            efw_value = efw
    else:
        efw_value = raw
        formula = None
    if efw_value is None:
        raise ValueError("Existing EFW function did not return an EFW value")

    return {
        "source": "real",
        "measurements": {
            "efw_g": {
                "value": float(efw_value),
                "unit": "g",
                "status": "ACCEPT",
            }
        },
        "warnings": [],
        "provenance": {
            "formula": formula or _env("EFW_FORMULA_VERSION", "existing-validated-efw"),
            "inputs": {"hc_mm": hc_mm, "ac_mm": ac_mm, "fl_mm": fl_mm},
        },
    }


def self_test() -> dict:
    repo = _env("MODEL_REPO_PATH")
    if not repo or not Path(repo).exists():
        raise FileNotFoundError(f"MODEL_REPO_PATH does not exist: {repo!r}")
    scan_module = _env("REAL_SCAN_MODULE")
    scan_function = _env("REAL_SCAN_FUNCTION")
    scan_callable = None
    if scan_module and scan_function:
        fn = _load_callable(scan_module, scan_function)
        scan_callable = f"{scan_module}:{scan_function}"
        _ = inspect.signature(fn)
    else:
        # Verify the exact known head fallback is importable.
        importlib.import_module("src.pipeline_devansh")
        importlib.import_module("src.segmentation.devansh_resunet")
        scan_callable = "HEAD fallback: src.pipeline_devansh:run_case_devansh"

    efw = None
    if _env("REAL_EFW_MODULE") and _env("REAL_EFW_FUNCTION"):
        _load_callable(_env("REAL_EFW_MODULE"), _env("REAL_EFW_FUNCTION"))
        efw = f"{_env('REAL_EFW_MODULE')}:{_env('REAL_EFW_FUNCTION')}"

    return {
        "ok": True,
        "python": sys.executable,
        "model_repo_path": repo,
        "scan_callable": scan_callable,
        "efw_callable": efw,
        "head_checkpoint": _env("HEAD_CHECKPOINT_PATH") or None,
        "hrnet_python": _env("HRNET_PYTHON") or None,
        "pipeline_release": _env("PIPELINE_MODEL_RELEASE", "local-real"),
    }
