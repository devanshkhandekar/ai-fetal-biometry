from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from app.config import settings

CANONICAL_MEASUREMENT_UNITS = {
    "hc_mm": "mm",
    "bpd_mm": "mm",
    "ofd_mm": "mm",
    "head_area_mm2": "mm2",
    "ga_weeks": "weeks",
    "ac_mm": "mm",
    "fl_mm": "mm",
    "efw_g": "g",
}


def _mock(anatomy: str) -> dict[str, Any]:
    values = {
        "HEAD": {
            "hc_mm": 320.2,
            "bpd_mm": 82.4,
            "ofd_mm": 101.8,
            "head_area_mm2": 8200.0,
            "ga_weeks": 34.1,
        },
        "ABDOMEN": {"ac_mm": 301.5},
        "FEMUR": {"fl_mm": 60.3},
    }[anatomy]
    return {
        "plane": {"label": anatomy, "confidence": 0.97, "advisory": True},
        "quality": {"label": "GOOD", "confidence": 0.92, "advisory": True},
        "route": anatomy,
        "fusion": {"status": "ACCEPT", "confidence": 0.91, "reasons": []},
        "measurements": {
            k: {
                "value": v,
                "unit": CANONICAL_MEASUREMENT_UNITS[k],
                "confidence": 0.9,
                "status": "ACCEPT",
            }
            for k, v in values.items()
        },
        "artifacts": {},
        "dicom_metadata": {},
        "warnings": [
            "MOCK PIPELINE RESULT - replace PIPELINE_MODE=mock before clinical/model validation."
        ],
        "device": "mock-cpu",
        "provenance": {"source": "mock"},
    }


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
        try:
            return _json_safe(value.model_dump())
        except Exception:
            pass
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


def _normalize_status(value: Any, default: str = "REVIEW") -> str:
    if value is None:
        return default
    text = str(value).strip().upper()
    mapping = {
        "ACCEPT": "ACCEPT",
        "PASS": "ACCEPT",
        "GOOD": "ACCEPT",
        "WARN": "REVIEW",
        "WARNING": "REVIEW",
        "REVIEW": "REVIEW",
        "REJECT": "REJECT",
        "FAIL": "REJECT",
        "FAILED": "REJECT",
        "POOR": "REJECT",
    }
    return mapping.get(text, default)


def _measurement(value: Any, unit: str, status: str = "REVIEW", confidence: Any = None):
    if value is None:
        return None
    if isinstance(value, dict) and "value" in value:
        out = dict(value)
        out.setdefault("unit", unit)
        out.setdefault("status", status)
        return _json_safe(out)
    return {
        "value": float(value),
        "unit": unit,
        "confidence": confidence,
        "status": status,
    }


# def _canonical_result_status(raw: dict[str, Any]) -> str:
#     """Use the final fusion/decision status as the canonical measurement status.

#     The real subprocess adapter returns fusion.status but usually no legacy
#     decision object.  Falling back only to raw.decision incorrectly marked
#     ACCEPTed measurements as REVIEW.
#     """
#     fusion = raw.get("fusion") if isinstance(raw.get("fusion"), dict) else {}
#     decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
#     return _normalize_status(
#         fusion.get("status")
#         or decision.get("decision")
#         or raw.get("measurement_status")
#     )


def _canonical_result_status(
    raw,
):

    fusion = (
        raw.get("fusion")
        if isinstance(
            raw.get("fusion"),
            dict,
        )
        else {}
    )

    decision = (
        raw.get("decision")
        if isinstance(
            raw.get("decision"),
            dict,
        )
        else {}
    )

    return _normalize_status(

        fusion.get("status")

        or decision.get(
            "decision"
        )

        or raw.get(
            "measurement_status"
        )
    )


def _flat_measurements(raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "hc_mm": ["hc_mm", "HC_mm", "hc", "HC"],
        "bpd_mm": ["bpd_mm", "BPD_mm", "bpd", "BPD"],
        "ofd_mm": ["ofd_mm", "OFD_mm", "ofd", "OFD"],
        "head_area_mm2": ["head_area_mm2", "Head_Area_mm2", "head_area", "HEAD_AREA"],
        "ga_weeks": ["ga_weeks", "GA_weeks", "ga", "GA"],
        "ac_mm": ["ac_mm", "AC_mm", "ac", "AC"],
        "fl_mm": ["fl_mm", "FL_mm", "fl", "FL"],
        "efw_g": ["efw_g", "EFW_g", "efw", "EFW"],
    }
    out = {}
    status = _canonical_result_status(raw)
    for canonical, keys in aliases.items():
        found = next((raw[k] for k in keys if k in raw and raw[k] is not None), None)
        if found is not None:
            out[canonical] = _measurement(
                found,
                CANONICAL_MEASUREMENT_UNITS[canonical],
                status=status,
                confidence=raw.get(f"{canonical}_confidence"),
            )
    return out


def _normalize_measurement_dict(raw_measurements: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "hc_mm": ["hc_mm", "HC_mm", "HC", "hc"],
        "bpd_mm": ["bpd_mm", "BPD_mm", "BPD", "bpd"],
        "ofd_mm": ["ofd_mm", "OFD_mm", "OFD", "ofd"],
        "head_area_mm2": ["head_area_mm2", "HEAD_AREA", "Head Area", "head_area"],
        "ga_weeks": ["ga_weeks", "GA", "ga", "ga_weeks_decimal"],
        "ac_mm": ["ac_mm", "AC_mm", "AC", "ac"],
        "fl_mm": ["fl_mm", "FL_mm", "FL", "fl"],
        "efw_g": ["efw_g", "EFW_g", "EFW", "efw"],
    }
    default_status = _canonical_result_status(raw)
    out: dict[str, Any] = {}
    for canonical, keys in aliases.items():
        for key in keys:
            if key not in raw_measurements or raw_measurements[key] is None:
                continue
            item = _measurement(
                raw_measurements[key],
                CANONICAL_MEASUREMENT_UNITS[canonical],
                status=default_status,
            )
            if item is not None:
                out[canonical] = item
            break

    # Current head MVP stores GA as a nested ga_estimate object.
    ga = raw_measurements.get("ga_estimate")
    if isinstance(ga, dict):
        val = ga.get("ga_weeks_decimal")
        if val is None and ga.get("ga_weeks") is not None:
            try:
                val = float(ga["ga_weeks"]) + float(ga.get("ga_days", 0)) / 7.0
            except Exception:
                val = None
        if val is not None:
            out["ga_weeks"] = {
                "value": float(val),
                "unit": "weeks",
                "confidence": None,
                "status": default_status,
                "method": ga.get("method"),
                "display": ga.get("ga_display") or ga.get("ga_text") or ga.get("estimate_text"),
            }
    return out


def normalize_result(raw: Any, anatomy: str) -> dict[str, Any]:
    raw = _json_safe(raw)
    if not isinstance(raw, dict):
        raise TypeError("Existing pipeline must return a dict or Pydantic model")

    raw_measurements = raw.get("measurements") or {}
    if isinstance(raw_measurements, list):
        raw_measurements = {
            str(x.get("name")): x for x in raw_measurements if isinstance(x, dict) and x.get("name")
        }
    measurements = _normalize_measurement_dict(raw_measurements, raw) if isinstance(raw_measurements, dict) else {}
    if not measurements:
        measurements = _flat_measurements(raw)

    decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else {}
    confidence = raw.get("confidence") if isinstance(raw.get("confidence"), dict) else {}
    fusion_raw = raw.get("fusion") if isinstance(raw.get("fusion"), dict) else confidence
    fusion_status = fusion_raw.get("status") or decision.get("decision") or raw.get("confidence")
    fusion = {
        **(fusion_raw if isinstance(fusion_raw, dict) else {}),
        "status": _normalize_status(fusion_status),
    }
    if decision.get("reason_codes") and not fusion.get("reasons"):
        fusion["reasons"] = decision.get("reason_codes")

    quality_raw = raw.get("quality") if isinstance(raw.get("quality"), dict) else {}
    quality = dict(quality_raw)
    if "label" not in quality and quality.get("quality_label") is not None:
        quality["label"] = quality.get("quality_label")
    if "confidence" not in quality:
        quality["confidence"] = quality.get("quality_confidence")
    if "score" not in quality and quality.get("quality_score") is not None:
        quality["score"] = quality.get("quality_score")

    plane = raw.get("plane") or raw.get("plane_result") or raw.get("plane_detection") or {}
    if hasattr(plane, "model_dump"):
        plane = plane.model_dump()
    if not isinstance(plane, dict):
        plane = {"label": str(plane)} if plane else {}
    if plane.get("label"):
        plane["label"] = str(plane["label"]).upper()

    route = raw.get("route") or raw.get("anatomy")
    routing = raw.get("routing") if isinstance(raw.get("routing"), dict) else {}
    route = route or routing.get("route") or routing.get("anatomy") or anatomy
    route = str(route).upper()

    artifacts_raw = raw.get("artifacts") or {}
    artifacts: dict[str, Any] = {}
    if isinstance(artifacts_raw, list):
        for item in artifacts_raw:
            if isinstance(item, dict) and item.get("name") and item.get("path"):
                artifacts[str(item["name"])] = {
                    "path": item["path"],
                    "content_type": item.get("mime"),
                }
    elif isinstance(artifacts_raw, dict):
        artifacts.update(artifacts_raw)
        # Current head MVP uses *_path keys.
        for old_key, new_key in {
            "overlay_path": "overlay",
            "mask_path": "mask",
            "preview_path": "preview",
        }.items():
            if artifacts_raw.get(old_key) and new_key not in artifacts:
                artifacts[new_key] = {"path": artifacts_raw[old_key]}

    warnings = list(raw.get("warnings") or [])
    if quality_raw.get("warnings"):
        warnings.extend(quality_raw.get("warnings") or [])
    if decision.get("reason_codes"):
        warnings.extend([str(x) for x in decision.get("reason_codes") or []])
    warnings = list(dict.fromkeys(str(w) for w in warnings if w))

    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {"raw": provenance}
    model_info = raw.get("model_info")
    audit = raw.get("audit")
    if model_info is not None:
        provenance.setdefault("model_info", model_info)
    if audit is not None:
        provenance.setdefault("pipeline_audit", audit)

    result = {
        "schema_version": str(raw.get("schema_version") or "1.0"),
        "source": str(raw.get("source") or raw.get("mode") or "real"),
        "plane": plane,
        "quality": quality,
        "route": route,
        "fusion": fusion,
        "measurements": measurements,
        "artifacts": artifacts,
        "dicom_metadata": raw.get("dicom_metadata")
        or ((raw.get("input_info") or {}).get("metadata_summary") if isinstance(raw.get("input_info"), dict) else {})
        or {},
        "warnings": warnings,
        "device": raw.get("device") or provenance.get("device"),
        "provenance": provenance,
    }
    return _json_safe(result)


def _call_existing_pipeline(
    input_path: str,
    anatomy: str,
    pixel_spacing: tuple[float, float] | None,
    roi: dict | None,
    workdir: str | None,
):
    if settings.model_repo_path and settings.model_repo_path not in sys.path:
        sys.path.insert(0, settings.model_repo_path)
    module = importlib.import_module(settings.pipeline_module)
    fn = getattr(module, settings.pipeline_function)
    kwargs = {
        "input_path": input_path,
        "anatomy": anatomy,
        "pixel_spacing": pixel_spacing,
        "roi": roi,
    }
    if workdir is not None:
        kwargs["workdir"] = workdir
    try:
        return fn(**kwargs)
    except TypeError as exc:
        # Preserve Phase-1 compatibility with a four-argument wrapper.
        if "workdir" in kwargs:
            kwargs.pop("workdir", None)
            try:
                return fn(**kwargs)
            except TypeError:
                raise exc
        raise


def _call_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    if not settings.pipeline_python:
        raise RuntimeError("PIPELINE_PYTHON is required when PIPELINE_MODE=subprocess")
    if not settings.pipeline_bridge_script:
        raise RuntimeError("PIPELINE_BRIDGE_SCRIPT is required when PIPELINE_MODE=subprocess")
    python_path = Path(settings.pipeline_python)
    bridge_path = Path(settings.pipeline_bridge_script)
    if not python_path.exists():
        raise FileNotFoundError(f"PIPELINE_PYTHON not found: {python_path}")
    if not bridge_path.exists():
        raise FileNotFoundError(f"PIPELINE_BRIDGE_SCRIPT not found: {bridge_path}")

    proc = subprocess.run(
        [str(python_path), str(bridge_path)],
        input=json.dumps(_json_safe(payload)),
        text=True,
        capture_output=True,
        timeout=settings.pipeline_timeout_sec,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Real pipeline subprocess failed "
            f"(exit={proc.returncode}). stderr={proc.stderr[-6000:]} stdout={proc.stdout[-2000:]}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Real pipeline bridge did not return valid JSON. "
            f"stdout={proc.stdout[-4000:]} stderr={proc.stderr[-4000:]}"
        ) from exc
    if not isinstance(data, dict):
        raise TypeError("Real pipeline bridge must return a JSON object")
    return data


def run_model(
    input_path: str,
    anatomy: str,
    pixel_spacing_x: float | None,
    pixel_spacing_y: float | None,
    roi: dict | None,
    workdir: str | None = None,
) -> dict[str, Any]:
    mode = settings.pipeline_mode.lower()
    if mode == "mock":
        return _mock(anatomy)

    spacing = None
    if pixel_spacing_x is not None and pixel_spacing_y is not None:
        spacing = (float(pixel_spacing_x), float(pixel_spacing_y))

    if mode == "python":
        raw = _call_existing_pipeline(input_path, anatomy, spacing, roi, workdir)
    elif mode == "subprocess":
        raw = _call_subprocess(
            {
                "operation": "scan",
                "input_path": input_path,
                "workdir": workdir,
                "expected_anatomy": anatomy,
                "pixel_spacing": (
                    {
                        "x_mm": spacing[0],
                        "y_mm": spacing[1],
                        "row_mm": spacing[1],
                        "column_mm": spacing[0],
                        "source": "manual",
                    }
                    if spacing is not None
                    else None
                ),
                "roi": roi,
            }
        )
    else:
        raise ValueError(
            f"Unsupported PIPELINE_MODE={settings.pipeline_mode!r}; use mock, python or subprocess"
        )

    return normalize_result(raw, anatomy)


def aggregate_study_results(scan_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run the project's validated study-level aggregation (for example EFW).

    No clinical formula is implemented here. In subprocess mode the exact saved
    HEAD/ABDOMEN/FEMUR results are passed to the existing model environment.
    """
    mode = settings.pipeline_mode.lower()
    if mode == "mock":
        return {
            "source": "mock",
            "measurements": {},
            "warnings": ["Mock mode does not generate EFW."],
            "provenance": {},
        }
    if mode == "subprocess":
        raw = _call_subprocess({"operation": "aggregate", "scans": scan_results})
        raw = _json_safe(raw)
        if not isinstance(raw, dict):
            raise TypeError("Study aggregator must return a dictionary")
        measurements = raw.get("measurements") or {}
        if not measurements and raw.get("efw") is not None:
            efw = raw["efw"]
            if isinstance(efw, dict):
                measurements = {"efw_g": _measurement(efw, "g", status="ACCEPT")}
            else:
                measurements = {"efw_g": _measurement(efw, "g", status="ACCEPT")}
        if not measurements:
            measurements = _flat_measurements(raw)
        return {
            "source": raw.get("source", "real"),
            "measurements": measurements,
            "warnings": list(raw.get("warnings") or raw.get("flags") or []),
            "provenance": raw.get("provenance") or {},
        }

    # Direct-python mode is retained for users who intentionally keep the
    # aggregator importable in the platform worker process.
    if mode == "python" and settings.study_aggregator_module and settings.study_aggregator_function:
        if settings.model_repo_path and settings.model_repo_path not in sys.path:
            sys.path.insert(0, settings.model_repo_path)
        module = importlib.import_module(settings.study_aggregator_module)
        fn = getattr(module, settings.study_aggregator_function)
        simple = {}
        for anatomy, result in scan_results.items():
            simple[anatomy] = {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in (result.get("measurements") or {}).items()
            }
        raw = _json_safe(fn(scans=simple))
        if not isinstance(raw, dict):
            raise TypeError("Study aggregator must return a dict")
        measurements = raw.get("measurements") or _flat_measurements(raw)
        return {
            "source": raw.get("source", "real"),
            "measurements": measurements,
            "warnings": list(raw.get("warnings") or []),
            "provenance": raw.get("provenance") or {},
        }

    raise RuntimeError(
        "No study aggregator is configured for the current PIPELINE_MODE. "
        "Use subprocess mode for Phase 3 or configure STUDY_AGGREGATOR_MODULE/FUNCTION."
    )


def canonical_json_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
