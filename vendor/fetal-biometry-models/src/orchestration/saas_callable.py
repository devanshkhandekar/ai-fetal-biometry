from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from src.orchestration.project_backend import ProjectBackend
from src.orchestration.scan_input import ScanInput
from src.orchestration.unified_pipeline import UnifiedFetalBiometryPipeline


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _config_path() -> str:
    return os.environ.get(
        "PROJECT_BACKEND_CONFIG",
        "configs/project_backend.json",
    ).strip()


@lru_cache(maxsize=1)
def _backend() -> ProjectBackend:
    return ProjectBackend(config_path=_config_path())


@lru_cache(maxsize=1)
def _pipeline() -> UnifiedFetalBiometryPipeline:
    # The SaaS real-inference path must actually route to HEAD/ABDOMEN/FEMUR.
    # Plane/quality remain evidence and confidence inputs; unsupported/review
    # cases are still preserved by the existing router/fusion policy.
    return UnifiedFetalBiometryPipeline(
        _backend(),
        routing_enabled=_env_bool("ORCHESTRATION_ROUTING_ENABLED", True),
        allow_quality_review_route=_env_bool(
            "ORCHESTRATION_ALLOW_QUALITY_REVIEW_ROUTE",
            True,
        ),
    )


def _scalar_spacing(pixel_spacing_mm: Any) -> Optional[float]:
    if pixel_spacing_mm is None:
        return None

    if isinstance(pixel_spacing_mm, (int, float)):
        return float(pixel_spacing_mm)

    if isinstance(pixel_spacing_mm, (list, tuple)) and len(pixel_spacing_mm) == 2:
        x = float(pixel_spacing_mm[0])
        y = float(pixel_spacing_mm[1])
        if abs(x - y) > 1e-9:
            raise ValueError(
                "Current src.orchestration.ScanInput stores one scalar pixel spacing. "
                "Anisotropic manual spacing was supplied; preserve it only after the "
                "existing ML orchestration is upgraded to support row/column spacing."
            )
        return x

    raise TypeError(f"Unsupported pixel_spacing_mm value: {pixel_spacing_mm!r}")


# def _normalize_roi(roi_xyxy: Any) -> Optional[Tuple[int, int, int, int]]:
#     if roi_xyxy is None:
#         return None
#     if not isinstance(roi_xyxy, (list, tuple)) or len(roi_xyxy) != 4:
#         raise ValueError("roi_xyxy must contain exactly four values: x0,y0,x1,y1")

#     vals = tuple(int(round(float(v))) for v in roi_xyxy)
#     x0, y0, x1, y1 = vals
#     if x1 <= x0 or y1 <= y0:
#         raise ValueError(f"Invalid roi_xyxy: {vals!r}")
#     return vals


def _normalize_roi(
    roi_xyxy,
):
    if roi_xyxy is None:
        return None

    # React/FastAPI representation:
    # {x, y, width, height}
    if isinstance(roi_xyxy, dict):

        if all(
            k in roi_xyxy
            for k in (
                "x",
                "y",
                "width",
                "height",
            )
        ):
            x0 = int(
                round(
                    float(
                        roi_xyxy["x"]
                    )
                )
            )

            y0 = int(
                round(
                    float(
                        roi_xyxy["y"]
                    )
                )
            )

            width = int(
                round(
                    float(
                        roi_xyxy["width"]
                    )
                )
            )

            height = int(
                round(
                    float(
                        roi_xyxy["height"]
                    )
                )
            )

            x1 = x0 + width
            y1 = y0 + height

        elif all(
            k in roi_xyxy
            for k in (
                "x0",
                "y0",
                "x1",
                "y1",
            )
        ):
            x0 = int(
                round(
                    float(
                        roi_xyxy["x0"]
                    )
                )
            )

            y0 = int(
                round(
                    float(
                        roi_xyxy["y0"]
                    )
                )
            )

            x1 = int(
                round(
                    float(
                        roi_xyxy["x1"]
                    )
                )
            )

            y1 = int(
                round(
                    float(
                        roi_xyxy["y1"]
                    )
                )
            )

        else:
            raise ValueError(
                "ROI dict must contain either "
                "{x,y,width,height} or "
                "{x0,y0,x1,y1}"
            )

    elif (
        isinstance(
            roi_xyxy,
            (list, tuple),
        )
        and
        len(roi_xyxy) == 4
    ):

        x0, y0, x1, y1 = (
            int(
                round(float(v))
            )
            for v in roi_xyxy
        )

    else:
        raise ValueError(
            "ROI must be a dict or "
            "four-element sequence."
        )

    if (
        x0 < 0
        or y0 < 0
        or x1 <= x0
        or y1 <= y0
    ):
        raise ValueError(
            "Invalid ROI: "
            f"{(x0, y0, x1, y1)}"
        )

    return (
        x0,
        y0,
        x1,
        y1,
    )

def _route_to_anatomy(route: Dict[str, Any], plane: Dict[str, Any], expected: str) -> str:
    route_name = str(route.get("route") or "").strip().lower()
    mapped = {
        "head_biometry": "HEAD",
        "abdomen_biometry": "ABDOMEN",
        "femur_biometry": "FEMUR",
        "head": "HEAD",
        "abdomen": "ABDOMEN",
        "femur": "FEMUR",
    }.get(route_name)
    if mapped:
        return mapped

    app_plane = str(plane.get("application_plane") or "").strip().lower()
    mapped = {
        "head": "HEAD",
        "abdomen": "ABDOMEN",
        "femur": "FEMUR",
        "unsupported": "UNSUPPORTED",
        "unknown": "UNSUPPORTED",
    }.get(app_plane)
    if mapped:
        return mapped

    return expected if expected in {"HEAD", "ABDOMEN", "FEMUR"} else "UNSUPPORTED"


def _status_from_existing(
    fusion: Dict[str, Any],
    measurement: Dict[str, Any],
    route: Dict[str, Any],
) -> str:
    candidates: Iterable[Any] = (
        fusion.get("status"),
        fusion.get("decision"),
        fusion.get("final_status"),
        fusion.get("final_decision"),
        fusion.get("result"),
        measurement.get("status"),
    )

    for value in candidates:
        if value is None:
            continue
        text = str(value).strip().upper()
        if text in {"ACCEPT", "PASS", "GOOD", "VALID"}:
            return "ACCEPT"
        if text in {"REVIEW", "WARN", "WARNING"}:
            return "REVIEW"
        if text in {"REJECT", "FAIL", "FAILED", "POOR", "INVALID"}:
            return "REJECT"

    if bool(fusion.get("discard_measurement")):
        return "REJECT"
    if str(route.get("route_status") or "").lower() != "routed":
        return "REVIEW"
    return "REVIEW"


def _plane_for_platform(plane: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(plane or {})
    label = out.get("application_plane") or out.get("predicted_plane")
    if label is not None:
        out.setdefault("label", str(label).upper())
    if out.get("top_score") is not None:
        out.setdefault("confidence", out.get("top_score"))
    out.setdefault("advisory", True)
    return out


def _quality_for_platform(quality: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(quality or {})
    if out.get("score") is not None:
        out.setdefault("confidence", out.get("score"))
    out.setdefault("advisory", True)
    return out


def _measurement_payload(measurement: Dict[str, Any], anatomy: str) -> Dict[str, Any]:
    m = dict(measurement or {})
    if anatomy == "HEAD":
        keys = (
            "hc_mm",
            "bpd_mm",
            "ofd_mm",
            "head_area_mm2",
            "ga_estimate",
        )
    elif anatomy == "ABDOMEN":
        # Keep TAD/APAD as evidence as well; the SaaS normalizer persists AC.
        keys = ("tad_mm", "apad_mm", "ac_mm")
    elif anatomy == "FEMUR":
        keys = ("fl_mm",)
    else:
        keys = ()
    return {k: m[k] for k in keys if k in m and m[k] is not None}


def _collect_artifacts(measurement: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    def absorb(source: Any) -> None:
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            if not value:
                continue
            if key in {"overlay", "mask", "preview"}:
                result[key] = value
            elif key in {"overlay_path", "mask_path", "preview_path"}:
                result[key] = value
            elif key.endswith("_path") and isinstance(value, (str, Path)):
                result[key] = str(value)

    absorb(measurement.get("artifacts"))
    phase0 = measurement.get("phase0_payload")
    if isinstance(phase0, dict):
        absorb(phase0.get("artifacts"))
    absorb(measurement)
    return result


# def _warnings(
#     plane: Dict[str, Any],
#     quality: Dict[str, Any],
#     route: Dict[str, Any],
#     measurement: Dict[str, Any],
#     fusion: Dict[str, Any],
# ) -> list[str]:
#     out: list[str] = []

#     for source in (route, measurement, fusion):
#         if not isinstance(source, dict):
#             continue
#         for key in ("reasons", "reason_codes", "warnings", "flags"):
#             value = source.get(key)
#             if isinstance(value, str):
#                 out.append(value)
#             elif isinstance(value, (list, tuple, set)):
#                 out.extend(str(x) for x in value if x)

#     if str(plane.get("status") or "").lower() == "review":
#         out.append("Plane detector requested review.")
#     if str(quality.get("label") or "").lower() in {"review", "poor"}:
#         out.append(f"Image quality={quality.get('label')}.")

#     # stable de-duplication
#     return list(dict.fromkeys(x for x in out if x))



def _warnings(
    plane,
    quality,
    route,
    measurement,
    fusion,
):

    out = []

    for source in (
        route,
        measurement,
        fusion,
    ):

        if not isinstance(
            source,
            dict,
        ):
            continue

        for key in (
            "reasons",
            "reason_codes",
            "warnings",
            "flags",
        ):

            value = source.get(
                key
            )

            if isinstance(
                value,
                str,
            ):

                out.append(value)

            elif isinstance(
                value,
                (
                    list,
                    tuple,
                    set,
                ),
            ):

                out.extend(
                    str(x)
                    for x in value
                    if x
                )

    if (
        str(
            plane.get(
                "status"
            )
            or ""
        ).lower()
        == "review"
    ):

        out.append(
            "Plane detector "
            "requested review."
        )

    qlabel = str(
        quality.get(
            "label"
        )
        or ""
    ).lower()

    if qlabel in {
        "review",
        "poor",
        "unavailable",
    }:

        if not bool(
            quality.get(
                "policy_applicable",
                True,
            )
        ):

            anatomy = str(
                quality.get(
                    "scope_anatomy"
                )
                or "this anatomy"
            ).upper()

            out.append(
                "Image quality model "
                f"returned {qlabel}, "
                "but the current quality "
                "model is advisory-only "
                f"for {anatomy}."
            )

        else:

            out.append(
                f"Image quality={qlabel}."
            )

    return list(
        dict.fromkeys(
            x
            for x in out
            if x
        )
    )


def run_scan(
    input_path: str,
    anatomy: str,
    pixel_spacing_mm: Any = None,
    roi_xyxy: Any = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """SaaS entrypoint around the existing frozen orchestration pipeline.

    The anatomy argument is the upload slot selected by the application. The existing
    FetalCLIP/router still decides the actual anatomy; mismatches remain visible to the
    platform instead of silently forcing the requested slot.
    """
    expected = str(anatomy).upper().strip()
    if expected not in {"HEAD", "ABDOMEN", "FEMUR"}:
        raise ValueError(f"Unsupported expected anatomy: {anatomy!r}")

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    spacing = _scalar_spacing(pixel_spacing_mm)
    roi = _normalize_roi(roi_xyxy)

    scan = ScanInput(
        image_path=str(path),
        pixel_spacing_mm=spacing,
        roi_xyxy=roi,
    )

    raw = _pipeline().run(scan)
    plane = dict(raw.get("plane") or {})
    quality = dict(raw.get("quality") or {})
    route = dict(raw.get("route") or {})
    measurement = dict(raw.get("measurement") or {})
    fusion = dict(raw.get("fusion") or {})

    actual = _route_to_anatomy(route, plane, expected)
    status = _status_from_existing(fusion, measurement, route)

    fusion_out = dict(fusion)
    fusion_out["status"] = status

    provenance = {
        "release": str(raw.get("pipeline_version") or "phase8_v3_anatomy_aware_quality"),
        "orchestration": "src.orchestration.UnifiedFetalBiometryPipeline",
        "backend": "src.orchestration.ProjectBackend",
        "project_backend_config": _config_path(),
        "routing_enabled": _env_bool("ORCHESTRATION_ROUTING_ENABLED", True),
        "allow_quality_review_route": _env_bool(
            "ORCHESTRATION_ALLOW_QUALITY_REVIEW_ROUTE",
            True,
        ),
        "expected_upload_anatomy": expected,
        "pixel_spacing_mm": spacing,
        "roi_xyxy": list(roi) if roi is not None else None,
        "quality_policy_applicable":
    quality.get(
        "policy_applicable"
    ),

"quality_affects_routing":
    quality.get(
        "affects_routing"
    ),

"quality_affects_fusion":
    quality.get(
        "affects_fusion"
    ),

"quality_scope":
    quality.get(
        "scope"
    ),
    }

    return {
        "schema_version": "1.0",
        "source": "real",
        "anatomy": actual,
        "route": actual,
        "routing": route,
        "measurements": _measurement_payload(measurement, actual),
        "plane": _plane_for_platform(plane),
        "quality": _quality_for_platform(quality),
        "fusion": fusion_out,
        "warnings": _warnings(plane, quality, route, measurement, fusion_out),
        "artifacts": _collect_artifacts(measurement),
        "provenance": provenance,
        # Keep the frozen orchestration output for technical/debug review. This is
        # JSON-safe in the current implementation and is not used for measurement math.
        "orchestration_result": raw,
    }


def calculate_study_efw(hc_mm: float, ac_mm: float, fl_mm: float) -> Dict[str, Any]:
    """Call the already-integrated deterministic EFW implementation."""
    return _backend().calculate_efw(
        hc_mm=float(hc_mm),
        ac_mm=float(ac_mm),
        fl_mm=float(fl_mm),
    )
