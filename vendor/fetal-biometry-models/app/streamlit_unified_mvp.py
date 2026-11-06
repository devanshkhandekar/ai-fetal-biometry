"""Unified Streamlit MVP for the completed fetal-biometry phases.

Run from the project root:

    conda activate fetal_biometry_unified
    export PYTHONPATH="$PWD:$PWD/external/Multicentre-Fetal-Biometry:$PYTHONPATH"
    streamlit run app/streamlit_unified_mvp.py

Integrated phases:
- Phase 0: head segmentation + ellipse + HC/BPD/OFD/head area + HC-derived GA
- Phase 1: FetalCLIP plane detection + frozen linear probe
- Phase 2: FetalCLIP proxy image-quality probe
- Phase 3: deterministic confidence fusion
- Phase 4: TAD/APAD -> AC
- Phase 5: FL
- Phase 6: safe anatomy routing
- Phase 7: Hadlock HC/AC/FL EFW
- Phase 8: unified orchestration + session-level UI

Research/demo use only. This application is not a clinically validated device.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io.image_loader import load_image  # noqa: E402
from src.orchestration.project_backend import ProjectBackend  # noqa: E402
from src.orchestration.scan_input import ScanInput  # noqa: E402
from src.orchestration.unified_pipeline import UnifiedFetalBiometryPipeline  # noqa: E402


OUTPUTS = ROOT / "outputs"
UPLOAD_ROOT = OUTPUTS / "unified_uploads"
SUPPORTED = ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom"]


st.set_page_config(
    page_title="AI Fetal Biometry — Unified MVP",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading validated fetal-biometry models...")
def get_backend() -> ProjectBackend:
    return ProjectBackend()


def ensure_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
    if "biometry_basket" not in st.session_state:
        st.session_state.biometry_basket = {
            "hc_mm": None,
            "ac_mm": None,
            "fl_mm": None,
            "head": None,
            "abdomen": None,
            "femur": None,
        }
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def session_dir() -> Path:
    path = UPLOAD_ROOT / st.session_state.session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_original_name(name: str) -> str:
    # Preserve basename exactly enough for FP metadata lookup while removing path tricks.
    return Path(name).name.replace("/", "_").replace("\\", "_")


def save_upload(uploaded) -> Path:
    target = session_dir() / safe_original_name(uploaded.name)
    target.write_bytes(uploaded.getbuffer())
    return target


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.ndim == 3 and x.shape[-1] > 3:
        x = x[..., :3]

    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            x = np.zeros_like(x, dtype=np.float32)
        else:
            lo = float(np.nanmin(x[finite]))
            hi = float(np.nanmax(x[finite]))
            if hi > lo:
                x = (x - lo) / (hi - lo)
                x = np.clip(x * 255.0, 0, 255)
            else:
                x = np.zeros_like(x)
        x = x.astype(np.uint8)

    return x


def _extract_spacing(meta: Dict[str, Any]) -> Optional[float]:
    """Best-effort lookup without assuming one loader metadata schema."""
    if not isinstance(meta, dict):
        return None

    candidate_keys = {
        "pixel_spacing_mm",
        "spacing_mm",
        "pixel_spacing",
        "mm_per_pixel",
        "mm_per_px",
        "physical_pixel_spacing_mm",
    }

    for key, value in meta.items():
        if str(key).lower() in candidate_keys:
            if isinstance(value, (tuple, list, np.ndarray)):
                vals = []
                for v in value:
                    try:
                        fv = float(v)
                        if np.isfinite(fv) and fv > 0:
                            vals.append(fv)
                    except Exception:
                        pass
                if vals:
                    return float(np.mean(vals))
            try:
                fv = float(value)
                if np.isfinite(fv) and fv > 0:
                    return fv
            except Exception:
                pass

    # Search one nested level; useful for metadata_summary/dicom tags.
    for value in meta.values():
        if isinstance(value, dict):
            found = _extract_spacing(value)
            if found is not None:
                return found

    return None


def prepare_input(
    saved_path: Path,
    manual_spacing_mm: Optional[float],
) -> Tuple[Path, Optional[float], Dict[str, Any]]:
    """Return a normal image path for all downstream models.

    For DICOM:
      existing project loader -> RGB/gray ndarray -> PNG
      preserve physical spacing from loader metadata where available.

    For standard images:
      original path is retained so FP abdomen/femur filename metadata lookup works.
    """
    suffix = saved_path.suffix.lower()
    if suffix not in {".dcm", ".dicom"}:
        return saved_path, manual_spacing_mm, {}

    image, meta = load_image(
        str(saved_path),
        pixel_spacing_mm=manual_spacing_mm,
    )

    rgb = _to_uint8_rgb(image)
    converted = saved_path.with_suffix(".dicom_preview.png")

    # cv2 expects BGR when saving 3-channel arrays.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(converted), bgr)
    if not ok:
        raise RuntimeError(f"Could not write converted DICOM PNG: {converted}")

    spacing = manual_spacing_mm or _extract_spacing(meta or {})
    return converted, spacing, meta or {}


def fmt(value: Any, digits=2, unit="") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{unit}"
    except Exception:
        return str(value)


def decision_ui(status: str):
    s = str(status or "unknown").lower()
    if s == "accept":
        st.success("Final confidence decision: ACCEPT")
    elif s == "review":
        st.warning("Final confidence decision: REVIEW")
    elif s == "reject":
        st.error("Final confidence decision: REJECT")
    else:
        st.info(f"Final confidence decision: {s.upper()}")


def show_image(path: Optional[str], caption: str):
    if not path:
        return
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_column_width=True)


def show_triage(result: Dict[str, Any]):
    plane = result.get("plane") or {}
    quality = result.get("quality") or {}
    route = result.get("route") or {}

    st.subheader("2. AI triage and routing")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Detected plane",
        str(plane.get("predicted_plane") or "unknown").title(),
    )
    c2.metric(
        "Plane score",
        fmt(plane.get("top_score"), 3),
        delta=f"margin {fmt(plane.get('margin'), 3)}",
    )
    c3.metric(
        "Image quality",
        str(quality.get("label") or "unknown").upper(),
    )
    c4.metric(
        "Route",
        str(route.get("route") or route.get("route_status") or "none"),
    )

    if quality.get("mode") == "advisory":
        st.caption(
            "Image-quality classification is an engineering/advisory signal trained "
            "with proxy labels, not clinician-validated image-quality ground truth."
        )

    if route.get("route_status") != "routed":
        reason = route.get("reason", "No supported automatic route.")
        st.warning(f"No anatomy measurement was executed: {reason}")


def show_measurement(result: Dict[str, Any]):
    measurement = result.get("measurement")
    if not measurement:
        return

    plane = (result.get("route") or {}).get("application_plane")
    st.subheader("3. Anatomy measurement")

    if plane == "head":
        cols = st.columns(5)
        cols[0].metric("HC", fmt(measurement.get("hc_mm"), 2, " mm"))
        cols[1].metric("BPD", fmt(measurement.get("bpd_mm"), 2, " mm"))
        cols[2].metric("OFD", fmt(measurement.get("ofd_mm"), 2, " mm"))
        cols[3].metric(
            "Head area",
            fmt(measurement.get("head_area_mm2"), 2, " mm²"),
        )
        ga = measurement.get("ga_estimate")
        if isinstance(ga, dict):
            ga_value = (
                ga.get("ga_display")
                or ga.get("ga_text")
                or ga.get("estimate_text")
                or ga.get("gestational_age")
                or "N/A"
            )
        else:
            ga_value = ga or "N/A"
        cols[4].metric("HC-derived GA", str(ga_value))

        artifacts = measurement.get("artifacts") or {}
        show_image(artifacts.get("overlay_path"), "Head segmentation + ellipse overlay")

    elif plane == "abdomen":
        cols = st.columns(3)
        cols[0].metric("TAD", fmt(measurement.get("tad_mm"), 2, " mm"))
        cols[1].metric("APAD", fmt(measurement.get("apad_mm"), 2, " mm"))
        cols[2].metric("AC", fmt(measurement.get("ac_mm"), 2, " mm"))

        artifacts = measurement.get("artifacts") or {}
        c1, c2 = st.columns(2)
        with c1:
            show_image(artifacts.get("tad_overlay"), "TAD landmarks")
        with c2:
            show_image(artifacts.get("apad_overlay"), "APAD landmarks")

        roi_source = measurement.get("roi_source")
        if roi_source:
            st.caption(
                f"ROI source: `{roi_source}` | "
                f"benchmark preprocessing: `{measurement.get('benchmark_preprocessing')}`"
            )

    elif plane == "femur":
        c1, c2 = st.columns(2)
        c1.metric("Femur length", fmt(measurement.get("fl_mm"), 2, " mm"))
        c2.metric("FL in pixels", fmt(measurement.get("fl_px"), 2, " px"))

        artifacts = measurement.get("artifacts") or {}
        show_image(artifacts.get("fl_overlay"), "Femur endpoints / FL overlay")

        roi_source = measurement.get("roi_source")
        if roi_source:
            st.caption(
                f"ROI source: `{roi_source}` | "
                f"benchmark preprocessing: `{measurement.get('benchmark_preprocessing')}`"
            )

    if measurement.get("reasons"):
        st.warning("Measurement flags: " + ", ".join(measurement["reasons"]))


def show_fusion(result: Dict[str, Any]):
    fusion = result.get("fusion")
    if not fusion:
        return

    st.subheader("4. Confidence fusion")
    decision_ui(fusion.get("status"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Automatic accept", str(bool(fusion.get("automatic_accept"))))
    c2.metric("Review required", str(bool(fusion.get("review_required"))))
    c3.metric("Discard measurement", str(bool(fusion.get("discard_measurement"))))

    reasons = fusion.get("reasons") or []
    warnings = fusion.get("warnings") or []
    if reasons:
        st.write("**Decision reasons:**", reasons)
    if warnings:
        st.write("**Warnings:**", warnings)


def update_basket(result: Dict[str, Any], source_name: str):
    fusion = result.get("fusion") or {}
    measurement = result.get("measurement") or {}
    route = result.get("route") or {}

    if fusion.get("status") not in {"accept", "review"}:
        return

    anatomy = route.get("application_plane")

    if anatomy == "head" and measurement.get("hc_mm") is not None:
        st.session_state.biometry_basket["hc_mm"] = measurement.get("hc_mm")
        st.session_state.biometry_basket["head"] = {
            "source": source_name,
            "status": fusion.get("status"),
            "measurement": measurement,
        }

    elif anatomy == "abdomen" and measurement.get("ac_mm") is not None:
        st.session_state.biometry_basket["ac_mm"] = measurement.get("ac_mm")
        st.session_state.biometry_basket["abdomen"] = {
            "source": source_name,
            "status": fusion.get("status"),
            "measurement": measurement,
        }

    elif anatomy == "femur" and measurement.get("fl_mm") is not None:
        st.session_state.biometry_basket["fl_mm"] = measurement.get("fl_mm")
        st.session_state.biometry_basket["femur"] = {
            "source": source_name,
            "status": fusion.get("status"),
            "measurement": measurement,
        }


def show_basket(backend: ProjectBackend):
    st.subheader("5. Multi-image biometry session")
    basket = st.session_state.biometry_basket

    c1, c2, c3 = st.columns(3)
    c1.metric("HC", fmt(basket.get("hc_mm"), 2, " mm"))
    c2.metric("AC", fmt(basket.get("ac_mm"), 2, " mm"))
    c3.metric("FL", fmt(basket.get("fl_mm"), 2, " mm"))

    details = []
    for anatomy in ("head", "abdomen", "femur"):
        entry = basket.get(anatomy)
        if entry:
            details.append({
                "anatomy": anatomy,
                "source": entry.get("source"),
                "fusion_status": entry.get("status"),
            })
    if details:
        st.dataframe(pd.DataFrame(details), use_container_width=True)

    hc = basket.get("hc_mm")
    ac = basket.get("ac_mm")
    fl = basket.get("fl_mm")

    if hc is not None and ac is not None and fl is not None:
        efw = backend.calculate_efw(hc_mm=hc, ac_mm=ac, fl_mm=fl)
        if efw.get("status") == "calculated":
            st.success(
                "Estimated fetal weight available: "
                f"{fmt(efw.get('efw_g'), 1, ' g')} "
                f"({fmt(efw.get('efw_kg'), 3, ' kg')})"
            )
            with st.expander("EFW formula/audit"):
                st.json(efw)
        else:
            st.warning(f"EFW unavailable: {efw.get('reason')}")
    else:
        missing = [
            name.upper()
            for name, value in (("hc", hc), ("ac", ac), ("fl", fl))
            if value is None
        ]
        st.info(
            "EFW is calculated only after valid physical measurements for "
            + ", ".join(missing)
            + " are added to this session."
        )


def download_section():
    st.subheader("6. Technical result / downloads")

    result = st.session_state.last_result
    if result:
        st.download_button(
            "Download latest result JSON",
            data=json.dumps(result, indent=2, default=str).encode("utf-8"),
            file_name="latest_fetal_biometry_result.json",
            mime="application/json",
        )

    session_payload = {
        "session_id": st.session_state.session_id,
        "biometry_basket": st.session_state.biometry_basket,
        "history": st.session_state.history,
    }
    st.download_button(
        "Download session JSON",
        data=json.dumps(session_payload, indent=2, default=str).encode("utf-8"),
        file_name="fetal_biometry_session.json",
        mime="application/json",
    )

    if result:
        with st.expander("Latest complete pipeline JSON"):
            st.json(result)


def main():
    ensure_session()

    st.title("AI Fetal Biometry — Unified MVP")
    st.caption(
        "Plane detection → image-quality triage → safe anatomy routing → "
        "head/abdomen/femur biometry → confidence fusion → session-level EFW"
    )
    st.warning(
        "Research/demo use only. This system is not a clinically validated "
        "diagnostic or medical-device workflow."
    )

    with st.sidebar:
        st.header("Pipeline")
        routing_enabled = st.checkbox(
            "Enable automatic anatomy routing",
            value=True,
            help="Uses the validated Phase-1 policy and Phase-6 router.",
        )

        st.header("Physical calibration")
        use_manual_spacing = st.checkbox(
            "Provide manual pixel spacing",
            value=False,
        )
        manual_spacing = None
        if use_manual_spacing:
            manual_spacing = st.number_input(
                "Pixel spacing (mm / pixel)",
                min_value=0.001,
                max_value=5.0,
                value=0.10,
                step=0.001,
                format="%.4f",
            )

        st.header("Optional ROI")
        use_roi = st.checkbox(
            "Provide ROI for arbitrary abdomen/femur image",
            value=False,
            help=(
                "Known FP benchmark images find center/scale automatically by filename. "
                "For other abdomen/femur images, an explicit ROI is safer than full-image fallback."
            ),
        )
        roi = None
        if use_roi:
            x1 = st.number_input("ROI x1", min_value=0, value=0)
            y1 = st.number_input("ROI y1", min_value=0, value=0)
            x2 = st.number_input("ROI x2", min_value=1, value=256)
            y2 = st.number_input("ROI y2", min_value=1, value=256)
            roi = (int(x1), int(y1), int(x2), int(y2))

        st.divider()

        if st.button("Reset biometry session"):
            st.session_state.biometry_basket = {
                "hc_mm": None,
                "ac_mm": None,
                "fl_mm": None,
                "head": None,
                "abdomen": None,
                "femur": None,
            }
            st.session_state.history = []
            st.session_state.last_result = None
            st.session_state.session_id = uuid.uuid4().hex[:12]
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

    backend = get_backend()
    pipeline = UnifiedFetalBiometryPipeline(
        backend,
        routing_enabled=routing_enabled,
        allow_quality_review_route=True,
    )

    st.subheader("1. Upload fetal ultrasound")
    uploaded = st.file_uploader(
        "Upload one head, abdomen, or femur ultrasound image",
        type=SUPPORTED,
    )

    if uploaded is None:
        st.info(
            "For EFW, process one valid head image, one valid abdomen image, "
            "and one valid femur image in the same Streamlit session."
        )
        show_basket(backend)
        return

    saved = save_upload(uploaded)

    try:
        prepared_path, spacing, dicom_meta = prepare_input(
            saved,
            manual_spacing_mm=manual_spacing,
        )
    except Exception as exc:
        st.error(f"Input preparation failed: {exc}")
        st.stop()

    left, right = st.columns([1.3, 1])
    with left:
        if prepared_path.exists():
            st.image(
                str(prepared_path),
                caption=uploaded.name,
                use_column_width=True,
            )
    with right:
        st.write("**Saved input:**", str(saved))
        st.write("**Model input:**", str(prepared_path))
        st.write(
            "**Pixel spacing:**",
            f"{spacing:.4f} mm/pixel" if spacing is not None else "not available",
        )
        if roi:
            st.write("**Manual ROI:**", roi)
        if dicom_meta:
            with st.expander("DICOM/input metadata"):
                st.json(dicom_meta)

    if not st.button("Run complete AI biometry pipeline", type="primary"):
        show_basket(backend)
        return

    scan = ScanInput(
        image_path=str(prepared_path),
        pixel_spacing_mm=spacing,
        roi_xyxy=roi,
    )

    try:
        with st.spinner(
            "Running plane detection, quality triage, routing and anatomy biometry..."
        ):
            result = pipeline.run(scan)
    except Exception as exc:
        st.error("Unified pipeline failed.")
        st.exception(exc)
        st.stop()

    st.session_state.last_result = result

    history_row = {
        "source": uploaded.name,
        "detected_plane": (result.get("plane") or {}).get("predicted_plane"),
        "quality": (result.get("quality") or {}).get("label"),
        "route": (result.get("route") or {}).get("route"),
        "route_status": (result.get("route") or {}).get("route_status"),
        "fusion": (result.get("fusion") or {}).get("status"),
    }
    st.session_state.history.append(history_row)

    update_basket(result, source_name=uploaded.name)

    show_triage(result)
    show_measurement(result)
    show_fusion(result)
    show_basket(backend)
    download_section()


if __name__ == "__main__":
    main()
