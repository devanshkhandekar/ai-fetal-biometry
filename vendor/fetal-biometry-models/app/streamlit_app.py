"""Streamlit dashboard for the fetal head biometry Core MVP.

Run from project root:
    streamlit run app/streamlit_app.py

Expected checkpoints:
    models/checkpoints/4a_best_pruned_ft_v10.pth
    models/checkpoints/phase0_model.pth

This version displays HC-derived GA in the dashboard and in all report
previews/downloads. It is intentionally model-provider neutral and uses the
current devansh_* backend modules.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline_devansh import run_case_devansh  # noqa: E402
from src.segmentation.devansh_resunet import load_devansh_static_checkpoint  # noqa: E402
from src.reporting.pdf_report import build_simple_pdf_report  # noqa: E402

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"

st.set_page_config(
    page_title="Fetal Head Biometry MVP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def cached_model_bundle(checkpoint_path: str, variant: str):
    """Cache the pretrained static segmentation model across Streamlit reruns."""
    return load_devansh_static_checkpoint(checkpoint_path, variant=variant)


def ensure_dirs() -> None:
    for sub in ["uploads", "predictions", "overlays", "reports", "logs"]:
        (DEFAULT_OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    ensure_dirs()
    suffix = Path(uploaded_file.name).suffix.lower()
    safe_stem = Path(uploaded_file.name).stem.replace(" ", "_")
    out_path = DEFAULT_OUTPUT_DIR / "uploads" / f"{safe_stem}_{int(time.time())}{suffix}"
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


def find_checkpoints() -> List[Path]:
    if not DEFAULT_CHECKPOINT_DIR.exists():
        return []
    return sorted(DEFAULT_CHECKPOINT_DIR.glob("*.pth")) + sorted(DEFAULT_CHECKPOINT_DIR.glob("*.pt"))


def read_image_for_display(path: str | Path) -> Optional[np.ndarray]:
    path = Path(path)
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def format_value(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}{suffix}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}{suffix}"
    return str(value)


def decision_badge(decision: str) -> str:
    decision = (decision or "unknown").lower()
    if decision == "accept":
        return "✅ ACCEPT"
    if decision == "warn":
        return "⚠️ REVIEW"
    if decision == "reject":
        return "❌ REJECT"
    return "ℹ️ UNKNOWN"


def normalize_ga_dict(ga: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize GA payloads from either old chart lookup or HC-polynomial lookup.

    Supported incoming formats:
      - {ga_display: "21w 5d", trimester: ..., method: ...}
      - {ga_text: "21w 5d", ...}
      - {estimate_text: "21w 5d", ...}
      - {ga_weeks: 21, ga_days: 5, ga_weeks_decimal: 21.714, ...}
    """
    if not isinstance(ga, dict) or not ga:
        return {}

    out = dict(ga)
    display = (
        out.get("ga_display")
        or out.get("ga_text")
        or out.get("estimate_text")
        or out.get("gestational_age")
    )

    weeks = out.get("ga_weeks")
    days = out.get("ga_days")
    decimal_weeks = out.get("ga_weeks_decimal")

    if display is None and weeks is not None:
        try:
            w = int(weeks)
            d = int(days or 0)
            display = f"{w}w {d}d"
        except Exception:
            display = str(weeks)

    if decimal_weeks is None and weeks is not None:
        try:
            decimal_weeks = float(weeks) + (float(days or 0) / 7.0)
        except Exception:
            decimal_weeks = None

    trimester = out.get("trimester")
    if trimester is None and decimal_weeks is not None:
        try:
            ga_days_total = float(decimal_weeks) * 7.0
            if ga_days_total < 98:
                trimester = "First trimester (<14w)"
            elif ga_days_total < 196:
                trimester = "Second trimester (14w–27w6d)"
            else:
                trimester = "Third trimester (≥28w)"
        except Exception:
            trimester = None

    out["ga_display"] = display or "N/A"
    out["ga_weeks_decimal"] = decimal_weeks
    out["trimester"] = trimester or "N/A"
    out.setdefault("method", "HC_TO_GA_ESTIMATION")
    out.setdefault(
        "source_note",
        "Gestational age is estimated from head circumference only and must be reviewed by a licensed doctor.",
    )
    return out


def get_ga_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    measurements = payload.get("measurements", {}) or {}
    return normalize_ga_dict(measurements.get("ga_estimate"))


def flatten_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    m = payload.get("measurements", {}) or {}
    q = payload.get("quality", {}) or {}
    c = payload.get("confidence", {}) or {}
    d = payload.get("decision", {}) or {}
    e = payload.get("ellipse_parameters", {}) or {}
    inp = payload.get("input_info", {}) or {}
    audit = payload.get("audit", {}) or {}
    ga = get_ga_from_payload(payload)
    return {
        "filename": inp.get("filename"),
        "decision": d.get("decision"),
        "hc_mm": m.get("hc_mm"),
        "bpd_mm": m.get("bpd_mm"),
        "ofd_mm": m.get("ofd_mm"),
        "head_area_mm2": m.get("head_area_mm2"),
        "hc_px": m.get("hc_px"),
        "ga_estimate": ga.get("ga_display"),
        "ga_weeks_decimal": ga.get("ga_weeks_decimal"),
        "trimester": ga.get("trimester"),
        "quality_score": q.get("quality_score"),
        "quality_label": q.get("quality_label"),
        "confidence_score": c.get("confidence_score"),
        "confidence_band": c.get("confidence_band"),
        "ellipse_major_axis_px": e.get("major_axis_px") if e else None,
        "ellipse_minor_axis_px": e.get("minor_axis_px") if e else None,
        "runtime_ms": audit.get("runtime_ms"),
        "reason_codes": "; ".join(d.get("reason_codes", []) or []),
    }


def metric_value(measurements: Dict[str, Any], mm_key: str, px_key: str, unit: str) -> str:
    if measurements.get(mm_key) is not None:
        return format_value(measurements.get(mm_key), 2, f" {unit}")
    return format_value(measurements.get(px_key), 2, " px")


def show_top_metrics(payload: Dict[str, Any]) -> None:
    decision = payload.get("decision", {}) or {}
    measurements = payload.get("measurements", {}) or {}
    quality = payload.get("quality", {}) or {}
    confidence = payload.get("confidence", {}) or {}
    ga = get_ga_from_payload(payload)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", decision_badge(decision.get("decision")))
    c2.metric("Quality", f"{format_value(quality.get('quality_score'), 2)} / 100", quality.get("quality_label", ""))
    c3.metric("Confidence", f"{format_value(confidence.get('confidence_score'), 2)} / 100", confidence.get("confidence_band", ""))
    c4.metric("Scale", "mm available" if measurements.get("scale_available") else "pixel only")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("HC", metric_value(measurements, "hc_mm", "hc_px", "mm"))
    m2.metric("BPD", metric_value(measurements, "bpd_mm", "bpd_px", "mm"))
    m3.metric("OFD", metric_value(measurements, "ofd_mm", "ofd_px", "mm"))
    m4.metric("Head Area", metric_value(measurements, "head_area_mm2", "head_area_px2", "mm²"))
    m5.metric("Estimated GA", ga.get("ga_display", "N/A"))


def show_overlay_and_geometry(payload: Dict[str, Any]) -> None:
    artifacts = payload.get("artifacts", {}) or {}
    ellipse = payload.get("ellipse_parameters", {}) or {}
    left, right = st.columns([1.25, 1])
    with left:
        overlay_path = artifacts.get("overlay_path")
        overlay = read_image_for_display(overlay_path) if overlay_path else None
        if overlay is not None:
            # use_column_width is kept for compatibility with older Streamlit builds.
            st.image(overlay, caption="Mask + ellipse overlay", use_column_width=True)
        else:
            st.info("Overlay image was not generated or could not be read.")
    with right:
        st.markdown("#### Ellipse geometry")
        if ellipse:
            st.dataframe(pd.DataFrame([ellipse]).T.rename(columns={0: "value"}), use_container_width=True)
        else:
            st.warning("Ellipse could not be fitted for this scan.")


def show_patient_tab(payload: Dict[str, Any]) -> None:
    previews = payload.get("report_previews", {}) or {}
    patient = previews.get("patient", {}) or {}
    measurements = payload.get("measurements", {}) or {}
    ga = get_ga_from_payload(payload)

    st.subheader("Patient Summary")
    st.write(patient.get("summary", "The uploaded scan was processed by the fetal head biometry workflow."))
    st.info(patient.get("disclaimer", "AI result must be reviewed by a licensed doctor."))

    rows = {
        "HC": metric_value(measurements, "hc_mm", "hc_px", "mm"),
        "BPD": metric_value(measurements, "bpd_mm", "bpd_px", "mm"),
        "OFD": metric_value(measurements, "ofd_mm", "ofd_px", "mm"),
        "Head Area": metric_value(measurements, "head_area_mm2", "head_area_px2", "mm²"),
        "Estimated GA": ga.get("ga_display", "N/A"),
        "Trimester": ga.get("trimester", "N/A"),
        "Quality": (payload.get("quality", {}) or {}).get("quality_label"),
        "Decision": (payload.get("decision", {}) or {}).get("decision"),
    }
    st.table(pd.DataFrame(rows.items(), columns=["Field", "Value"]))
    st.caption("GA is estimated from head circumference only and is not a replacement for clinician review.")


def show_doctor_tab(payload: Dict[str, Any]) -> None:
    st.subheader("Doctor Review")
    show_overlay_and_geometry(payload)

    st.markdown("#### Full measurement table")
    measurements = payload.get("measurements", {}) or {}
    ga = get_ga_from_payload(payload)
    measurement_display = dict(measurements)
    measurement_display["ga_display"] = ga.get("ga_display", "N/A")
    measurement_display["ga_weeks_decimal"] = ga.get("ga_weeks_decimal")
    measurement_display["trimester"] = ga.get("trimester", "N/A")
    measurement_display["ga_method"] = ga.get("method", "N/A")
    st.dataframe(pd.DataFrame([measurement_display]).T.rename(columns={0: "value"}), use_container_width=True)

    meta = (payload.get("input_info", {}) or {}).get("metadata_summary")
    if meta:
        st.markdown("#### DICOM metadata summary")
        st.dataframe(pd.DataFrame([meta]).T.rename(columns={0: "value"}), use_container_width=True)

    reason_codes = (payload.get("decision", {}) or {}).get("reason_codes", []) or []
    warnings = (payload.get("quality", {}) or {}).get("warnings", []) or []
    if reason_codes or warnings:
        st.warning("Review flags: " + ", ".join(sorted(set(reason_codes + warnings))))


def show_growth_tab(payload: Dict[str, Any]) -> None:
    st.subheader("GA / Growth-Ready Output")
    measurements = payload.get("measurements", {}) or {}
    ga = get_ga_from_payload(payload)

    if not ga:
        st.info("GA estimate is unavailable because HC in mm is unavailable. Provide DICOM pixel spacing or manual pixel spacing.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated GA", ga.get("ga_display", "N/A"))
    c2.metric("GA weeks", format_value(ga.get("ga_weeks_decimal"), 2))
    c3.metric("Trimester", ga.get("trimester", "N/A"))

    st.markdown("#### GA details")
    st.json(ga)
    st.warning("GA is estimated from HC only. It must be reviewed by a licensed doctor and should not be treated as a full multi-parameter dating result.")

    hc = measurements.get("hc_mm")
    if hc is not None:
        df = pd.DataFrame({"measurement": ["HC"], "value_mm": [hc]})
        st.bar_chart(df.set_index("measurement"))

    st.caption("For production clinical use, the reference method/chart must be approved and versioned.")


def show_technical_tab(payload: Dict[str, Any]) -> None:
    st.subheader("Technical Audit")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Model")
        st.json(payload.get("model_info", {}) or {})
        st.markdown("#### Preprocessing")
        st.json(payload.get("preprocessing", {}) or {})
    with col2:
        st.markdown("#### Audit")
        st.json(payload.get("audit", {}) or {})
        st.markdown("#### Validity gate")
        st.json(payload.get("validity_gate", {}) or {})

    st.markdown("#### GA calculation audit")
    ga = get_ga_from_payload(payload)
    if ga:
        st.json({
            "ga_display": ga.get("ga_display"),
            "ga_weeks_decimal": ga.get("ga_weeks_decimal"),
            "trimester": ga.get("trimester"),
            "method": ga.get("method"),
            "source_note": ga.get("source_note"),
        })
    else:
        st.info("No GA calculation was generated for this scan.")


def show_downloads(payload: Dict[str, Any]) -> None:
    st.subheader("Downloads")
    artifacts = payload.get("artifacts", {}) or {}

    json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
    st.download_button(
        "Download structured JSON",
        data=json_bytes,
        file_name="fetal_biometry_report.json",
        mime="application/json",
    )

    summary_csv = pd.DataFrame([flatten_summary(payload)]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV summary",
        data=summary_csv,
        file_name="fetal_biometry_summary.csv",
        mime="text/csv",
    )

    overlay_path = artifacts.get("overlay_path")
    if overlay_path and Path(overlay_path).exists():
        st.download_button(
            "Download overlay PNG",
            data=Path(overlay_path).read_bytes(),
            file_name=Path(overlay_path).name,
            mime="image/png",
        )

    pdf_cols = st.columns(3)
    for idx, report_type in enumerate(["patient", "doctor", "technical"]):
        try:
            pdf_bytes = build_simple_pdf_report(payload, report_type=report_type)
            pdf_cols[idx].download_button(
                f"Download {report_type.title()} PDF",
                data=pdf_bytes,
                file_name=f"{report_type}_fetal_biometry_report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            pdf_cols[idx].caption(f"PDF unavailable: {exc}")


def show_payload_result(payload: Dict[str, Any]) -> None:
    st.subheader("MVP Result")
    show_top_metrics(payload)
    tabs = st.tabs(["Doctor Review", "Patient Summary", "GA / Growth", "Technical Audit", "JSON / Downloads"])
    with tabs[0]:
        show_doctor_tab(payload)
    with tabs[1]:
        show_patient_tab(payload)
    with tabs[2]:
        show_growth_tab(payload)
    with tabs[3]:
        show_technical_tab(payload)
    with tabs[4]:
        show_downloads(payload)
        with st.expander("Full structured payload"):
            st.json(payload)


def run_single_case_ui(checkpoint_path: str, variant: str, threshold: float, pixel_spacing: Optional[float]):
    st.header("1. Single scan inference")
    uploaded = st.file_uploader(
        "Upload fetal head ultrasound image or DICOM",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom"],
    )
    if uploaded is None:
        st.info("Upload one ultrasound image to run the complete MVP pipeline.")
        return

    saved_path = save_uploaded_file(uploaded)
    st.caption(f"Saved temporary input: `{saved_path}`")

    if not st.button("Run MVP pipeline", type="primary"):
        return

    if not checkpoint_path or not Path(checkpoint_path).exists():
        st.error("Model checkpoint path is missing or invalid.")
        return

    with st.spinner("Running segmentation, ellipse fitting, measurements, GA, quality/confidence and report generation..."):
        model_bundle = cached_model_bundle(checkpoint_path, variant)
        payload = run_case_devansh(
            image_path=str(saved_path),
            checkpoint_path=checkpoint_path,
            model_variant=variant,
            output_dir=str(DEFAULT_OUTPUT_DIR),
            pixel_spacing_mm=pixel_spacing,
            model_bundle=model_bundle,
            threshold=threshold,
        )
    show_payload_result(payload)


def list_batch_files(input_dir: Path) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    return [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and "Annotation" not in p.name
    ]


def run_batch_ui(checkpoint_path: str, variant: str, threshold: float, pixel_spacing: Optional[float]):
    st.header("2. Batch demo")
    input_dir_text = st.text_input("Folder containing sample images", value=str(ROOT_DIR / "data" / "sample_cases"))
    max_cases = st.number_input("Max cases to process", min_value=1, max_value=200, value=10, step=1)
    files = list_batch_files(Path(input_dir_text))[: int(max_cases)]
    st.caption(f"Found {len(files)} supported image/DICOM file(s).")
    if files:
        with st.expander("Files to process"):
            st.write([str(p) for p in files])
    if not st.button("Run batch inference"):
        return
    if not files:
        st.error("No supported files found in the selected folder.")
        return
    if not checkpoint_path or not Path(checkpoint_path).exists():
        st.error("Model checkpoint path is missing or invalid.")
        return

    model_bundle = cached_model_bundle(checkpoint_path, variant)
    rows = []
    progress = st.progress(0)
    status = st.empty()
    for idx, path in enumerate(files, start=1):
        status.write(f"Processing {idx}/{len(files)}: `{path.name}`")
        try:
            payload = run_case_devansh(
                image_path=str(path),
                checkpoint_path=checkpoint_path,
                model_variant=variant,
                output_dir=str(DEFAULT_OUTPUT_DIR),
                pixel_spacing_mm=pixel_spacing,
                model_bundle=model_bundle,
                threshold=threshold,
            )
            rows.append(flatten_summary(payload))
        except Exception as exc:
            rows.append({"filename": path.name, "decision": "error", "reason_codes": str(exc)})
        progress.progress(idx / len(files))

    df = pd.DataFrame(rows)
    st.subheader("Batch summary")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download batch CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="batch_biometry_summary.csv",
        mime="text/csv",
    )


def main():
    ensure_dirs()
    st.title("Fetal Head Biometry MVP")
    st.caption("Core fetal head workflow: upload → segmentation → ellipse geometry → HC/BPD/OFD/area → GA → quality/confidence → reports.")

    with st.sidebar:
        st.header("Model settings")
        detected = find_checkpoints()
        detected_options = [str(p) for p in detected]
        default_ckpt = detected_options[0] if detected_options else str(DEFAULT_CHECKPOINT_DIR / "4a_best_pruned_ft_v10.pth")
        checkpoint_path = st.selectbox(
            "Checkpoint path",
            options=detected_options if detected_options else [default_ckpt],
            index=0,
        )
        checkpoint_path = st.text_input("Or custom checkpoint path", value=checkpoint_path).strip() or checkpoint_path
        variant = st.selectbox("Model variant", options=["phase4a", "phase0"], index=0)
        threshold = st.slider("Mask threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)

        st.header("Measurement settings")
        use_spacing = st.checkbox("Use manual pixel spacing", value=False)
        pixel_spacing = None
        if use_spacing:
            pixel_spacing = st.number_input(
                "Pixel spacing in mm / pixel",
                min_value=0.001,
                max_value=5.0,
                value=0.10,
                step=0.001,
                format="%.4f",
            )

        st.header("Clinical-use note")
        st.info("MVP/research demo only. AI result must be reviewed by a licensed doctor.")

    tab_single, tab_batch, tab_arch = st.tabs(["Single Scan", "Batch Demo", "MVP Architecture"])
    with tab_single:
        run_single_case_ui(checkpoint_path, variant, threshold, pixel_spacing)
    with tab_batch:
        run_batch_ui(checkpoint_path, variant, threshold, pixel_spacing)
    with tab_arch:
        st.subheader("Implemented MVP flow")
        st.code(
            "scan input → preprocessing/validity gate → pretrained segmentation → mask cleanup → ellipse fitting → "
            "HC/BPD/OFD/head area → HC-derived GA → quality/confidence → overlay/report JSON/PDF",
            language="text",
        )
        st.markdown(
            "**Backend modules:** `src.pipeline_devansh`, `src.segmentation.devansh_resunet`, `src.postprocess`, "
            "`src.biometry`, `src.quality`, `src.visualization`, `src.reporting`."
        )


if __name__ == "__main__":
    main()
