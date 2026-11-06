from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pydicom


def _get(ds, name, default=None):
    value = getattr(ds, name, default)
    try:
        return str(value) if value is not None else default
    except Exception:
        return default


def load_dicom(path: str | Path) -> Tuple[np.ndarray, Dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM not found: {path}")

    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)

    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    spacing = None
    spacing_raw = None
    if hasattr(ds, "PixelSpacing"):
        spacing_raw = [float(x) for x in ds.PixelSpacing]
        spacing = float(np.mean(spacing_raw))

    meta = {
        "source_path": str(path),
        "filename": path.name,
        "format": "dicom",
        "original_shape": tuple(arr.shape),
        "pixel_spacing_mm": spacing,
        "pixel_spacing_raw": spacing_raw,
        "has_pixel_spacing": spacing is not None,
        "metadata_summary": {
            "manufacturer": _get(ds, "Manufacturer"),
            "modality": _get(ds, "Modality"),
            "study_date": _get(ds, "StudyDate"),
            "study_time": _get(ds, "StudyTime"),
            "series_date": _get(ds, "SeriesDate"),
            "series_time": _get(ds, "SeriesTime"),
            "acquisition_date": _get(ds, "AcquisitionDate"),
            "acquisition_time": _get(ds, "AcquisitionTime"),
            "patient_id": _get(ds, "PatientID"),
            "study_instance_uid": _get(ds, "StudyInstanceUID"),
            "series_instance_uid": _get(ds, "SeriesInstanceUID"),
            "sop_instance_uid": _get(ds, "SOPInstanceUID"),
        },
    }
    return arr.astype(np.float32), meta
