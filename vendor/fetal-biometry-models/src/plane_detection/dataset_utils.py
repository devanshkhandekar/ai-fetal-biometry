from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import pandas as pd

IMAGE_COLUMN_CANDIDATES = ("Image_name", "image_name", "image", "filename", "file")
PLANE_COLUMN_CANDIDATES = ("Plane", "plane", "label", "class")
PATIENT_COLUMN_CANDIDATES = ("Patient_num", "patient_num", "patient", "patient_id")
SPLIT_COLUMN_CANDIDATES = ("Train", "train", "split", "Split")

CANONICAL_LABELS = ("abdomen", "brain", "femur", "thorax", "cervix", "other")


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the expected columns were found: {tuple(candidates)}. "
                       f"Available columns: {list(df.columns)}")
    return None


# def load_metadata(csv_path: str | Path) -> pd.DataFrame:
#     csv_path = Path(csv_path)
#     if not csv_path.exists():
#         raise FileNotFoundError(csv_path)
#     df = pd.read_csv(csv_path)
#     df.columns = [str(c).strip() for c in df.columns]
#     return df

def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """
    Load FETAL_PLANES_DB metadata.

    The original FETAL_PLANES_DB metadata CSV is semicolon-delimited,
    e.g.:

    Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train

    This loader first lets pandas auto-detect the delimiter and also contains
    a fallback for the common semicolon-separated format.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    # Auto-detect delimiter.
    df = pd.read_csv(
        csv_path,
        sep=None,
        engine="python",
    )

    # Defensive fallback: if pandas somehow reads the entire header as one
    # column and it contains semicolons, reload explicitly with sep=";".
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(
            csv_path,
            sep=";",
        )

    # Clean column names.
    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


# def canonical_plane(value) -> str:
#     s = str(value).strip().lower().replace("_", " ")
#     aliases = {
#         "abdomen": "abdomen",
#         "abdominal": "abdomen",
#         "brain": "brain",
#         "head": "brain",
#         "femur": "femur",
#         "thorax": "thorax",
#         "thoracic": "thorax",
#         "maternal cervix": "cervix",
#         "maternal cervical": "cervix",
#         "cervix": "cervix",
#         "other": "other",
#     }
#     return aliases.get(s, s)


def canonical_plane(value) -> str:
    """
    Convert the original FETAL_PLANES_DB labels into the canonical labels
    used by our Phase-1 pipeline.

    Original dataset labels include:
        Fetal brain
        Fetal abdomen
        Fetal femur
        Fetal thorax
        Cervix
        Other

    Canonical labels:
        brain
        abdomen
        femur
        thorax
        cervix
        other
    """

    s = str(value).strip().lower().replace("_", " ")

    aliases = {
        # Brain / head
        "brain": "brain",
        "fetal brain": "brain",
        "head": "brain",
        "fetal head": "brain",

        # Abdomen
        "abdomen": "abdomen",
        "abdominal": "abdomen",
        "fetal abdomen": "abdomen",
        "fetal abdominal": "abdomen",

        # Femur
        "femur": "femur",
        "fetal femur": "femur",

        # Thorax
        "thorax": "thorax",
        "thoracic": "thorax",
        "fetal thorax": "thorax",
        "fetal thoracic": "thorax",

        # Cervix
        "cervix": "cervix",
        "maternal cervix": "cervix",
        "maternal cervical": "cervix",

        # Other
        "other": "other",
    }

    return aliases.get(s, s)


def normalize_split_value(value) -> str:
    """
    Normalizes common representations of FETAL_PLANES_DB's Train field.
    Returns: "train", "test", or "unknown".
    """
    if pd.isna(value):
        return "unknown"

    # Handle numeric/bool first.
    if isinstance(value, bool):
        return "train" if value else "test"
    try:
        if float(value) == 1.0:
            return "train"
        if float(value) == 0.0:
            return "test"
    except (TypeError, ValueError):
        pass

    s = str(value).strip().lower()
    if s in {"train", "training", "true", "yes", "y"}:
        return "train"
    if s in {"test", "testing", "false", "no", "n"}:
        return "test"
    return "unknown"


def add_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    image_col = find_column(out, IMAGE_COLUMN_CANDIDATES)
    plane_col = find_column(out, PLANE_COLUMN_CANDIDATES)
    patient_col = find_column(out, PATIENT_COLUMN_CANDIDATES, required=False)
    split_col = find_column(out, SPLIT_COLUMN_CANDIDATES, required=False)

    out["_image_name"] = out[image_col].astype(str).str.strip()
    out["_plane"] = out[plane_col].map(canonical_plane)
    out["_patient_id"] = (
        out[patient_col].astype(str).str.strip()
        if patient_col is not None else
        pd.Series([f"row_{i}" for i in range(len(out))], index=out.index)
    )
    out["_split"] = (
        out[split_col].map(normalize_split_value)
        if split_col is not None else
        "unknown"
    )
    return out


def resolve_image_path(images_dir: str | Path, image_name: str) -> Path:
    images_dir = Path(images_dir)
    raw = str(image_name).strip()
    candidate = images_dir / raw
    if candidate.exists():
        return candidate

    stem = Path(raw).stem
    suffix = Path(raw).suffix.lower()

    if not suffix:
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
            p = images_dir / f"{raw}{ext}"
            if p.exists():
                return p

    # Fall back to searching by stem in case metadata and filesystem extensions differ.
    matches = []
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            matches.append(p)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous image name {image_name!r}: {matches}")

    raise FileNotFoundError(f"Could not resolve image {image_name!r} under {images_dir}")


def official_train_test(df: pd.DataFrame):
    cdf = add_canonical_columns(df)
    known = cdf[cdf["_split"].isin(["train", "test"])].copy()
    if known.empty:
        raise ValueError(
            "Could not interpret the dataset Train/split column. "
            "Run the audit script and inspect unique split values before proceeding."
        )
    train_df = known[known["_split"] == "train"].copy()
    test_df = known[known["_split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Expected both train and test rows, got train={len(train_df)}, test={len(test_df)}."
        )
    return train_df, test_df
