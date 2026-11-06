from __future__ import annotations

import base64
import json
import mimetypes
import os

from pathlib import Path


DEFAULT_BRANDING = {
    "hospital_name":
        "Fetal Biometry Platform",

    "address":
        "",

    "report_title":
        "Fetal Biometry Report",

    "primary_color":
        "#16324F",

    "secondary_color":
        "#2563EB",

    "doctor_name":
        "",

    "watermark":
        False,

    "watermark_text":
        "CONFIDENTIAL",

    "disclaimer":
        (
            "AI-assisted measurements "
            "must be reviewed by a "
            "qualified clinician."
        ),

    "logo_path":
        "",
}


def _image_data_uri(
    path_value,
):
    if not path_value:
        return None

    path = Path(
        path_value
    )

    if not path.exists():
        return None

    mime = (
        mimetypes.guess_type(
            str(path)
        )[0]
        or
        "image/png"
    )

    encoded = (
        base64
        .b64encode(
            path.read_bytes()
        )
        .decode(
            "ascii"
        )
    )

    return (
        f"data:{mime};"
        f"base64,{encoded}"
    )


def load_branding():

    config_path = Path(
        os.getenv(
            "HOSPITAL_BRANDING_FILE",
            "/branding/branding.json",
        )
    )

    branding = dict(
        DEFAULT_BRANDING
    )

    if config_path.exists():

        data = json.loads(
            config_path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            data,
            dict,
        ):
            branding.update(
                data
            )

    branding[
        "logo_data_uri"
    ] = _image_data_uri(
        branding.get(
            "logo_path"
        )
    )

    return branding
