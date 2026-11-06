from __future__ import annotations

import base64
import json
import mimetypes
import tempfile
import traceback
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from weasyprint import HTML

from app.config import settings
from app.db import SessionLocal
from app.models import InferenceRun, Report, ScanObject, Study
from app.queueing import receive
from app.services import build_study_report_json
from app.storage import download_file, upload_file
from app.branding import load_branding

from app.report_i18n import (
    get_report_labels,
)

TEMPLATE_DIR = (
    Path(__file__).resolve().parent.parent
    / "templates"
)

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(
        ["html", "xml"]
    ),
)


# ============================================================
# JSON normalization
# ============================================================

def _json_safe(value: Any) -> Any:
    """
    Recursively convert report data into standard JSON-compatible
    Python primitives.

    This is intentionally applied only at the report boundary.

    Important:
    - datetime/date -> ISO-8601 string
    - UUID -> string
    - Decimal -> float
    - Path -> string
    - Enum -> its value
    - dict/list structures are recursively normalized
    - measurement floats/ints remain numeric
    """

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Enum):
        return _json_safe(
            value.value
        )

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [
            _json_safe(item)
            for item in value
        ]

    # Pydantic v2 objects
    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(model_dump):
        return _json_safe(
            model_dump()
        )

    # Handle NumPy scalar-like objects without importing NumPy.
    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(item_method):
        try:
            return _json_safe(
                item_method()
            )
        except Exception:
            pass

    raise TypeError(
        "Unsupported report JSON value: "
        f"{type(value).__module__}."
        f"{type(value).__name__}"
    )


def _validate_json_payload(
    payload: Any,
) -> None:
    """
    Fail before DB commit/PDF generation if anything non-JSON-safe
    remains.
    """

    json.dumps(
        payload,
        ensure_ascii=False,
    )


# ============================================================
# Report images
# ============================================================

def _as_data_uri(
    bucket: str,
    key: str,
    content_type: str | None,
    workdir: Path,
    name: str,
) -> str | None:
    """
    Download a stored image into the temporary report workdir and
    return an inline data URI for the HTML/PDF template.

    Missing/non-image artifacts are simply omitted from the report.
    """

    try:
        suffix = (
            Path(key).suffix
            or ".bin"
        )

        path = (
            workdir
            / f"{name}{suffix}"
        )

        download_file(
            bucket,
            key,
            str(path),
        )

        mime = (
            content_type
            or mimetypes.guess_type(
                str(path)
            )[0]
            or "application/octet-stream"
        )

        if not mime.startswith(
            "image/"
        ):
            return None

        encoded = (
            base64
            .b64encode(
                path.read_bytes()
            )
            .decode("ascii")
        )

        return (
            f"data:{mime};"
            f"base64,{encoded}"
        )

    except Exception:
        # Report generation should still work even if an optional
        # image artifact is unavailable.
        return None


def _attach_report_images(
    db,
    study: Study,
    data: dict,
    workdir: Path,
) -> None:
    """
    Attach original/overlay/mask/preview images to each scan entry
    for rendering in the Doctor/Technical report.
    """

    for scan_data in data.get(
        "scans",
        [],
    ):
        scan_id = scan_data.get(
            "id"
        )

        if not scan_id:
            continue

        scan = db.get(
            ScanObject,
            scan_id,
        )

        if scan is None:
            continue

        images: dict[
            str,
            str,
        ] = {}

        # --------------------------------------------
        # Original uploaded scan
        # --------------------------------------------

        original = _as_data_uri(
            settings.s3_originals_bucket,
            scan.object_key,
            scan.content_type,
            workdir,
            f"{scan.id}-original",
        )

        if original:
            images[
                "original"
            ] = original

        # --------------------------------------------
        # Latest inference artifacts
        # --------------------------------------------

        run = db.scalar(
            select(
                InferenceRun
            )
            .where(
                InferenceRun.scan_id
                == scan.id
            )
            .order_by(
                InferenceRun
                .created_at
                .desc()
            )
        )

        result = (
            run.result_json
            or {}
        ) if run else {}

        artifacts = (
            result.get(
                "artifacts"
            )
            or {}
        )

        for name in (
            "overlay",
            "mask",
            "preview",
        ):
            info = artifacts.get(
                name
            )

            if not isinstance(
                info,
                dict,
            ):
                continue

            object_key = info.get(
                "object_key"
            )

            if not object_key:
                continue

            uri = _as_data_uri(
                settings.s3_derived_bucket,
                object_key,
                info.get(
                    "content_type"
                ),
                workdir,
                f"{scan.id}-{name}",
            )

            if uri:
                images[
                    name
                ] = uri

        scan_data[
            "report_images"
        ] = images


# ============================================================
# Report processing
# ============================================================

def process_report(
    report_id: str,
) -> None:

    db = SessionLocal()

    try:
        report = db.get(
            Report,
            report_id,
        )

        if report is None:
            print(
                "Report not found:",
                report_id,
            )
            return

        if (
            report.status
            == "COMPLETE"
        ):
            return

        study = db.get(
            Study,
            report.study_id,
        )

        if study is None:
            raise RuntimeError(
                "Study not found for "
                f"report {report_id}"
            )

        # --------------------------------------------
        # Mark running
        # --------------------------------------------

        report.status = "RUNNING"
        report.error = None

        db.commit()

        print(
            "Generating report:",
            report.id,
            "type=",
            report.report_type,
            "study=",
            study.id,
        )

        # --------------------------------------------
        # Build canonical report JSON
        # --------------------------------------------

        data = (
            build_study_report_json(
                db,
                study,
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "build_study_report_json "
                "must return dict"
            )

        # --------------------------------------------
        # Temporary PDF rendering workspace
        # --------------------------------------------

        with tempfile.TemporaryDirectory(
            prefix="fetal-report-"
        ) as td:

            td_path = Path(td)

            # Add original/overlay/mask/preview images.
            _attach_report_images(
                db,
                study,
                data,
                td_path,
            )

            # ==================================================
            # CRITICAL FIX
            #
            # SQLAlchemy model attributes such as created_at,
            # completed_at, acquisition timestamps, etc. may be
            # real Python datetime objects.
            #
            # Normalize them BEFORE:
            #   1. PostgreSQL JSON/JSONB storage
            #   2. Jinja rendering / tojson filters
            #   3. any Python json.dumps call
            #
            # Measurement floats remain floats.
            # ==================================================

            safe_data = _json_safe(
                data
            )

            # Explicit validation so serialization problems are
            # caught here with a clear traceback.
            _validate_json_payload(
                safe_data
            )

            # Save exactly the JSON-safe representation that the
            # PDF is generated from.
            report.report_json = (
                safe_data
            )

            # Force SQLAlchemy to serialize report_json now,
            # before the relatively expensive PDF rendering.
            db.flush()

            # --------------------------------------------
            # Render HTML
            # --------------------------------------------

            template = (
                env.get_template(
                    "report.html"
                )
            )

            branding = load_branding()
            labels = get_report_labels(report.language)
            html = template.render(report=report,data=safe_data,branding=branding,labels=labels,)

            # --------------------------------------------
            # Generate PDF
            # --------------------------------------------

            pdf_path = (
                td_path
                / "report.pdf"
            )

            HTML(
                string=html
            ).write_pdf(
                str(pdf_path)
            )

            if (
                not pdf_path.exists()
                or
                pdf_path.stat().st_size
                <= 0
            ):
                raise RuntimeError(
                    "Generated PDF is "
                    "missing or empty"
                )

            # --------------------------------------------
            # Upload PDF
            # --------------------------------------------

            key = (
                f"{study.hospital_id}/"
                f"{study.id}/"
                f"{report.id}/"
                f"{report.report_type.lower()}"
                f"-{report.language}.pdf"
            )

            upload_file(
                settings.s3_reports_bucket,
                key,
                str(pdf_path),
                "application/pdf",
            )

        # --------------------------------------------
        # Complete report
        # --------------------------------------------

        report.object_key = key
        report.status = "COMPLETE"
        report.error = None

        report.completed_at = (
            datetime.now(
                timezone.utc
            )
        )

        db.commit()

        print(
            "Report complete:",
            report.id,
            key,
        )

    except Exception as exc:

        db.rollback()

        print(
            "Report generation failed:",
            report_id,
            f"{type(exc).__name__}: "
            f"{exc}",
        )

        traceback.print_exc()

        # Store the failure so React can surface it.
        try:
            report = db.get(
                Report,
                report_id,
            )

            if report:
                report.status = (
                    "FAILED"
                )

                report.error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                db.commit()

        except Exception:
            db.rollback()
            traceback.print_exc()

    finally:
        db.close()


# ============================================================
# Worker loop
# ============================================================

def main() -> None:

    print(
        "Report worker started",
        flush=True,
    )

    while True:

        client, url, messages = (
            receive(
                settings.report_queue_name
            )
        )

        for msg in messages:

            try:
                body = json.loads(
                    msg["Body"]
                )

                report_id = body.get(
                    "report_id"
                )

                if not report_id:
                    raise ValueError(
                        "report queue message "
                        "missing report_id"
                    )

                process_report(
                    report_id
                )

            except Exception:
                traceback.print_exc()

            finally:
                # Preserve current queue behavior:
                # a processed report message is removed.
                client.delete_message(
                    QueueUrl=url,
                    ReceiptHandle=(
                        msg[
                            "ReceiptHandle"
                        ]
                    ),
                )


if __name__ == "__main__":
    main()
