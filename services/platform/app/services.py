from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import AuditEvent, InferenceRun, ScanObject, Study, StudyAggregate
from app.storage import presign_get


def audit(
    db: Session,
    hospital_id: str,
    user_id: str | None,
    action: str,
    entity_type: str,
    entity_id: str,
    metadata: dict | None = None,
):
    db.add(
        AuditEvent(
            hospital_id=hospital_id,
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            metadata_json=metadata,
        )
    )


def _humanize_warnings(values) -> list[str]:
    """Collapse duplicate internal reason codes into clinician-facing messages."""
    raw = [str(x).strip() for x in (values or []) if str(x).strip()]
    lower = [x.lower() for x in raw]
    out: list[str] = []

    if any("image quality=poor" in x or "image_quality:poor" in x for x in lower):
        out.append("Image quality was classified as poor; automatic acceptance is not allowed.")
    elif any("image quality=review" in x or "image_quality:review" in x for x in lower):
        out.append("Image quality requires review before automatic acceptance.")

    if any("full_image_roi_fallback_unvalidated" in x for x in lower):
        out.append(
            "Full-image ROI fallback was used; provide/confirm a trusted ROI for automatic acceptance."
        )
    elif any("measurement_review" in x or "measurement:review" in x for x in lower):
        out.append("The measurement requires review before automatic acceptance.")

    known = (
        "image quality=poor",
        "image quality=review",
        "image_quality:poor",
        "image_quality:review",
        "full_image_roi_fallback_unvalidated",
        "measurement_review",
        "measurement:review",
    )
    for item in raw:
        if any(k in item.lower() for k in known):
            continue
        if item not in out:
            out.append(item)
    return out


def serialize_scan_result(db: Session, scan: ScanObject) -> dict:
    run = db.scalar(
        select(InferenceRun)
        .where(InferenceRun.scan_id == scan.id)
        .order_by(InferenceRun.created_at.desc())
    )
    stored_result = run.result_json if run else None
    result = dict(stored_result) if isinstance(stored_result, dict) else stored_result

    if result:
        raw_warnings = list(result.get("warnings") or [])
        result["warning_codes"] = raw_warnings
        result["warnings"] = _humanize_warnings(raw_warnings)

        if result.get("artifacts"):
            urls = {}
            for name, info in result["artifacts"].items():
                if isinstance(info, dict) and info.get("object_key"):
                    urls[name] = presign_get(settings.s3_derived_bucket, info["object_key"])
            result = {**result, "artifact_urls": urls}

    return {
        "id": scan.id,
        "anatomy": scan.anatomy,
        "filename": scan.original_filename,
        "status": scan.status,
        "pixel_spacing_x": scan.pixel_spacing_x,
        "pixel_spacing_y": scan.pixel_spacing_y,
        "roi": scan.roi,
        "original_url": presign_get(settings.s3_originals_bucket, scan.object_key),
        "result": result,
    }


def build_study_report_json(db: Session, study: Study) -> dict:
    scans = db.scalars(select(ScanObject).where(ScanObject.study_id == study.id)).all()
    scan_results = [serialize_scan_result(db, s) for s in scans]

    measurements: dict = {}
    for s in scan_results:
        if s.get("result"):
            measurements.update(s["result"].get("measurements", {}))

    aggregate = db.scalar(select(StudyAggregate).where(StudyAggregate.study_id == study.id))
    aggregate_payload = None
    aggregate_warnings: list[str] = []
    if aggregate:
        aggregate_payload = {
            "id": aggregate.id,
            "status": aggregate.status,
            "selected_run_ids": aggregate.selected_run_ids,
            "result": aggregate.result_json,
            "error": aggregate.error,
            "updated_at": aggregate.updated_at,
        }
        if aggregate.status == "COMPLETE" and aggregate.result_json:
            measurements.update(aggregate.result_json.get("measurements", {}))
            aggregate_warnings.extend(aggregate.result_json.get("warnings", []) or [])
        elif aggregate.error:
            aggregate_warnings.append(f"Study aggregation: {aggregate.error}")

    warnings = [
        w
        for s in scan_results
        if s.get("result")
        for w in s["result"].get("warnings", [])
    ] + _humanize_warnings(aggregate_warnings)
    warnings = list(dict.fromkeys(warnings))

    return {
        "study_id": study.id,
        "patient": {
            "external_id": study.patient.external_id,
            "display_name": study.patient.display_name,
            "date_of_birth": study.patient.date_of_birth,
        },
        "measurements": measurements,
        "study_aggregate": aggregate_payload,
        "scans": scan_results,
        "warnings": warnings,
    }
