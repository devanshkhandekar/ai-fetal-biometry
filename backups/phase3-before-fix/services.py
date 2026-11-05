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


def serialize_scan_result(db: Session, scan: ScanObject) -> dict:
    run = db.scalar(
        select(InferenceRun)
        .where(InferenceRun.scan_id == scan.id)
        .order_by(InferenceRun.created_at.desc())
    )
    result = run.result_json if run else None
    if result and result.get("artifacts"):
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
        "warnings": [
            w
            for s in scan_results
            if s.get("result")
            for w in s["result"].get("warnings", [])
        ]
        + aggregate_warnings,
    }
