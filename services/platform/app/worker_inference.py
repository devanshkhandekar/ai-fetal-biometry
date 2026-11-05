from __future__ import annotations

import hashlib
import json
import mimetypes
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import (
    InferenceJob,
    InferenceRun,
    Measurement,
    ScanObject,
    Study,
    StudyAggregate,
)
from app.model_adapter import aggregate_study_results, canonical_json_sha256, run_model
from app.queueing import receive
from app.storage import download_file, upload_file


def now():
    return datetime.now(timezone.utc)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_artifacts(result: dict, study: Study, scan: ScanObject, run_id: str) -> dict:
    artifacts = result.get("artifacts") or {}
    normalized = {}
    for name, value in artifacts.items():
        local_path = value.get("path") if isinstance(value, dict) else value
        if not local_path:
            continue
        path = Path(local_path)
        if not path.exists():
            continue
        ext = path.suffix or ".bin"
        key = f"{study.hospital_id}/{study.id}/{scan.id}/{run_id}/{name}{ext}"
        content_type = (
            value.get("content_type") if isinstance(value, dict) else None
        ) or mimetypes.guess_type(str(path))[0]
        upload_file(settings.s3_derived_bucket, key, str(path), content_type)
        normalized[name] = {"object_key": key, "content_type": content_type}
    result["artifacts"] = normalized
    return result


def _latest_run_for_scan(db, scan_id: str) -> InferenceRun | None:
    return db.scalar(
        select(InferenceRun)
        .where(InferenceRun.scan_id == scan_id)
        .order_by(InferenceRun.created_at.desc())
    )


def rebuild_study_aggregate(db, study: Study) -> StudyAggregate:
    aggregate = db.scalar(select(StudyAggregate).where(StudyAggregate.study_id == study.id))
    if aggregate is None:
        aggregate = StudyAggregate(study_id=study.id, status="PENDING")
        db.add(aggregate)
        db.flush()

    scans = db.scalars(
        select(ScanObject).where(ScanObject.study_id == study.id).order_by(ScanObject.created_at)
    ).all()
    by_anatomy: dict[str, list[ScanObject]] = {}
    for scan in scans:
        by_anatomy.setdefault(scan.anatomy, []).append(scan)

    required = ("HEAD", "ABDOMEN", "FEMUR")
    missing = [a for a in required if not by_anatomy.get(a)]
    ambiguous = [a for a in required if len(by_anatomy.get(a, [])) > 1]
    aggregate.updated_at = now()

    if missing:
        aggregate.status = "INCOMPLETE"
        aggregate.error = f"Missing anatomy for aggregation: {', '.join(missing)}"
        aggregate.selected_run_ids = None
        aggregate.result_json = None
        return aggregate
    if ambiguous:
        aggregate.status = "REVIEW"
        aggregate.error = (
            "Multiple scans exist for anatomy/anatomies "
            f"{', '.join(ambiguous)}; explicit run selection is required before EFW."
        )
        aggregate.selected_run_ids = None
        aggregate.result_json = None
        return aggregate

    scan_results: dict[str, dict] = {}
    selected_run_ids: dict[str, str] = {}
    for anatomy in required:
        run = _latest_run_for_scan(db, by_anatomy[anatomy][0].id)
        if run is None:
            aggregate.status = "INCOMPLETE"
            aggregate.error = f"No saved inference run for {anatomy}"
            aggregate.result_json = None
            return aggregate
        result = run.result_json or {}
        status = str((result.get("fusion") or {}).get("status") or "REVIEW").upper()
        if status != "ACCEPT":
            aggregate.status = "REVIEW"
            aggregate.error = f"{anatomy} result is {status}; EFW aggregation requires ACCEPT for all three scans."
            aggregate.selected_run_ids = {**selected_run_ids, anatomy: run.id}
            aggregate.result_json = None
            return aggregate
        scan_results[anatomy] = result
        selected_run_ids[anatomy] = run.id

    aggregate.status = "RUNNING"
    aggregate.error = None
    aggregate.selected_run_ids = selected_run_ids
    db.flush()

    try:
        result = aggregate_study_results(scan_results)
        aggregate.result_json = result
        aggregate.status = "COMPLETE"
        aggregate.error = None
    except Exception as exc:
        aggregate.result_json = None
        aggregate.status = "FAILED"
        aggregate.error = f"{type(exc).__name__}: {exc}"
    aggregate.updated_at = now()
    return aggregate


def process_job(job_id: str):
    db = SessionLocal()
    started = time.perf_counter()
    try:
        job = db.get(InferenceJob, job_id)
        if not job or job.status == "COMPLETE":
            return
        scan = db.get(ScanObject, job.scan_id)
        study = db.get(Study, job.study_id)
        if not scan or not study:
            raise RuntimeError("Inference job references missing scan/study")

        job.status = "RUNNING"
        job.progress_step = "Preparing scan"
        job.started_at = now()
        scan.status = "PROCESSING"
        db.commit()

        suffix = Path(scan.original_filename).suffix or ".bin"
        with tempfile.TemporaryDirectory(prefix="fetal-infer-") as td:
            input_path = str(Path(td) / f"input{suffix}")
            download_file(settings.s3_originals_bucket, scan.object_key, input_path)
            input_sha256 = sha256_file(input_path)

            job.progress_step = "Running biometry"
            db.commit()

            result = run_model(
                input_path,
                scan.anatomy,
                scan.pixel_spacing_x,
                scan.pixel_spacing_y,
                scan.roi,
                workdir=td,
            )

            run = InferenceRun(
                job_id=job.id,
                scan_id=scan.id,
                model_release=settings.pipeline_model_release,
                result_json={},
                device=result.get("device"),
            )
            db.add(run)
            db.flush()

            result = upload_artifacts(result, study, scan, run.id)
            runtime_seconds = time.perf_counter() - started
            provenance = result.setdefault("provenance", {})
            if not isinstance(provenance, dict):
                provenance = {"pipeline": provenance}
                result["provenance"] = provenance
            provenance.setdefault("release", settings.pipeline_model_release)
            provenance["input_sha256"] = input_sha256
            provenance["run_id"] = run.id
            provenance["runtime_seconds"] = runtime_seconds
            provenance["completed_at"] = now().isoformat()
            provenance["output_sha256"] = canonical_json_sha256(result)

            run.result_json = result
            run.device = result.get("device") or provenance.get("device")
            run.runtime_seconds = runtime_seconds

            for name, m in (result.get("measurements") or {}).items():
                if isinstance(m, (int, float)):
                    value, unit, confidence, status = float(m), "", None, None
                else:
                    if m.get("value") is None:
                        continue
                    value = float(m["value"])
                    unit = m.get("unit", "")
                    confidence = m.get("confidence")
                    status = m.get("status")
                db.add(
                    Measurement(
                        run_id=run.id,
                        name=name,
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        status=status,
                    )
                )

            job.status = "COMPLETE"
            job.progress_step = "Complete"
            job.completed_at = now()
            scan.status = "COMPLETE"
            db.flush()

            all_jobs = db.scalars(
                select(InferenceJob).where(InferenceJob.study_id == study.id)
            ).all()
            if all(j.status == "COMPLETE" for j in all_jobs):
                if settings.enable_study_aggregation:
                    aggregate = rebuild_study_aggregate(db, study)
                    study.status = "COMPLETE" if aggregate.status == "COMPLETE" else "REVIEW"
                else:
                    study.status = "COMPLETE"
            elif all(j.status in ("COMPLETE", "FAILED") for j in all_jobs):
                study.status = "REVIEW"

            db.commit()
    except Exception as exc:
        db.rollback()
        try:
            job = db.get(InferenceJob, job_id)
            if job:
                job.status = "FAILED"
                job.progress_step = "Failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.completed_at = now()
                scan = db.get(ScanObject, job.scan_id)
                if scan:
                    scan.status = "FAILED"
                study = db.get(Study, job.study_id)
                if study:
                    study.status = "REVIEW"
                db.commit()
        finally:
            traceback.print_exc()
    finally:
        db.close()


def main():
    print(f"Inference worker started. PIPELINE_MODE={settings.pipeline_mode}", flush=True)
    while True:
        client, url, messages = receive(settings.inference_queue_name)
        for msg in messages:
            body = json.loads(msg["Body"])
            process_job(body["job_id"])
            client.delete_message(QueueUrl=url, ReceiptHandle=msg["ReceiptHandle"])


if __name__ == "__main__":
    main()
