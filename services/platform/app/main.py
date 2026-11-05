from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session
from openai import OpenAI

from app.auth import CurrentUser, get_current_user
from app.config import settings
from app.db import Base, engine, get_db
from app.models import Hospital, Patient, Study, ScanObject, InferenceJob, InferenceRun, Report
from app.queueing import send
from app.schemas import StudyCreate, UploadInitiate, UploadComplete, ReportCreate, ExplainRequest
from app.services import audit, serialize_scan_result, build_study_report_json
from app.storage import presign_put, head_object, presign_get

app = FastAPI(title="Fetal Biometry Platform API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # Local convenience only. Keep Alembic as the production migration mechanism.
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return user.__dict__


@app.post("/v1/studies")
def create_study(payload: StudyCreate, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    patient = db.scalar(select(Patient).where(Patient.hospital_id == user.hospital_id, Patient.external_id == payload.patient_external_id))
    if not patient:
        patient = Patient(hospital_id=user.hospital_id, external_id=payload.patient_external_id, display_name=payload.patient_name, date_of_birth=payload.date_of_birth)
        db.add(patient)
        db.flush()
    study = Study(hospital_id=user.hospital_id, patient_id=patient.id, accession_number=payload.accession_number, created_by=user.id)
    db.add(study)
    db.flush()
    audit(db, user.hospital_id, user.id, "study.created", "study", study.id)
    db.commit()
    return {"id": study.id, "status": study.status}


@app.get("/v1/studies")
def list_studies(db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    studies = db.scalars(select(Study).where(Study.hospital_id == user.hospital_id).order_by(Study.created_at.desc())).all()
    return [
        {"id": s.id, "patient_external_id": s.patient.external_id, "patient_name": s.patient.display_name, "accession_number": s.accession_number, "status": s.status, "created_at": s.created_at}
        for s in studies
    ]


@app.get("/v1/studies/{study_id}")
def get_study(study_id: str, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    if not study or study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Study not found")
    scans = db.scalars(select(ScanObject).where(ScanObject.study_id == study.id).order_by(ScanObject.created_at)).all()
    jobs = db.scalars(select(InferenceJob).where(InferenceJob.study_id == study.id).order_by(InferenceJob.created_at)).all()
    return {
        "id": study.id,
        "patient": {"external_id": study.patient.external_id, "display_name": study.patient.display_name, "date_of_birth": study.patient.date_of_birth},
        "accession_number": study.accession_number,
        "status": study.status,
        "scans": [serialize_scan_result(db, s) for s in scans],
        "jobs": [{"id": j.id, "scan_id": j.scan_id, "status": j.status, "progress_step": j.progress_step, "error": j.error} for j in jobs],
    }


@app.post("/v1/studies/{study_id}/uploads/initiate")
def initiate_upload(study_id: str, payload: UploadInitiate, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    if not study or study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Study not found")
    scan_id = str(uuid4())
    safe_name = Path(payload.filename).name.replace(" ", "_")
    key = f"{user.hospital_id}/{study.id}/{scan_id}/{safe_name}"
    scan = ScanObject(
        id=scan_id, study_id=study.id, anatomy=payload.anatomy, object_key=key,
        original_filename=payload.filename, content_type=payload.content_type,
        pixel_spacing_x=payload.pixel_spacing_x, pixel_spacing_y=payload.pixel_spacing_y,
        roi=payload.roi,
    )
    db.add(scan)
    audit(db, user.hospital_id, user.id, "upload.initiated", "scan", scan.id, {"anatomy": payload.anatomy})
    db.commit()
    return {
        "scan_id": scan.id,
        "object_key": key,
        "upload_url": presign_put(settings.s3_originals_bucket, key, payload.content_type),
        "required_headers": {"Content-Type": payload.content_type},
    }


@app.post("/v1/studies/{study_id}/uploads/complete")
def complete_upload(study_id: str, payload: UploadComplete, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    scan = db.get(ScanObject, payload.scan_id)
    if not study or study.hospital_id != user.hospital_id or not scan or scan.study_id != study.id:
        raise HTTPException(404, "Study/scan not found")
    try:
        meta = head_object(settings.s3_originals_bucket, scan.object_key)
    except Exception as exc:
        raise HTTPException(400, f"Object not found in storage: {exc}") from exc
    scan.status = "UPLOADED"
    study.status = "READY_FOR_INFERENCE"
    audit(db, user.hospital_id, user.id, "upload.completed", "scan", scan.id, {"size": meta.get("ContentLength")})
    db.commit()
    return {"scan_id": scan.id, "status": scan.status, "size": meta.get("ContentLength")}


@app.post("/v1/studies/{study_id}/inference")
def start_inference(study_id: str, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    if not study or study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Study not found")
    scans = db.scalars(select(ScanObject).where(ScanObject.study_id == study.id, ScanObject.status.in_(["UPLOADED", "FAILED"]))).all()
    if not scans:
        raise HTTPException(400, "No uploaded scans are ready for inference")
    jobs = []
    for scan in scans:
        job = InferenceJob(study_id=study.id, scan_id=scan.id, status="QUEUED", progress_step="Queued")
        db.add(job)
        db.flush()
        scan.status = "QUEUED"
        jobs.append(job)
    study.status = "PROCESSING"
    audit(db, user.hospital_id, user.id, "inference.queued", "study", study.id, {"job_count": len(jobs)})
    db.commit()
    for job in jobs:
        send(settings.inference_queue_name, {"job_id": job.id})
    return {"jobs": [{"id": j.id, "scan_id": j.scan_id, "status": j.status} for j in jobs]}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    job = db.get(InferenceJob, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    study = db.get(Study, job.study_id)
    if study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Job not found")
    return {"id": job.id, "status": job.status, "progress_step": job.progress_step, "error": job.error}


@app.get("/v1/studies/{study_id}/results")
def study_results(study_id: str, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    if not study or study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Study not found")
    return build_study_report_json(db, study)


@app.post("/v1/studies/{study_id}/reports")
def create_report(study_id: str, payload: ReportCreate, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    if not study or study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Study not found")
    report = Report(study_id=study.id, report_type=payload.report_type, language=payload.language, status="QUEUED")
    db.add(report)
    db.flush()
    audit(db, user.hospital_id, user.id, "report.queued", "report", report.id, {"type": payload.report_type})
    db.commit()
    send(settings.report_queue_name, {"report_id": report.id})
    return {"id": report.id, "status": report.status}


@app.get("/v1/studies/{study_id}/reports")
def list_reports(study_id: str, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    study = db.get(Study, study_id)
    if not study or study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Study not found")
    reports = db.scalars(select(Report).where(Report.study_id == study.id).order_by(Report.created_at.desc())).all()
    return [{"id": r.id, "type": r.report_type, "language": r.language, "status": r.status, "error": r.error, "download_url": presign_get(settings.s3_reports_bucket, r.object_key) if r.object_key else None, "narrative": r.narrative} for r in reports]


@app.post("/v1/reports/{report_id}/explain")
def explain_report(report_id: str, payload: ExplainRequest, db: Session = Depends(get_db), user: CurrentUser = Depends(get_current_user)):
    report = db.get(Report, report_id)
    if not report:
        raise HTTPException(404, "Report not found")
    study = db.get(Study, report.study_id)
    if study.hospital_id != user.hospital_id:
        raise HTTPException(404, "Report not found")
    if not report.report_json:
        raise HTTPException(409, "Report is not complete")
    if not settings.openai_api_key or not settings.openai_model:
        raise HTTPException(503, "OPENAI_API_KEY/OPENAI_MODEL are not configured")

    # Deliberately exclude patient identifiers and image pixels.
    safe_payload = {
        "measurements": report.report_json.get("measurements", {}),
        "warnings": report.report_json.get("warnings", []),
        "scan_summaries": [
            {
                "anatomy": s.get("anatomy"),
                "result": {
                    "plane": (s.get("result") or {}).get("plane"),
                    "quality": (s.get("result") or {}).get("quality"),
                    "fusion": (s.get("result") or {}).get("fusion"),
                },
            }
            for s in report.report_json.get("scans", [])
        ],
    }
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.openai_model,
        input=(
            f"Explain this fetal biometry report in {payload.language}. "
            "Do not change or recalculate any measurement. State that AI output is supportive and requires clinician review. "
            f"Structured result: {safe_payload}"
        ),
    )
    report.narrative = response.output_text
    audit(db, user.hospital_id, user.id, "report.explained", "report", report.id, {"language": payload.language})
    db.commit()
    return {"narrative": report.narrative}
