from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import String, DateTime, Float, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


def uid() -> str:
    return str(uuid4())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Hospital(Base):
    __tablename__ = "hospitals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    oidc_sub: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class UserHospitalRole(Base):
    __tablename__ = "user_hospital_roles"
    __table_args__ = (UniqueConstraint("user_id", "hospital_id", name="uq_user_hospital"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    hospital_id: Mapped[str] = mapped_column(ForeignKey("hospitals.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="DOCTOR")


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    hospital_id: Mapped[str] = mapped_column(ForeignKey("hospitals.id"), index=True, nullable=False)
    external_id: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(200))
    date_of_birth: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class Study(Base):
    __tablename__ = "studies"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    hospital_id: Mapped[str] = mapped_column(ForeignKey("hospitals.id"), index=True, nullable=False)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True, nullable=False)
    accession_number: Mapped[str | None] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(40), default="CREATED", index=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)

    patient: Mapped[Patient] = relationship()


class ScanObject(Base):
    __tablename__ = "scan_objects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"), index=True, nullable=False)
    anatomy: Mapped[str] = mapped_column(String(30), index=True, nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(300), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="PENDING_UPLOAD", index=True)
    pixel_spacing_x: Mapped[float | None] = mapped_column(Float)
    pixel_spacing_y: Mapped[float | None] = mapped_column(Float)
    roi: Mapped[dict | None] = mapped_column(JSONB)
    dicom_metadata: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class InferenceJob(Base):
    __tablename__ = "inference_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"), index=True, nullable=False)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scan_objects.id", ondelete="CASCADE"), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="QUEUED", index=True)
    progress_step: Mapped[str] = mapped_column(String(80), default="Queued")
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class InferenceRun(Base):
    __tablename__ = "inference_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(ForeignKey("inference_jobs.id"), unique=True, nullable=False)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scan_objects.id"), index=True, nullable=False)
    model_release: Mapped[str] = mapped_column(String(120), nullable=False)
    result_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    device: Mapped[str | None] = mapped_column(String(100))
    runtime_seconds: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class Measurement(Base):
    __tablename__ = "measurements"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    run_id: Mapped[str] = mapped_column(ForeignKey("inference_runs.id", ondelete="CASCADE"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(60), index=True, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(30), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str | None] = mapped_column(String(30))


class StudyAggregate(Base):
    __tablename__ = "study_aggregates"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    study_id: Mapped[str] = mapped_column(
        ForeignKey("studies.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(40), default="PENDING", index=True)
    selected_run_ids: Mapped[dict | None] = mapped_column(JSONB)
    result_json: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    study_id: Mapped[str] = mapped_column(ForeignKey("studies.id", ondelete="CASCADE"), index=True, nullable=False)
    report_type: Mapped[str] = mapped_column(String(40), nullable=False)
    language: Mapped[str] = mapped_column(String(20), default="en")
    status: Mapped[str] = mapped_column(String(40), default="QUEUED", index=True)
    object_key: Mapped[str | None] = mapped_column(String(1024))
    report_json: Mapped[dict | None] = mapped_column(JSONB)
    narrative: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uid)
    hospital_id: Mapped[str] = mapped_column(ForeignKey("hospitals.id"), index=True, nullable=False)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(120), index=True, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(80), nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc)
