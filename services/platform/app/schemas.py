from typing import Any, Literal
from pydantic import BaseModel, Field

Anatomy = Literal["HEAD", "ABDOMEN", "FEMUR"]


class StudyCreate(BaseModel):
    patient_external_id: str = Field(min_length=1, max_length=120)
    patient_name: str | None = Field(default=None, max_length=200)
    date_of_birth: str | None = None
    accession_number: str | None = Field(default=None, max_length=120)


class UploadInitiate(BaseModel):
    anatomy: Anatomy
    filename: str
    content_type: str = "application/octet-stream"
    pixel_spacing_x: float | None = None
    pixel_spacing_y: float | None = None
    roi: dict[str, Any] | None = None


class UploadComplete(BaseModel):
    scan_id: str


class ReportCreate(BaseModel):
    report_type: Literal["DOCTOR", "PATIENT", "TECHNICAL"] = "DOCTOR"
    language: str = "en"


class ExplainRequest(BaseModel):
    language: str = "en"
