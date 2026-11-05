from __future__ import annotations
import json
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from app.config import settings
from app.db import SessionLocal
from app.models import Report, Study
from app.queueing import receive
from app.services import build_study_report_json
from app.storage import upload_file

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=select_autoescape(["html", "xml"]))


def process_report(report_id: str):
    db = SessionLocal()
    try:
        report = db.get(Report, report_id)
        if not report or report.status == "COMPLETE":
            return
        study = db.get(Study, report.study_id)
        report.status = "RUNNING"
        db.commit()

        data = build_study_report_json(db, study)
        report.report_json = data
        template_name = "report.html"
        html = env.get_template(template_name).render(report=report, data=data)
        with tempfile.TemporaryDirectory(prefix="fetal-report-") as td:
            pdf_path = Path(td) / "report.pdf"
            HTML(string=html).write_pdf(pdf_path)
            key = f"{study.hospital_id}/{study.id}/{report.id}/{report.report_type.lower()}-{report.language}.pdf"
            upload_file(settings.s3_reports_bucket, key, str(pdf_path), "application/pdf")
        report.object_key = key
        report.status = "COMPLETE"
        report.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:
        db.rollback()
        report = db.get(Report, report_id)
        if report:
            report.status = "FAILED"
            report.error = f"{type(exc).__name__}: {exc}"
            db.commit()
        traceback.print_exc()
    finally:
        db.close()


def main():
    print("Report worker started")
    while True:
        client, url, messages = receive(settings.report_queue_name)
        for msg in messages:
            body = json.loads(msg["Body"])
            process_report(body["report_id"])
            client.delete_message(QueueUrl=url, ReceiptHandle=msg["ReceiptHandle"])


if __name__ == "__main__":
    main()
