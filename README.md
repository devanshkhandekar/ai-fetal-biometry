# Local Production-Shaped AI Fetal Biometry Platform

This repository is a runnable local scaffold for the production stack decided in the roadmap:

- React + TypeScript + Vite + Tailwind
- TanStack Query + React Hook Form + Zod
- FastAPI + Pydantic
- PostgreSQL + SQLAlchemy + Alembic
- MinIO as local S3-compatible object storage
- ElasticMQ as local SQS-compatible queues
- Keycloak as local OIDC identity provider (Cognito replacement for local development)
- Separate inference worker
- Jinja2 + WeasyPrint report worker
- Optional OpenAI narrative endpoint

## Important model-integration boundary

The full web/backend/storage/queue path is concrete. The uploaded roadmap does not contain the literal import path/signature of your current Phase 0-7 function, so one seam is intentionally configurable:

```env
PIPELINE_MODE=python
PIPELINE_MODULE=pipeline_devansh
PIPELINE_FUNCTION=run_inference
# Optional: point these to your existing validated HC+AC+FL -> EFW/study aggregator
STUDY_AGGREGATOR_MODULE=
STUDY_AGGREGATOR_FUNCTION=
MODEL_REPO_PATH=/absolute/path/to/your/existing/model/repo
```

The worker calls the function as:

```python
run_inference(
    input_path="/tmp/input.dcm",
    anatomy="HEAD",            # HEAD | ABDOMEN | FEMUR
    pixel_spacing=(x, y),       # or None
    roi={...},                  # or None
)
```

Return this canonical structure (or update only `app/model_adapter.py::normalize_result`):

```json
{
  "plane": {"label": "HEAD", "confidence": 0.97, "advisory": true},
  "quality": {"label": "GOOD", "confidence": 0.91, "advisory": true},
  "route": "HEAD",
  "fusion": {"status": "ACCEPT", "confidence": 0.90, "reasons": []},
  "measurements": {
    "hc_mm": {"value": 320.2, "unit": "mm", "confidence": 0.94, "status": "ACCEPT"}
  },
  "artifacts": {
    "overlay": {"path": "/tmp/head_overlay.png"},
    "mask": {"path": "/tmp/head_mask.png"}
  },
  "dicom_metadata": {},
  "warnings": [],
  "device": "cuda:0"
}
```

A flat result containing `hc_mm`, `bpd_mm`, `ofd_mm`, `ac_mm`, `fl_mm`, etc. is also partially normalized automatically.

---

## 0. Prerequisites on Ubuntu

Install:

- Docker Engine + Docker Compose plugin
- Node is not needed if you run the web service in Docker
- For the real model worker: keep your existing working Conda/venv exactly as-is
- NVIDIA driver/CUDA remain whatever already runs your current pipeline

Check:

```bash
docker --version
docker compose version
nvidia-smi
```

## 1. Configure

```bash
cp .env.example .env
```

For the first infrastructure smoke test leave:

```env
PIPELINE_MODE=mock
```

## 2. Start the full local stack with the mock inference worker

```bash
docker compose --profile mock-worker up --build
```

Open:

- Web: http://localhost:5173
- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001
- Keycloak admin: http://localhost:8081

Demo login:

```text
doctor@local.test
password: doctor123
```

Keycloak admin:

```text
admin
password: admin123
```

The mock worker is only for proving React -> FastAPI -> PostgreSQL -> MinIO -> Queue -> Worker -> Report Worker. It is not a model result.

## 3. Validate persistence

Create a study, upload scans, start analysis, and generate a PDF. Then restart:

```bash
docker compose down
docker compose --profile mock-worker up
```

The studies/scans/results/reports persist because PostgreSQL and MinIO use named Docker volumes.

To intentionally erase local data:

```bash
docker compose down -v
```

## 4. Switch from mock worker to YOUR REAL existing models

Stop only the mock inference worker:

```bash
docker compose stop inference-worker-mock
```

Edit `.env`:

```env
PIPELINE_MODE=python
PIPELINE_MODULE=pipeline_devansh
PIPELINE_FUNCTION=run_inference
# Optional: point these to your existing validated HC+AC+FL -> EFW/study aggregator
STUDY_AGGREGATOR_MODULE=
STUDY_AGGREGATOR_FUNCTION=
MODEL_REPO_PATH=/absolute/path/to/your/current/fetal/model/repo
PIPELINE_MODEL_RELEASE=phase0-7-local-v1
```

Then activate the environment that ALREADY runs your current head/abdomen/femur pipeline:

```bash
conda activate <your-existing-working-env>
```

Make sure the platform dependencies are present in that env (these do not replace your ML dependencies):

```bash
pip install -r services/platform/requirements.txt
```

Run the worker on the host:

```bash
./scripts/run-real-worker-host.sh
```

This process consumes the SAME ElasticMQ queue, downloads from the SAME MinIO bucket, writes to the SAME PostgreSQL DB, and uploads overlays to the SAME derived bucket. Only model execution runs in your proven ML environment.

### If your real function signature differs

Do NOT modify the API or worker. Edit only:

```text
services/platform/app/model_adapter.py
```

Specifically `_call_existing_pipeline()` and, if required, `normalize_result()`.

### Optional existing study-level deterministic function

If EFW is currently calculated outside the per-scan function, expose your existing validated function and configure:

```env
STUDY_AGGREGATOR_MODULE=<your_module>
STUDY_AGGREGATOR_FUNCTION=<your_function>
```

Expected call:

```python
fn(measurements={"hc_mm": 320.2, "ac_mm": 301.5, "fl_mm": 60.3})
```

The platform intentionally does **not** invent or replace your validated Hadlock implementation.

## 5. Main REST flow

For Swagger/curl testing, obtain a local bearer token:

```bash
TOKEN=$(./scripts/get-local-token.sh)
```

Then use `Authorization: Bearer $TOKEN`, or click **Authorize** in http://localhost:8000/docs.


### Create study

`POST /v1/studies`

```json
{
  "patient_external_id": "DEMO-001",
  "patient_name": "Demo Patient",
  "accession_number": "ACC-001"
}
```

### Initiate direct object upload

`POST /v1/studies/{study_id}/uploads/initiate`

```json
{
  "anatomy": "HEAD",
  "filename": "head.dcm",
  "content_type": "application/dicom",
  "pixel_spacing_x": 0.15,
  "pixel_spacing_y": 0.15,
  "roi": null
}
```

Response contains a presigned MinIO/S3 PUT URL. The browser PUTs directly to object storage.

### Confirm upload

`POST /v1/studies/{study_id}/uploads/complete`

```json
{"scan_id":"..."}
```

### Queue all uploaded scans for inference

`POST /v1/studies/{study_id}/inference`

No body.

### Poll

`GET /v1/jobs/{job_id}`

or simply:

`GET /v1/studies/{study_id}`

### Result

`GET /v1/studies/{study_id}/results`

### Create report

`POST /v1/studies/{study_id}/reports`

```json
{"report_type":"DOCTOR","language":"en"}
```

### List/download reports

`GET /v1/studies/{study_id}/reports`

### Optional OpenAI explanation

Set server-side only:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=<model-you-have-enabled>
```

Then call:

`POST /v1/reports/{report_id}/explain`

```json
{"language":"Gujarati"}
```

The endpoint deliberately sends measurements/quality/warnings only; it omits patient identifiers and image pixels.

## 6. Persistent storage mapping

PostgreSQL stores metadata/status/relationships/results indexes. MinIO stores binaries.

Original object key:

```text
fetal-originals/{hospital_id}/{study_id}/{scan_id}/original-name.dcm
```

Derived:

```text
fetal-derived/{hospital_id}/{study_id}/{scan_id}/{run_id}/overlay.png
```

Reports:

```text
fetal-reports/{hospital_id}/{study_id}/{report_id}/doctor-en.pdf
```

## 7. Database migrations

For the first local boot the API calls `Base.metadata.create_all()` for convenience. Before cloud/staging, freeze a migration:

```bash
docker compose exec api alembic revision --autogenerate -m "initial schema"
docker compose exec api alembic upgrade head
```

Commit the generated migration under `services/platform/alembic/versions/` and remove `Base.metadata.create_all()` from startup for staging/prod.

## 8. Real ML integration checklist

Verify one known-good sample per anatomy against your current Streamlit/MVP baseline:

- HEAD: route, plane, quality, HC, BPD, OFD, head area, GA, mask/ellipse overlay
- ABDOMEN: route, plane, quality, TAD/APAD-derived AC, overlay, spacing/ROI behavior
- FEMUR: route, plane, quality, FL, overlay, spacing/ROI behavior
- confidence fusion: ACCEPT / REVIEW / REJECT
- model/version metadata

Do not accept the local platform migration until these match your frozen baseline within the numerical tolerance you already use.

## 9. AWS mapping later

No application redesign should be required:

| Local | AWS |
|---|---|
| PostgreSQL | Aurora PostgreSQL Serverless v2 |
| MinIO | S3 |
| ElasticMQ | SQS |
| Keycloak | Cognito User Pools |
| FastAPI Docker | ECS Fargate |
| host/GPU worker | ECS on GPU EC2 |
| local Docker images | ECR |
| `.env` secrets | Secrets Manager |
| local logs | CloudWatch/OpenTelemetry |

## 10. Current intentional omissions

This scaffold does not invent missing medical/model behavior. In particular:

- It does not invent a new EFW formula. Preserve the validated Hadlock implementation from your current pipeline and return `efw_g` from that pipeline/study aggregator.
- It does not invent abdomen/femur ROI localization. Pass your existing ROI/calibration inputs and preserve REVIEW behavior for untrusted cases.
- It does not guess the exact import/signature of your current model runner; that is isolated to `model_adapter.py`.
- PACS/FHIR/payments/multi-region are intentionally outside this first local milestone.
