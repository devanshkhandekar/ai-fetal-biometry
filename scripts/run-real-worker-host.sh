#!/usr/bin/env bash
set -euo pipefail
# Run this FROM YOUR EXISTING WORKING fetal-biometry Conda/venv.
# It reuses Docker PostgreSQL/MinIO/ElasticMQ but executes the ML pipeline in
# the environment where it already works today.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
source "$ROOT/.env"
set +a
export DATABASE_URL="postgresql+psycopg://fetal:fetal_local_password@localhost:5432/fetal_biometry"
export S3_INTERNAL_ENDPOINT="http://localhost:9000"
export S3_PUBLIC_ENDPOINT="http://localhost:9000"
export SQS_ENDPOINT="http://localhost:9324"
export OIDC_JWKS_URL="http://localhost:8081/realms/fetal-local/protocol/openid-connect/certs"
export PIPELINE_MODE=python
cd "$ROOT/services/platform"
python -m app.worker_inference
