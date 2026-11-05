#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/scripts/phase2.env"
VENV="$ROOT/.venv-worker"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE does not exist. Copy scripts/phase2.env.example -> scripts/phase2.env and edit it." >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: $VENV is missing. Run scripts/phase2-preflight.sh first." >&2
  exit 2
fi

set -a
source "$ROOT/.env"
source "$ENV_FILE"
set +a

# Docker-internal names must be replaced by host endpoints for this host worker.
export DATABASE_URL="postgresql+psycopg://fetal:${POSTGRES_PASSWORD:-fetal_local_password}@localhost:5432/${POSTGRES_DB:-fetal_biometry}"
export S3_INTERNAL_ENDPOINT="http://localhost:9000"
export S3_PUBLIC_ENDPOINT="http://localhost:9000"
export SQS_ENDPOINT="http://localhost:9324"
export OIDC_JWKS_URL="http://localhost:8081/realms/fetal-local/protocol/openid-connect/certs"
export PIPELINE_MODE="subprocess"

cd "$ROOT/services/platform"
exec "$VENV/bin/python" -m app.worker_inference
