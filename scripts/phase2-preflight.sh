#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/scripts/phase2.env"
VENV="$ROOT/.venv-worker"

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ROOT/scripts/phase2.env.example" "$ENV_FILE"
  echo "Created $ENV_FILE"
  echo "Edit its /ABSOLUTE/... paths, then rerun this script."
  exit 2
fi

set -a
source "$ROOT/.env"
source "$ENV_FILE"
set +a

for name in PIPELINE_PYTHON PIPELINE_BRIDGE_SCRIPT MODEL_REPO_PATH; do
  value="${!name:-}"
  if [[ -z "$value" || "$value" == /ABSOLUTE/* ]]; then
    echo "ERROR: $name is not configured in scripts/phase2.env" >&2
    exit 2
  fi
done
[[ -x "$PIPELINE_PYTHON" ]] || { echo "ERROR: PIPELINE_PYTHON is not executable: $PIPELINE_PYTHON" >&2; exit 2; }
[[ -f "$PIPELINE_BRIDGE_SCRIPT" ]] || { echo "ERROR: bridge not found: $PIPELINE_BRIDGE_SCRIPT" >&2; exit 2; }
[[ -d "$MODEL_REPO_PATH" ]] || { echo "ERROR: model repo not found: $MODEL_REPO_PATH" >&2; exit 2; }

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/services/platform/requirements-worker-host.txt"

export PIPELINE_MODEL_RELEASE="${PIPELINE_MODEL_RELEASE:-phase0-7-real-local-v1}"
SELF_TEST='{"operation":"self_test"}'
echo "$SELF_TEST" | "$PIPELINE_PYTHON" "$PIPELINE_BRIDGE_SCRIPT" | python3 -m json.tool

echo
echo "Phase 2 preflight PASS."
echo "Next: stop ONLY inference-worker-mock, then run scripts/run-phase2-real-worker.sh"
