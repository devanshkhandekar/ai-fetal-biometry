#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/scripts/phase2.env"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE" >&2; exit 2; }
set -a
source "$ENV_FILE"
set +a
[[ -d "${MODEL_REPO_PATH:-}" ]] || { echo "MODEL_REPO_PATH is invalid" >&2; exit 2; }

echo "=== Candidate run/inference functions ==="
grep -RniE '^def[[:space:]]+(run|infer|analy|process|predict)[A-Za-z0-9_]*[[:space:]]*\(' \
  "$MODEL_REPO_PATH/src" "$MODEL_REPO_PATH/app" 2>/dev/null \
  | grep -Ei 'pipeline|biometry|scan|anatom|route|unified|infer|process|predict' \
  | head -120 || true

echo
echo "=== Files referencing the multi-anatomy components ==="
grep -RniE 'FetalCLIP|route_anatomy|BiometryNetSubprocess|TAD|APAD|femur|HRNET|FL' \
  "$MODEL_REPO_PATH/src" "$MODEL_REPO_PATH/app" 2>/dev/null \
  | head -180 || true
