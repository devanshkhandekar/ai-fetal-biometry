#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="/home/devanshot/miniconda3/envs/fetal_biometry_unified/bin/python"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/external/Multicentre-Fetal-Biometry:${PYTHONPATH:-}"

echo "== compile =="
"$PY" -m py_compile \
  app/streamlit_unified_mvp.py \
  src/orchestration/project_backend.py \
  src/orchestration/unified_pipeline.py \
  src/confidence_fusion/policy.py \
  src/routing/router.py \
  src/models/landmarks/biometrynet_subprocess.py

echo "== imports =="
"$PY" - <<'PY'
from src.orchestration.project_backend import ProjectBackend
from src.orchestration.unified_pipeline import UnifiedFetalBiometryPipeline
from src.orchestration.scan_input import ScanInput

print("ProjectBackend import: OK")
print("UnifiedFetalBiometryPipeline import: OK")
print("ScanInput import: OK")
PY

echo "STREAMLIT INTEGRATION SMOKE CHECK PASSED"
