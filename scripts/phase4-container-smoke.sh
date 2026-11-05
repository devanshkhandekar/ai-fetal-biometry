#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${PHASE4_IMAGE:-fetal-biometry-inference:phase4-parity}"

sudo docker run --rm --gpus all \
  -v "$ROOT/phase4/model-runtime:/model_repo:ro" \
  -w /model_repo \
  -e PYTHONPATH=/model_repo \
  "$IMAGE" \
  /opt/fetal_unified/bin/python - <<'PY'
import sys
import torch
import yacs

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

from src.orchestration.saas_callable import run_scan, calculate_study_efw
print("run_scan:", run_scan)
print("calculate_study_efw:", calculate_study_efw)
print("PHASE4 CONTAINER SMOKE: PASS")
PY
