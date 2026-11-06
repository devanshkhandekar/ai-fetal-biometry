#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/external/Multicentre-Fetal-Biometry:${PYTHONPATH:-}"

exec /home/devanshot/miniconda3/envs/fetal_biometry_unified/bin/streamlit \
  run app/streamlit_unified_mvp.py
