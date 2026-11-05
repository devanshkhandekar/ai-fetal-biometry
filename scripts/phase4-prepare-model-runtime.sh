#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/scripts/phase2.env"
DEST="$ROOT/phase4/model-runtime"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: missing $ENV_FILE" >&2
  exit 2
fi
set -a
source "$ENV_FILE"
set +a

: "${MODEL_REPO_PATH:?MODEL_REPO_PATH must be set in scripts/phase2.env}"
: "${PIPELINE_PYTHON:?PIPELINE_PYTHON must be set in scripts/phase2.env}"
HRNET_PYTHON="${HRNET_PYTHON:-$PIPELINE_PYTHON}"

if [[ ! -d "$MODEL_REPO_PATH/src" ]]; then
  echo "ERROR: invalid MODEL_REPO_PATH=$MODEL_REPO_PATH" >&2
  exit 2
fi

rm -rf "$DEST"
mkdir -p "$DEST"

# Preserve source, checkpoints, probes/policies, HRNet repo/checkpoints and annotation CSVs.
# Exclude raw image datasets and obvious transient caches only.
rsync -a \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'data/HC18_dataset/' \
  --exclude 'data/fetal_planes_db/Images/' \
  --exclude 'data/acouslic-ai/' \
  "$MODEL_REPO_PATH/" "$DEST/"

export SRC_MODEL_REPO="$MODEL_REPO_PATH"
export SRC_PIPELINE_PY="$PIPELINE_PYTHON"
export SRC_HRNET_PY="$HRNET_PYTHON"
export DEST_MODEL_REPO="$DEST"

python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DEST_MODEL_REPO"])
old_repo = os.environ["SRC_MODEL_REPO"].rstrip("/")
old_py = os.environ["SRC_PIPELINE_PY"]
old_hr = os.environ["SRC_HRNET_PY"]
new_repo = "/model_repo"
new_py = "/opt/fetal_unified/bin/python"

changed = []

def repl(v):
    if isinstance(v, dict):
        return {k: repl(x) for k, x in v.items()}
    if isinstance(v, list):
        return [repl(x) for x in v]
    if isinstance(v, str):
        out = v
        if old_repo and old_repo in out:
            out = out.replace(old_repo, new_repo)
        if old_py and old_py in out:
            out = out.replace(old_py, new_py)
        if old_hr and old_hr in out:
            out = out.replace(old_hr, new_py)
        return out
    return v

for p in sorted((root / "configs").glob("*.json")):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    new = repl(data)
    if new != data:
        p.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
        changed.append(str(p.relative_to(root)))

print("Container-path config rewrites:")
for p in changed:
    print(" -", p)
PY

if [[ ! -f "$DEST/src/orchestration/saas_callable.py" ]]; then
  echo "ERROR: src/orchestration/saas_callable.py missing in runtime copy" >&2
  exit 2
fi
if [[ ! -f "$DEST/configs/project_backend.json" ]]; then
  echo "ERROR: configs/project_backend.json missing" >&2
  exit 2
fi
if [[ ! -f "$DEST/configs/biometrynet_bridge.json" ]]; then
  echo "ERROR: configs/biometrynet_bridge.json missing" >&2
  exit 2
fi

# Fail if the known host Python/repo strings still remain in JSON configs.
if grep -RniF "$MODEL_REPO_PATH" "$DEST/configs" --include='*.json'; then
  echo "ERROR: host model repo path still present in runtime configs" >&2
  exit 2
fi
if grep -RniF "$PIPELINE_PYTHON" "$DEST/configs" --include='*.json'; then
  echo "ERROR: host pipeline Python still present in runtime configs" >&2
  exit 2
fi
if [[ -n "$HRNET_PYTHON" ]] && grep -RniF "$HRNET_PYTHON" "$DEST/configs" --include='*.json'; then
  echo "ERROR: host HRNet Python still present in runtime configs" >&2
  exit 2
fi

echo "Runtime model tree size:"
du -sh "$DEST" || true
echo "PHASE4 MODEL RUNTIME: PASS"
