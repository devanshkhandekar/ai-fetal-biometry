#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CTX="$ROOT/phase4/context"

mkdir -p "$CTX/app"
rm -rf "$CTX/app" "$CTX/bridge_cli.py" "$CTX/real_pipeline_adapter.py"
mkdir -p "$CTX/app"

cp -a "$ROOT/services/platform/app/." "$CTX/app/"
cp "$ROOT/services/platform/requirements-worker-host.txt" "$CTX/requirements-worker-host.txt"
cp "$ROOT/ml/bridge_cli.py" "$CTX/bridge_cli.py"
cp "$ROOT/ml/real_pipeline_adapter.py" "$CTX/real_pipeline_adapter.py"
cp "$ROOT/phase4/Dockerfile.inference.parity" "$CTX/Dockerfile.inference.parity"

if [[ ! -f "$CTX/env/fetal_biometry_unified.tar.gz" ]]; then
  echo "ERROR: packed env missing. Run scripts/phase4-pack-env.sh first." >&2
  exit 2
fi

echo "Phase4 build context:" 
du -sh "$CTX" || true
find "$CTX" -maxdepth 2 -type f -printf '%p\n' | sort | head -80
echo "PHASE4 BUILD CONTEXT: PASS"
