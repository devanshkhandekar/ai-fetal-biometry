#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== Phase 5 finalization =="

# 1) Make sure Phase-5 CSS is imported exactly once.
MAIN="apps/web/src/main.tsx"
if [[ ! -f "$MAIN" ]]; then
  echo "ERROR: $MAIN not found"
  exit 1
fi

if ! grep -q "styles/phase5.css" "$MAIN"; then
  python3 - "$MAIN" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")

needle = "import './index.css'"
replacement = "import './index.css'\nimport './styles/phase5.css'"

if needle not in text:
    raise SystemExit(
        "Could not find import './index.css' in apps/web/src/main.tsx"
    )

p.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
PY
  echo "Added Phase-5 CSS import."
else
  echo "Phase-5 CSS import already present."
fi

# 2) Detect report-worker Dockerfile from merged Compose config.
TMP_JSON="$(mktemp)"
sudo docker compose \
  -f docker-compose.yml \
  -f docker-compose.phase5-local.yml \
  config --format json > "$TMP_JSON"

readarray -t BUILD_INFO < <(
  python3 - "$TMP_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)

svc = cfg["services"]["report-worker"]
build = svc.get("build")

if not build:
    raise SystemExit("report-worker has no build configuration")

if isinstance(build, str):
    print(build)
    print("Dockerfile")
else:
    print(build.get("context", "."))
    print(build.get("dockerfile", "Dockerfile"))
PY
)

rm -f "$TMP_JSON"

CONTEXT="${BUILD_INFO[0]}"
DOCKERFILE="${BUILD_INFO[1]}"

if [[ "$CONTEXT" = /* ]]; then
  DOCKER_PATH="$CONTEXT/$DOCKERFILE"
else
  DOCKER_PATH="$ROOT/$CONTEXT/$DOCKERFILE"
fi

DOCKER_PATH="$(realpath -m "$DOCKER_PATH")"

echo "Report-worker Dockerfile: $DOCKER_PATH"

if [[ ! -f "$DOCKER_PATH" ]]; then
  echo "ERROR: Dockerfile not found: $DOCKER_PATH"
  exit 1
fi

cp -n "$DOCKER_PATH" "${DOCKER_PATH}.before_phase5_fonts" || true

# 3) Add multilingual fonts only if not already present.
if grep -q "fonts-noto-core" "$DOCKER_PATH"; then
  echo "Noto fonts already configured."
else
  python3 - "$DOCKER_PATH" <<'PY'
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")

# Prefer extending an existing apt-get install command.
m = re.search(
    r"(apt-get\s+install\s+-y(?:\s+--no-install-recommends)?\s+)(.*?)(?=(?:&&|\\\n\s*&&|\nRUN|\Z))",
    text,
    flags=re.S,
)

if m:
    segment = m.group(0)
    if "fonts-noto-core" not in segment:
        new_segment = segment.rstrip() + " \\\n       fonts-noto-core \\\n       fonts-noto-extra \\\n       fonts-noto-cjk"
        text = text[:m.start()] + new_segment + text[m.end():]
else:
    # Safe independent RUN layer for Debian/Ubuntu based Python images.
    block = """
RUN apt-get update \\
    && apt-get install -y --no-install-recommends \\
       fonts-noto-core \\
       fonts-noto-extra \\
       fonts-noto-cjk \\
    && rm -rf /var/lib/apt/lists/*
"""
    # Insert after FROM so the layer remains valid.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("FROM "):
            lines.insert(i + 1, block.strip())
            text = "\n".join(lines) + "\n"
            break
    else:
        raise SystemExit("Could not find FROM line in report-worker Dockerfile")

p.write_text(text, encoding="utf-8")
PY

  echo "Added Noto multilingual fonts."
fi

echo
echo "Now run:"
echo "  cd apps/web && npx tsc -b && npm run build && cd ../.."
echo "  source scripts/load-phase5-env.sh"
echo "  sudo docker compose -f docker-compose.yml -f docker-compose.phase5-local.yml build web report-worker"
echo "  sudo docker compose -f docker-compose.yml -f docker-compose.phase5-local.yml up -d --force-recreate --no-deps web report-worker"
