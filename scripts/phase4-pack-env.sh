#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${1:-fetal_biometry_unified}"

OUT_DIR="$ROOT/phase4/context/env"
OUT_TAR="$OUT_DIR/fetal_biometry_unified.tar.gz"

mkdir -p "$OUT_DIR"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found" >&2
    exit 2
fi

if ! conda run -n base python -c \
    'import conda_pack' >/dev/null 2>&1
then
    echo "Installing conda-pack into base environment..."

    conda run -n base \
        python -m pip install conda-pack
fi

# conda run may emit a trailing blank line.
# Strip blank lines before taking the final value.
ENV_PREFIX="$(
    conda run -n "$ENV_NAME" \
        python -c 'import sys; print(sys.prefix)' \
    | sed '/^[[:space:]]*$/d' \
    | tail -n 1
)"

if [[ -z "$ENV_PREFIX" ]]; then
    echo "ERROR: could not determine environment prefix" >&2
    exit 2
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "ERROR: invalid environment prefix: $ENV_PREFIX" >&2
    exit 2
fi

echo "Packing environment: $ENV_NAME"
echo "Prefix: $ENV_PREFIX"

du -sh "$ENV_PREFIX"

rm -f "$OUT_TAR"

# Pack by explicit prefix so there is no ambiguity over env resolution.
conda run -n base \
    conda-pack \
    -p "$ENV_PREFIX" \
    -o "$OUT_TAR"

if [[ ! -s "$OUT_TAR" ]]; then
    echo "ERROR: packed environment was not created" >&2
    exit 2
fi

echo
ls -lh "$OUT_TAR"

echo
echo "PHASE4 ENV PACK: PASS"
