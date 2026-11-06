#!/usr/bin/env bash
set -euo pipefail

python --version
python -m pip freeze > requirements_phase2_quality_frozen.txt
conda env export --no-builds > environment_phase2_quality.yml

echo "Created:"
echo "  requirements_phase2_quality_frozen.txt"
echo "  environment_phase2_quality.yml"
