#!/usr/bin/env bash
set -euo pipefail
curl -fsS http://localhost:8000/health | python -m json.tool
echo "API OK"
echo "Open http://localhost:5173 and login with doctor@local.test / doctor123"
