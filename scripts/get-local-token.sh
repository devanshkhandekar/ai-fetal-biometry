#!/usr/bin/env bash
set -euo pipefail
USERNAME="${1:-doctor@local.test}"
PASSWORD="${2:-doctor123}"
curl -fsS -X POST 'http://localhost:8081/realms/fetal-local/protocol/openid-connect/token' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'client_id=fetal-web' \
  --data-urlencode 'grant_type=password' \
  --data-urlencode "username=${USERNAME}" \
  --data-urlencode "password=${PASSWORD}" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
