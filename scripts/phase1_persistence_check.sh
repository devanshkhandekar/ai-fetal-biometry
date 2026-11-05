#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
API="${API_BASE_URL:-http://localhost:8000}"
KC="${VITE_KEYCLOAK_URL:-http://localhost:8081}"
USER_NAME="${PHASE1_USER:-doctor@local.test}"
PASSWORD="${PHASE1_PASSWORD:-doctor123}"

if [[ ! -s .phase1-last-study-id ]]; then
  echo "Missing .phase1-last-study-id. Run scripts/phase1_acceptance.sh first." >&2
  exit 1
fi
STUDY_ID="$(cat .phase1-last-study-id)"
TOKEN="$(curl -fsS -X POST "$KC/realms/fetal-local/protocol/openid-connect/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'client_id=fetal-web' \
  --data-urlencode 'grant_type=password' \
  --data-urlencode "username=$USER_NAME" \
  --data-urlencode "password=$PASSWORD" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
AUTH=(-H "Authorization: Bearer $TOKEN")

echo "Checking persisted study $STUDY_ID"
STUDY="$(curl -fsS "${AUTH[@]}" "$API/v1/studies/$STUDY_ID")"
echo "$STUDY" | python3 -m json.tool
printf '%s' "$STUDY" | python3 -c '
import json,sys
s=json.load(sys.stdin)
assert len(s.get("scans",[])) == 3, "expected 3 scans, got %s" % len(s.get("scans",[]))
assert all(x.get("status")=="COMPLETE" for x in s["scans"]), "not all scans persisted COMPLETE"
assert len(s.get("jobs",[])) == 3, "expected 3 persisted jobs"
assert all(x.get("status")=="COMPLETE" for x in s["jobs"]), "not all jobs persisted COMPLETE"
print("Database persistence: PASS")
'
REPORTS="$(curl -fsS "${AUTH[@]}" "$API/v1/studies/$STUDY_ID/reports")"
echo "$REPORTS" | python3 -m json.tool
printf '%s' "$REPORTS" | python3 -c '
import json,sys
r=json.load(sys.stdin)
assert any(x.get("status")=="COMPLETE" and x.get("download_url") for x in r), "no persisted completed report"
print("Report/object metadata persistence: PASS")
'
DOWNLOAD_URL="$(printf '%s' "$REPORTS" | python3 -c 'import json,sys; r=json.load(sys.stdin); print(next(x["download_url"] for x in r if x.get("status")=="COMPLETE" and x.get("download_url")))')"
curl -fsS "$DOWNLOAD_URL" -o /tmp/phase1-persisted-report.pdf
test -s /tmp/phase1-persisted-report.pdf
echo "Object storage persistence: PASS"
echo "PHASE 1 PERSISTENCE ACCEPTANCE: PASS"
