#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API="${API_BASE_URL:-http://localhost:8000}"
KC="${VITE_KEYCLOAK_URL:-http://localhost:8081}"
USER_NAME="${PHASE1_USER:-doctor@local.test}"
PASSWORD="${PHASE1_PASSWORD:-doctor123}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

jget() {
  python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"
}

echo "[1/10] API health"
curl -fsS "$API/health" | python3 -m json.tool

echo "[2/10] Keycloak token"
TOKEN="$(curl -fsS -X POST "$KC/realms/fetal-local/protocol/openid-connect/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode 'client_id=fetal-web' \
  --data-urlencode 'grant_type=password' \
  --data-urlencode "username=$USER_NAME" \
  --data-urlencode "password=$PASSWORD" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')"
AUTH=(-H "Authorization: Bearer $TOKEN")

echo "[3/10] Authenticated identity"
curl -fsS "${AUTH[@]}" "$API/v1/me" | python3 -m json.tool

echo "[4/10] Create study"
STAMP="$(date +%s)"
CREATE_JSON="$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -X POST "$API/v1/studies" \
  -d "{\"patient_external_id\":\"PHASE1-$STAMP\",\"patient_name\":\"Phase One Demo\",\"accession_number\":\"ACC-$STAMP\"}")"
echo "$CREATE_JSON" | python3 -m json.tool
STUDY_ID="$(printf '%s' "$CREATE_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
printf '%s\n' "$STUDY_ID" > .phase1-last-study-id

echo "[5/10] Upload HEAD / ABDOMEN / FEMUR through presigned S3 URLs"
for ANATOMY in HEAD ABDOMEN FEMUR; do
  FILE="$TMP/${ANATOMY,,}.txt"
  printf 'phase-1 mock payload for %s\n' "$ANATOMY" > "$FILE"
  INIT="$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
    -X POST "$API/v1/studies/$STUDY_ID/uploads/initiate" \
    -d "{\"anatomy\":\"$ANATOMY\",\"filename\":\"${ANATOMY,,}.txt\",\"content_type\":\"application/octet-stream\"}")"
  SCAN_ID="$(printf '%s' "$INIT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["scan_id"])')"
  UPLOAD_URL="$(printf '%s' "$INIT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["upload_url"])')"
  curl -fsS -X PUT -H 'Content-Type: application/octet-stream' --data-binary "@$FILE" "$UPLOAD_URL" >/dev/null
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
    -X POST "$API/v1/studies/$STUDY_ID/uploads/complete" \
    -d "{\"scan_id\":\"$SCAN_ID\"}" | python3 -m json.tool
  echo "  uploaded $ANATOMY"
done

echo "[6/10] Queue inference"
curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -X POST "$API/v1/studies/$STUDY_ID/inference" | python3 -m json.tool

echo "[7/10] Wait for mock inference workers"
for i in $(seq 1 60); do
  STUDY="$(curl -fsS "${AUTH[@]}" "$API/v1/studies/$STUDY_ID")"
  STATE="$(printf '%s' "$STUDY" | python3 -c 'import json,sys; d=json.load(sys.stdin); jobs=d.get("jobs",[]); print("FAILED" if any(j["status"]=="FAILED" for j in jobs) else "COMPLETE" if jobs and all(j["status"]=="COMPLETE" for j in jobs) else "WAIT")')"
  if [[ "$STATE" == "FAILED" ]]; then
    echo "$STUDY" | python3 -m json.tool
    echo "Inference failed" >&2
    exit 1
  fi
  if [[ "$STATE" == "COMPLETE" ]]; then
    echo "$STUDY" | python3 -m json.tool
    break
  fi
  if [[ "$i" == "60" ]]; then
    echo "Timed out waiting for inference" >&2
    exit 1
  fi
  sleep 2
done

echo "[8/10] Study-level results"
curl -fsS "${AUTH[@]}" "$API/v1/studies/$STUDY_ID/results" | tee "$TMP/results.json" | python3 -m json.tool

echo "[9/10] Generate and wait for Doctor PDF"
REPORT_CREATE="$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -X POST "$API/v1/studies/$STUDY_ID/reports" \
  -d '{"report_type":"DOCTOR","language":"en"}')"
REPORT_ID="$(printf '%s' "$REPORT_CREATE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
for i in $(seq 1 60); do
  REPORTS="$(curl -fsS "${AUTH[@]}" "$API/v1/studies/$STUDY_ID/reports")"
  STATUS="$(printf '%s' "$REPORTS" | python3 -c 'import json,sys; rid=sys.argv[1]; d=json.load(sys.stdin); r=next(x for x in d if x["id"]==rid); print(r["status"])' "$REPORT_ID")"
  if [[ "$STATUS" == "FAILED" ]]; then
    echo "$REPORTS" | python3 -m json.tool
    echo "Report failed" >&2
    exit 1
  fi
  if [[ "$STATUS" == "COMPLETE" ]]; then
    DOWNLOAD_URL="$(printf '%s' "$REPORTS" | python3 -c 'import json,sys; rid=sys.argv[1]; d=json.load(sys.stdin); r=next(x for x in d if x["id"]==rid); print(r["download_url"])' "$REPORT_ID")"
    curl -fsS "$DOWNLOAD_URL" -o phase1-doctor-report.pdf
    test -s phase1-doctor-report.pdf
    echo "  downloaded: $ROOT/phase1-doctor-report.pdf"
    break
  fi
  if [[ "$i" == "60" ]]; then
    echo "Timed out waiting for report" >&2
    exit 1
  fi
  sleep 2
done

echo "[10/10] PASS"
echo "Study ID: $STUDY_ID"
echo "Saved study id to: .phase1-last-study-id"
echo "Next: docker compose down (WITHOUT -v), start again, then run scripts/phase1_persistence_check.sh"
