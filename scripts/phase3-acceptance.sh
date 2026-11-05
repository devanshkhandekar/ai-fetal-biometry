#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUDY_ID="${1:-}"
if [[ -z "$STUDY_ID" ]]; then
  echo "Usage: $0 <real-study-id>" >&2
  exit 2
fi
TOKEN="$($ROOT/scripts/get-local-token.sh)"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
curl -fsS -H "Authorization: Bearer $TOKEN" "http://localhost:8000/v1/studies/$STUDY_ID/results" > "$TMP"
python3 - "$TMP" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
errors=[]
agg=p.get("study_aggregate") or {}
if agg.get("status") != "COMPLETE": errors.append(f"study_aggregate status={agg.get('status')} error={agg.get('error')}")
m=p.get("measurements") or {}
efw=m.get("efw_g")
if not efw: errors.append("efw_g is missing")
else:
    val=efw.get("value") if isinstance(efw,dict) else efw
    if val is None or float(val) <= 0: errors.append(f"invalid efw_g={val}")
for s in p.get("scans",[]):
    r=s.get("result") or {}
    prov=r.get("provenance") or {}
    for k in ("input_sha256","output_sha256","runtime_seconds","release"):
        if not prov.get(k): errors.append(f"{s.get('anatomy')} provenance missing {k}")
    if (r.get("fusion") or {}).get("status") != "ACCEPT": errors.append(f"{s.get('anatomy')} not ACCEPT")
if errors:
    print("PHASE 3 ACCEPTANCE: FAIL")
    for e in errors: print(" -",e)
    raise SystemExit(1)
print("Study aggregate:", json.dumps(agg, indent=2, default=str))
print("EFW:", json.dumps(efw, indent=2, default=str))
print("PHASE 3 ACCEPTANCE: PASS")
PY
