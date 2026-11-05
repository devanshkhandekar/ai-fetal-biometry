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
p = json.load(open(sys.argv[1]))
scans = p.get("scans", [])
by = {s.get("anatomy"): s for s in scans}
errors=[]
required = {
    "HEAD": {"hc_mm","bpd_mm","ofd_mm"},
    "ABDOMEN": {"ac_mm"},
    "FEMUR": {"fl_mm"},
}
for anatomy, keys in required.items():
    s=by.get(anatomy)
    if not s:
        errors.append(f"missing scan: {anatomy}"); continue
    if s.get("status") != "COMPLETE": errors.append(f"{anatomy} status={s.get('status')}")
    r=s.get("result") or {}
    warnings=' '.join(r.get("warnings") or [])
    if "MOCK PIPELINE RESULT" in warnings: errors.append(f"{anatomy} is still MOCK")
    missing=keys-set((r.get("measurements") or {}).keys())
    if missing: errors.append(f"{anatomy} missing measurements {sorted(missing)}")
    source=str(r.get("source") or "").lower()
    if source == "mock": errors.append(f"{anatomy} source=mock")
    print(f"{anatomy}: route={r.get('route')} fusion={(r.get('fusion') or {}).get('status')} device={r.get('device')}")
    print("  measurements:", sorted((r.get("measurements") or {}).keys()))
if errors:
    print("\nPHASE 2 ACCEPTANCE: FAIL")
    for e in errors: print(" -", e)
    raise SystemExit(1)
print("\nPHASE 2 ACCEPTANCE: PASS")
PY
