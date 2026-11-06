from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT/"configs/biometrynet_bridge.json").read_text())

repo = ROOT / cfg["repo_root"]

checks = [
    ("HRNet Python", Path(cfg["hrnet_python"])),
    ("Bridge script", ROOT / cfg["single_infer_script"]),
    ("HRNet repo", repo),
    ("Abdomen metadata", repo / cfg["abdomen"]["metadata_csv"]),
    ("Femur metadata", repo / cfg["femur"]["metadata_csv"]),
]

for metric in ["TAD", "APAD"]:
    checks += [
        (f"{metric} config", repo / cfg["abdomen"][metric]["config"]),
        (f"{metric} checkpoint", repo / cfg["abdomen"][metric]["checkpoint"]),
    ]
checks += [
    ("FL config", repo / cfg["femur"]["FL"]["config"]),
    ("FL checkpoint", repo / cfg["femur"]["FL"]["checkpoint"]),
]

missing=[]
for name,p in checks:
    ok=p.exists()
    print(("OK   " if ok else "MISS "), f"{name}: {p}")
    if not ok:
        missing.append((name,str(p)))

if missing:
    raise SystemExit(f"\nMissing {len(missing)} bridge dependency/dependencies.")

print("\nPHASE 4/5 SINGLE-UPLOAD BRIDGE DEPENDENCIES READY.")
