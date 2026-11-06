from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT/"configs/project_backend.json").read_text())

required = [
    "src/pipeline_devansh.py",
    "src/segmentation/devansh_resunet.py",
    "src/plane_detection/fetalclip_model.py",
    "src/biometry/estimated_fetal_weight.py",
    cfg["head"]["checkpoint"],
    cfg["fetalclip"]["config"],
    cfg["fetalclip"]["weights"],
    cfg["plane"]["probe"],
    cfg["quality"]["probe"],
]

print("Core integration audit")
missing=[]
for rel in required:
    p=ROOT/rel
    ok=p.exists()
    print(("OK   " if ok else "MISS "), rel)
    if not ok:
        missing.append(rel)

print("\nPhase 4/5 single-upload bridge files:")
for rel in [
    "external_adapters/biometrynet_single_infer.py",
    "src/models/landmarks/biometrynet_subprocess.py",
]:
    p=ROOT/rel
    print(("OK   " if p.exists() else "MISS "), rel)

if missing:
    raise SystemExit(f"\nMissing {len(missing)} required core integration file(s).")

print("\nCORE PHASES 0/1/2/3/6/7 READY.")
