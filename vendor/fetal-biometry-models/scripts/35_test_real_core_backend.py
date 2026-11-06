import json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.orchestration.scan_input import ScanInput
from src.orchestration.project_backend import ProjectBackend
from src.confidence_fusion.policy import fuse_confidence

image = ROOT/"data/sample_cases/019_HC.png"
if not image.exists():
    image = next((ROOT/"data/sample_cases").glob("*_HC.png"))

backend=ProjectBackend()
scan=ScanInput(image_path=str(image), pixel_spacing_mm=0.10)

print("\n--- PHASE 1 ---")
plane=backend.detect_plane(scan)
print(json.dumps(plane,indent=2))

print("\n--- PHASE 2 ---")
quality=backend.assess_quality(scan)
print(json.dumps(quality,indent=2))

print("\n--- PHASE 0 ---")
head=backend.measure_head(scan)
head_view={k:v for k,v in head.items() if k!="phase0_payload"}
print(json.dumps(head_view,indent=2,default=str))

print("\n--- PHASE 3 ---")
fusion=fuse_confidence(plane,quality,head)
print(json.dumps(fusion,indent=2))

assert quality["label"] in {"good","review","poor"}
assert head["status"] in {"valid","review","invalid"}
assert fusion["status"] in {"accept","review","reject"}

print("\nPASS: real core backend Phase 0+1+2+3 smoke test.")
