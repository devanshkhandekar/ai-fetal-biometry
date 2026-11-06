import sys, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.confidence_fusion.policy import fuse_confidence

cases=[
("all_good",{"application_plane":"head","status":"accept"},{"label":"good"},{"status":"valid","confidence":0.9},"accept"),
("quality_review",{"application_plane":"head","status":"accept"},{"label":"review"},{"status":"valid","confidence":0.9},"review"),
("quality_poor",{"application_plane":"abdomen","status":"accept"},{"label":"poor"},{"status":"valid","confidence":0.9},"reject"),
("measurement_invalid",{"application_plane":"femur","status":"accept"},{"label":"good"},{"status":"invalid","confidence":0.1},"reject"),
("unsupported",{"application_plane":"unsupported","status":"accept"},{"label":"good"},{"status":"valid"},"reject")
]
for name,p,q,m,exp in cases:
    out=fuse_confidence(p,q,m)
    assert out["status"]==exp,(name,out)
    print("PASS",name,exp)
print("ALL PHASE-3 POLICY TESTS PASSED")
