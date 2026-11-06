import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.orchestration.demo_backend import DemoBackend
from src.orchestration.unified_pipeline import UnifiedFetalBiometryPipeline

for anatomy in ["head","abdomen","femur"]:
    pipe=UnifiedFetalBiometryPipeline(DemoBackend(anatomy=anatomy,quality="good"),routing_enabled=True)
    out=pipe.run("dummy")
    assert out["route"]["route_status"]=="routed"
    assert out["fusion"]["status"]=="accept"
    assert out["measurement"] is not None
    print("PASS unified route:",anatomy)

out=UnifiedFetalBiometryPipeline(DemoBackend("head","poor"),routing_enabled=True).run("dummy")
assert out["route"]["route_status"]=="rejected" and out["measurement"] is None
print("PASS poor-quality rejection")

out=UnifiedFetalBiometryPipeline(DemoBackend("thorax","good"),routing_enabled=True).run("dummy")
assert out["route"]["route_status"]=="unsupported" and out["measurement"] is None
print("PASS unsupported anatomy")

print("ALL PHASE-8 ORCHESTRATION CONTRACT TESTS PASSED")
