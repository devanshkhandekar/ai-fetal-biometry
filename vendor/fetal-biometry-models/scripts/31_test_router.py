import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.routing.router import route_anatomy

cases=[
("head",{"application_plane":"head","status":"accept"},{"label":"good"},True,"routed","head_biometry"),
("abd",{"application_plane":"abdomen","status":"accept"},{"label":"good"},True,"routed","abdomen_biometry"),
("femur",{"application_plane":"femur","status":"accept"},{"label":"review"},True,"routed","femur_biometry"),
("poor",{"application_plane":"head","status":"accept"},{"label":"poor"},True,"rejected",None),
("thorax",{"application_plane":"unsupported","status":"accept"},{"label":"good"},True,"unsupported",None),
("disabled",{"application_plane":"head","status":"accept"},{"label":"good"},False,"disabled",None),
]
for name,p,q,en,exp_status,exp_route in cases:
    out=route_anatomy(p,q,routing_enabled=en)
    assert out["route_status"]==exp_status,(name,out)
    assert out["route"]==exp_route,(name,out)
    print("PASS",name,exp_status,exp_route)
print("ALL PHASE-6 ROUTER TESTS PASSED")
