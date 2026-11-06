from src.confidence_fusion.policy import normalize_plane_result, normalize_quality_result

ROUTE_MAP={"head":"head_biometry","abdomen":"abdomen_biometry","femur":"femur_biometry"}

def route_anatomy(plane_result, quality_result, *, routing_enabled, allow_quality_review=True):
    plane=normalize_plane_result(plane_result)
    quality=normalize_quality_result(quality_result)

    if not routing_enabled:
        return {"phase":6,"routing_enabled":False,"route_status":"disabled","route":None,
                "application_plane":plane["application_plane"],"review_required":False,
                "reason":"routing_feature_flag_disabled"}

    if plane["status"] not in {"accept","accepted","ready"}:
        return {"phase":6,"routing_enabled":True,"route_status":"review","route":None,
                "application_plane":plane["application_plane"],"review_required":True,
                "reason":f"plane_not_accepted:{plane['status']}"}

    app=plane["application_plane"]
    if app not in ROUTE_MAP:
        status="unsupported" if app=="unsupported" else "unknown"
        return {"phase":6,"routing_enabled":True,"route_status":status,"route":None,
                "application_plane":app,"review_required":app=="unknown",
                "reason":f"no_supported_route:{app}"}

    if quality["label"]=="poor":
        return {"phase":6,"routing_enabled":True,"route_status":"rejected","route":None,
                "application_plane":app,"review_required":True,"reason":"poor_image_quality"}

    if quality["label"]=="review" and not allow_quality_review:
        return {"phase":6,"routing_enabled":True,"route_status":"review","route":None,
                "application_plane":app,"review_required":True,
                "reason":"quality_review_requires_manual_route"}

    if quality["label"] not in {"good","review"}:
        return {"phase":6,"routing_enabled":True,"route_status":"review","route":None,
                "application_plane":app,"review_required":True,
                "reason":f"unknown_quality_label:{quality['label']}"}

    return {"phase":6,"routing_enabled":True,"route_status":"routed","route":ROUTE_MAP[app],
            "application_plane":app,"review_required":quality["label"]=="review",
            "reason":"quality_review_advisory_route" if quality["label"]=="review" else "route_conditions_passed"}
