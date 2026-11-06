from __future__ import annotations
import json, math
from pathlib import Path

from typing import Any, Dict, Iterable


DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "configs" / "confidence_policy.json"

def load_policy(path=None):
    path = Path(path) if path else DEFAULT_POLICY_PATH
    return json.loads(path.read_text())

def _norm(v):
    return "" if v is None else str(v).strip().lower()

def _finite(v):
    try:
        v=float(v)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def normalize_plane_result(r):
    r=r or {}
    pred=_norm(r.get("predicted_plane") or r.get("plane") or r.get("label"))
    app=_norm(r.get("application_plane"))
    if not app:
        app={
            "brain":"head","fetal brain":"head","head":"head",
            "abdomen":"abdomen","fetal abdomen":"abdomen",
            "femur":"femur","fetal femur":"femur",
            "thorax":"unsupported","fetal thorax":"unsupported",
            "cervix":"unsupported","maternal cervix":"unsupported",
            "other":"unknown"
        }.get(pred,"unknown")
    return {
        "predicted_plane":pred,
        "application_plane":app,
        "status":_norm(r.get("status") or r.get("decision") or "review"),
        "top_score":_finite(r.get("top_score") or r.get("score") or r.get("confidence")),
        "margin":_finite(r.get("margin"))
    }

def normalize_quality_result(r):
    r=r or {}
    if isinstance(r.get("image_quality"),dict):
        r=r["image_quality"]
    return {
        "label":_norm(r.get("policy_label") or r.get("label") or r.get("quality_label") or r.get("raw_prediction") or "review"),
        "score":_finite(r.get("score") or r.get("top_score") or r.get("score_good") or r.get("confidence")),
        "margin":_finite(r.get("margin")),
        "mode":_norm(r.get("mode") or "advisory")
    }

def normalize_measurement_result(r):
    r=r or {}
    status=_norm(r.get("measurement_status") or r.get("status") or r.get("validity_status") or r.get("decision") or "review")
    valid=r.get("valid", r.get("is_valid", r.get("measurement_valid")))
    if isinstance(valid,bool):
        status="valid" if valid and status in {"","review"} else ("invalid" if not valid else status)
    reasons=r.get("reasons") or r.get("warnings") or r.get("reason_codes") or []
    if isinstance(reasons,str):
        reasons=[reasons]
    return {
        "status":status,
        "confidence":_finite(r.get("measurement_confidence") or r.get("confidence") or r.get("quality_score") or r.get("score")),
        "reasons":[str(x) for x in reasons]
    }

# def fuse_confidence(plane_result, quality_result, measurement_result, policy=None):
#     policy=policy or load_policy()
#     plane=normalize_plane_result(plane_result)
#     quality=normalize_quality_result(quality_result)
#     meas=normalize_measurement_result(measurement_result)
#     reasons=[]; warnings=[]; decision="accept"

#     if plane["application_plane"] not in set(policy["supported_application_planes"]):
#         decision="reject"; reasons.append(f"unsupported_or_unknown_plane:{plane['application_plane']}")
#     elif plane["status"] in set(policy["plane"]["reject_statuses"]):
#         decision="reject"; reasons.append(f"plane_rejected:{plane['status']}")
#     elif plane["status"] not in set(policy["plane"]["accept_statuses"]):
#         decision="review"; reasons.append(f"plane_not_fully_accepted:{plane['status']}")

#     if quality["label"] in set(policy["quality"]["reject_labels"]):
#         decision="reject"; reasons.append(f"image_quality:{quality['label']}")
#     elif quality["label"] in set(policy["quality"]["review_labels"]):
#         if decision!="reject": decision="review"
#         reasons.append(f"image_quality:{quality['label']}")
#     elif quality["label"] not in set(policy["quality"]["accept_labels"]):
#         if decision!="reject": decision="review"
#         reasons.append(f"image_quality_unknown:{quality['label']}")

#     mcfg=policy["measurement"]
#     if meas["status"] in set(mcfg["reject_statuses"]):
#         decision="reject"; reasons.append(f"measurement_rejected:{meas['status']}")
#     elif meas["status"] in set(mcfg["review_statuses"]):
#         if decision!="reject": decision="review"
#         reasons.append(f"measurement_review:{meas['status']}")
#     elif meas["status"] not in set(mcfg["accept_statuses"]):
#         if decision!="reject": decision="review"
#         reasons.append(f"measurement_status_unknown:{meas['status']}")

#     if meas["confidence"] is not None:
#         if meas["confidence"] < float(mcfg["reject_below_confidence"]):
#             decision="reject"; reasons.append(f"measurement_confidence_critical:{meas['confidence']:.4f}")
#         elif meas["confidence"] < float(mcfg["review_below_confidence"]):
#             if decision!="reject": decision="review"
#             reasons.append(f"measurement_confidence_low:{meas['confidence']:.4f}")

#     warnings += [f"measurement:{x}" for x in meas["reasons"]]

#     return {
#         "phase":3,
#         "policy_version":policy.get("policy_version","unknown"),
#         "status":decision,
#         "measurement_usable":decision in {"accept","review"},
#         "automatic_accept":decision=="accept",
#         "review_required":decision=="review",
#         "discard_measurement":decision=="reject",
#         "application_plane":plane["application_plane"],
#         "reasons":reasons,
#         "warnings":warnings,
#         "signals":{"plane":plane,"quality":quality,"measurement":meas}
#     }






def _text(
    value: Any,
) -> str:

    return str(
        value or ""
    ).strip().lower()


def _as_list(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return (
            [value]
            if value.strip()
            else []
        )

    if isinstance(
        value,
        Iterable,
    ):
        return [
            str(x)
            for x in value
            if str(x).strip()
        ]

    return [str(value)]


def fuse_confidence(
    plane: Dict[str, Any],
    quality: Dict[str, Any],
    measurement: Dict[str, Any],
    *,
    quality_affects_decision: bool = True,
    quality_policy_applicable: bool = True,
) -> Dict[str, Any]:

    plane = dict(
        plane or {}
    )

    quality = dict(
        quality or {}
    )

    measurement = dict(
        measurement or {}
    )

    plane_status = _text(
        plane.get("status")
    )

    quality_label = _text(
        quality.get("label")
        or quality.get(
            "raw_prediction"
        )
    )

    measurement_status = _text(
        measurement.get("status")
    )

    application_plane = _text(
        plane.get(
            "application_plane"
        )
        or plane.get(
            "predicted_plane"
        )
    )

    reasons = []
    warnings = []

    measurement_reasons = (
        _as_list(
            measurement.get(
                "reasons"
            )
        )
        + _as_list(
            measurement.get(
                "reason_codes"
            )
        )
        + _as_list(
            measurement.get(
                "warnings"
            )
        )
        + _as_list(
            measurement.get(
                "flags"
            )
        )
    )

    # --------------------------------
    # Measurement is always mandatory.
    # --------------------------------

    if measurement_status in {
        "invalid",
        "reject",
        "rejected",
        "fail",
        "failed",
    }:

        status = "REJECT"

        reasons.append(
            "measurement_invalid"
        )

        measurement_usable = False
        discard_measurement = True

    elif measurement_status in {
        "review",
        "warn",
        "warning",
    }:

        status = "REVIEW"

        reasons.append(
            "measurement_review:review"
        )

        measurement_usable = True
        discard_measurement = False

    elif measurement_status not in {
        "valid",
        "accept",
        "accepted",
        "pass",
        "good",
    }:

        # Do not silently ACCEPT if the
        # measurement backend forgot to
        # provide a validity status.
        status = "REVIEW"

        reasons.append(
            "measurement_status_missing_or_unknown"
        )

        measurement_usable = True
        discard_measurement = False

    else:

        measurement_usable = True
        discard_measurement = False

        # --------------------------
        # Plane always participates.
        # --------------------------

        if plane_status != "accept":

            status = "REVIEW"

            reasons.append(
                "plane_review:review"
            )

        # --------------------------------------
        # IQA participates only where validated
        # for current product policy.
        # --------------------------------------

        elif (
            quality_affects_decision
            and
            quality_policy_applicable
        ):

            if quality_label == "good":

                status = "ACCEPT"

            elif quality_label == "poor":

                status = "REVIEW"

                reasons.append(
                    "image_quality:poor"
                )

            else:

                status = "REVIEW"

                reasons.append(
                    "image_quality:"
                    + (
                        quality_label
                        or "review"
                    )
                )

        else:

            # Abdomen/Femur current policy.
            status = "ACCEPT"

            if (
                quality_label
                and
                quality_label
                != "good"
            ):

                warnings.append(
                    "image_quality_"
                    "advisory_only:"
                    f"{quality_label}"
                )

    for reason in measurement_reasons:

        tagged = (
            f"measurement:{reason}"
        )

        if status == "REVIEW":

            if tagged not in reasons:
                reasons.append(tagged)

        else:

            if tagged not in warnings:
                warnings.append(tagged)

    reasons = list(
        dict.fromkeys(reasons)
    )

    warnings = list(
        dict.fromkeys(warnings)
    )

    return {

        "phase": 3,

        "status": status,

        "reasons": reasons,

        "warnings": warnings,

        "policy_version":
            "phase3_v2_anatomy_aware_quality",

        "application_plane":
            application_plane,

        "quality_policy_applicable":
            bool(
                quality_policy_applicable
            ),

        "quality_affects_decision":
            bool(
                quality_affects_decision
            ),

        "review_required":
            status != "ACCEPT",

        "automatic_accept":
            status == "ACCEPT",

        "measurement_usable":
            measurement_usable,

        "discard_measurement":
            discard_measurement,

        "signals": {

            "plane": {

                "application_plane":
                    application_plane,

                "predicted_plane":
                    plane.get(
                        "predicted_plane"
                    ),

                "status":
                    plane.get(
                        "status"
                    ),

                "top_score":
                    plane.get(
                        "top_score"
                    ),

                "margin":
                    plane.get(
                        "margin"
                    ),
            },

            "quality": {

                "label":
                    quality.get(
                        "label"
                    ),

                "raw_prediction":
                    quality.get(
                        "raw_prediction"
                    ),

                "score":
                    quality.get(
                        "score"
                    ),

                "margin":
                    quality.get(
                        "margin"
                    ),

                "mode":
                    quality.get(
                        "mode"
                    ),

                "policy_applicable":
                    bool(
                        quality_policy_applicable
                    ),

                "affects_decision":
                    bool(
                        quality_affects_decision
                    ),

                "scope":
                    quality.get(
                        "scope"
                    ),
            },

            "measurement": {

                "status":
                    measurement.get(
                        "status"
                    ),

                "reasons":
                    measurement_reasons,

                "confidence":
                    measurement.get(
                        "confidence"
                    ),
            },
        },
    }