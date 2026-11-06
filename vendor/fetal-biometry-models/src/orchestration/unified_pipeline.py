# from src.confidence_fusion.policy import fuse_confidence
# from src.routing.router import route_anatomy

# class UnifiedFetalBiometryPipeline:
#     def __init__(self, backend, *, routing_enabled=False, allow_quality_review_route=True):
#         self.backend = backend
#         self.routing_enabled = bool(routing_enabled)
#         self.allow_quality_review_route = bool(allow_quality_review_route)

#     def _measure(self, route, scan):
#         if route == "head_biometry":
#             return self.backend.measure_head(scan)
#         if route == "abdomen_biometry":
#             return self.backend.measure_abdomen(scan)
#         if route == "femur_biometry":
#             return self.backend.measure_femur(scan)
#         raise ValueError(f"Unsupported route: {route}")

#     def run(self, scan):
#         plane = self.backend.detect_plane(scan)
#         quality = self.backend.assess_quality(scan)
#         route = route_anatomy(
#             plane,
#             quality,
#             routing_enabled=self.routing_enabled,
#             allow_quality_review=self.allow_quality_review_route,
#         )

#         result = {
#             "pipeline_version": "phase8_v2",
#             "plane": plane,
#             "quality": quality,
#             "route": route,
#             "measurement": None,
#             "fusion": None,
#         }

#         if route["route_status"] != "routed":
#             return result

#         measurement = self._measure(route["route"], scan)
#         result["measurement"] = measurement

#         fusion = fuse_confidence(plane, quality, measurement)
#         result["fusion"] = fusion
#         result["measurement_discarded"] = fusion["discard_measurement"]
#         return result


from __future__ import annotations

from copy import deepcopy

from src.confidence_fusion.policy import (
    fuse_confidence,
)

from src.routing.router import (
    route_anatomy,
)


class UnifiedFetalBiometryPipeline:

    def __init__(
        self,
        backend,
        *,
        routing_enabled=False,
        allow_quality_review_route=True,
    ):

        self.backend = backend

        self.routing_enabled = bool(
            routing_enabled
        )

        self.allow_quality_review_route = bool(
            allow_quality_review_route
        )

    def _measure(
        self,
        route,
        scan,
    ):

        if route == "head_biometry":

            return (
                self.backend
                .measure_head(scan)
            )

        if route == "abdomen_biometry":

            return (
                self.backend
                .measure_abdomen(scan)
            )

        if route == "femur_biometry":

            return (
                self.backend
                .measure_femur(scan)
            )

        raise ValueError(
            f"Unsupported route: {route}"
        )

    @staticmethod
    def _neutral_quality_for_router(
        quality,
    ):

        q = deepcopy(
            quality or {}
        )

        # Preserve original prediction.
        q[
            "router_original_label"
        ] = q.get("label")

        q[
            "router_original_prediction"
        ] = q.get(
            "raw_prediction"
        )

        # Router-only neutral value.
        # This does NOT modify the quality
        # returned to UI/report/audit.
        q["label"] = "good"
        q["raw_prediction"] = "good"

        q[
            "router_policy_bypass"
        ] = True

        q[
            "router_policy_bypass_reason"
        ] = (
            "quality_model_not_"
            "policy_applicable_for_"
            "this_anatomy"
        )

        return q

    def run(
        self,
        scan,
    ):

        # ----------------
        # Plane detection
        # ----------------

        plane = (
            self.backend
            .detect_plane(scan)
        )

        anatomy = str(
            plane.get(
                "application_plane"
            )
            or "unknown"
        ).strip().lower()

        # ----------------
        # Image quality
        # ----------------

        quality_error = None

        try:

            quality_raw = (
                self.backend
                .assess_quality(scan)
            )

        except Exception as exc:

            quality_error = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            quality_raw = {

                "label":
                    "unavailable",

                "raw_prediction":
                    "unavailable",

                "score":
                    None,

                "margin":
                    None,

                "probabilities":
                    {},

                "mode":
                    "advisory",

                "model":
                    "FetalCLIP + "
                    "quality linear probe",

                "clinical_ground_truth":
                    False,

                "error":
                    quality_error,
            }

        quality = (
            self.backend
            .annotate_quality_scope(
                quality_raw,
                anatomy,
            )
        )

        # ----------------
        # Routing
        # ----------------

        if not quality.get(
            "affects_routing",
            True,
        ):

            routing_quality = (
                self
                ._neutral_quality_for_router(
                    quality
                )
            )

        else:

            routing_quality = (
                quality
            )

            if (
                quality.get("label")
                == "unavailable"
            ):

                routing_quality = (
                    deepcopy(
                        quality
                    )
                )

                routing_quality[
                    "label"
                ] = "review"

        route = route_anatomy(

            plane,

            routing_quality,

            routing_enabled=
                self.routing_enabled,

            allow_quality_review=
                self
                .allow_quality_review_route,
        )

        result = {

            "pipeline_version":
                "phase8_v3_"
                "anatomy_aware_quality",

            "plane":
                plane,

            "quality":
                quality,

            "quality_error":
                quality_error,

            "route":
                route,

            "measurement":
                None,

            "fusion":
                None,
        }

        if (
            route.get(
                "route_status"
            )
            != "routed"
        ):

            return result

        # ----------------
        # Real measurement
        # ----------------

        measurement = (
            self._measure(
                route["route"],
                scan,
            )
        )

        result[
            "measurement"
        ] = measurement

        # ----------------
        # Confidence fusion
        # ----------------

        fusion = fuse_confidence(

            plane,

            quality,

            measurement,

            quality_affects_decision=
                bool(
                    quality.get(
                        "affects_fusion",
                        True,
                    )
                ),

            quality_policy_applicable=
                bool(
                    quality.get(
                        "policy_applicable",
                        True,
                    )
                ),
        )

        result[
            "fusion"
        ] = fusion

        result[
            "measurement_discarded"
        ] = fusion[
            "discard_measurement"
        ]

        return result