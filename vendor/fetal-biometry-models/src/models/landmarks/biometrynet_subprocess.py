from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "biometrynet_bridge.json"


class BiometryNetSubprocess:
    """
    Runs BiometryNet/HRNet in its dedicated Conda environment.

    This avoids mixing the older HRNet stack with FetalCLIP/Phase-0 runtime.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG):
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = ROOT / self.config_path
        self.cfg = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.python = Path(self.cfg["hrnet_python"])
        self.repo_root = ROOT / self.cfg["repo_root"]
        self.script = ROOT / self.cfg["single_infer_script"]
        self.timeout = int(self.cfg.get("timeout_seconds", 180))
        self.allow_full_image_fallback = bool(
            self.cfg.get("allow_full_image_fallback", True)
        )
        self.output_dir = ROOT / self.cfg.get(
            "output_dir", "outputs/phase8_landmarks"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for p in (self.python, self.repo_root, self.script):
            if not p.exists():
                raise FileNotFoundError(p)

    @staticmethod
    def _roi_args(roi_xyxy):
        if roi_xyxy is None:
            return []
        if len(roi_xyxy) != 4:
            raise ValueError("roi_xyxy must be (x1,y1,x2,y2)")
        return ["--roi", *[str(float(v)) for v in roi_xyxy]]

    def _spec(self, anatomy: str, metric: str):
        anatomy_cfg = self.cfg[anatomy]
        metric_cfg = anatomy_cfg[metric]
        return anatomy_cfg["metadata_csv"], metric_cfg["config"], metric_cfg["checkpoint"]

    def predict_metric(
        self,
        *,
        image_path: str | Path,
        anatomy: str,
        metric: str,
        pixel_spacing_mm: Optional[float] = None,
        roi_xyxy: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        image_path = Path(image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        metadata_csv, cfg_path, checkpoint = self._spec(anatomy, metric)

        cmd = [
            str(self.python),
            str(self.script),
            "--repo-root", str(self.repo_root),
            "--cfg", str(cfg_path),
            "--checkpoint", str(checkpoint),
            "--image", str(image_path),
            "--metadata-csv", str(metadata_csv),
            "--metric", metric,
            "--anatomy", anatomy,
            "--device", "cuda",
        ]
        if pixel_spacing_mm is not None:
            cmd += ["--pixel-spacing-mm", str(float(pixel_spacing_mm))]
        cmd += self._roi_args(roi_xyxy)
        if self.allow_full_image_fallback:
            cmd.append("--allow-full-image-fallback")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(
            self.cfg.get("cuda_visible_devices", "0")
        )

        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                "BiometryNet subprocess failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

        lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        if not lines:
            raise RuntimeError(
                "BiometryNet subprocess returned no output.\n"
                f"STDERR:\n{proc.stderr}"
            )

        try:
            result = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Could not parse BiometryNet JSON from final stdout line.\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            ) from exc

        result["stderr"] = proc.stderr.strip()
        result["overlay_path"] = self._make_overlay(
            image_path=image_path,
            metric=metric,
            landmarks=result.get("landmarks"),
            status=result.get("status", "review"),
        )
        return result

    def _make_overlay(self, *, image_path, metric, landmarks, status):
        if not landmarks or len(landmarks) != 2:
            return None

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        p1 = tuple(float(x) for x in landmarks[0])
        p2 = tuple(float(x) for x in landmarks[1])

        # Do not hard-code colors; use PIL defaults appropriate to RGB.
        # White line/markers are intentional neutral overlays.
        draw.line([p1, p2], fill="white", width=3)
        r = 5
        for x, y in (p1, p2):
            draw.ellipse((x-r, y-r, x+r, y+r), outline="white", width=2)

        name = f"{Path(image_path).stem}_{metric}_{status}.png"
        out = self.output_dir / name
        image.save(out)
        return str(out)

    def measure_abdomen(
        self,
        image_path: str | Path,
        *,
        pixel_spacing_mm: Optional[float] = None,
        roi_xyxy=None,
    ) -> Dict[str, Any]:
        tad = self.predict_metric(
            image_path=image_path,
            anatomy="abdomen",
            metric="TAD",
            pixel_spacing_mm=pixel_spacing_mm,
            roi_xyxy=roi_xyxy,
        )
        apad = self.predict_metric(
            image_path=image_path,
            anatomy="abdomen",
            metric="APAD",
            pixel_spacing_mm=pixel_spacing_mm,
            roi_xyxy=roi_xyxy,
        )

        statuses = {tad["status"], apad["status"]}
        if "invalid" in statuses:
            status = "invalid"
        elif "review" in statuses:
            status = "review"
        else:
            status = "valid"

        tad_mm = tad.get("distance_mm")
        apad_mm = apad.get("distance_mm")
        ac_mm = None
        if tad_mm is not None and apad_mm is not None:
            ac_mm = (math.pi / 2.0) * (float(tad_mm) + float(apad_mm))

        reasons = list(dict.fromkeys(
            [f"TAD:{x}" for x in tad.get("reasons", [])]
            + [f"APAD:{x}" for x in apad.get("reasons", [])]
        ))

        return {
            "status": status,
            "confidence": None,
            "tad_mm": tad_mm,
            "apad_mm": apad_mm,
            "ac_mm": ac_mm,
            "tad_px": tad.get("distance_px"),
            "apad_px": apad.get("distance_px"),
            "tad_landmarks": tad.get("landmarks"),
            "apad_landmarks": apad.get("landmarks"),
            "pixel_spacing_mm": tad.get("pixel_spacing_mm") or apad.get("pixel_spacing_mm"),
            "roi_source": tad.get("roi_source"),
            "benchmark_preprocessing": bool(
                tad.get("benchmark_preprocessing")
                and apad.get("benchmark_preprocessing")
            ),
            "reasons": reasons,
            "artifacts": {
                "tad_overlay": tad.get("overlay_path"),
                "apad_overlay": apad.get("overlay_path"),
            },
            "raw": {"TAD": tad, "APAD": apad},
        }

    def measure_femur(
        self,
        image_path: str | Path,
        *,
        pixel_spacing_mm: Optional[float] = None,
        roi_xyxy=None,
    ) -> Dict[str, Any]:
        fl = self.predict_metric(
            image_path=image_path,
            anatomy="femur",
            metric="FL",
            pixel_spacing_mm=pixel_spacing_mm,
            roi_xyxy=roi_xyxy,
        )
        return {
            "status": fl["status"],
            "confidence": None,
            "fl_mm": fl.get("distance_mm"),
            "fl_px": fl.get("distance_px"),
            "fl_landmarks": fl.get("landmarks"),
            "pixel_spacing_mm": fl.get("pixel_spacing_mm"),
            "roi_source": fl.get("roi_source"),
            "benchmark_preprocessing": bool(fl.get("benchmark_preprocessing")),
            "reasons": fl.get("reasons", []),
            "artifacts": {"fl_overlay": fl.get("overlay_path")},
            "raw": fl,
        }
