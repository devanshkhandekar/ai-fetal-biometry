from __future__ import annotations

import argparse
import json
from src.pipeline_devansh import run_case_devansh


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Core Biometry MVP inference with pretrained static fetal-head model.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", default="phase4a", choices=["phase0", "phase4a"])
    parser.add_argument("--pixel-spacing-mm", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    payload = run_case_devansh(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        model_variant=args.variant,
        output_dir=args.output_dir,
        pixel_spacing_mm=args.pixel_spacing_mm,
        threshold=args.threshold,
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
