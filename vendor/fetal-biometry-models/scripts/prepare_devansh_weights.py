"""Prepare pretrained static fetal-head model weights for the MVP.

The dashboard expects one or both checkpoint files under models/checkpoints/:
    - 4a_best_pruned_ft_v10.pth  # recommended default
    - phase0_model.pth           # optional high-capacity/static variant

Usage:
    python scripts/prepare_devansh_weights.py --source /path/to/4a_best_pruned_ft_v10.pth --variant phase4a
    python scripts/prepare_devansh_weights.py --source /path/to/phase0_model.pth --variant phase0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FILENAMES = {
    "phase0": "phase0_model.pth",
    "phase4a": "4a_best_pruned_ft_v10.pth",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy pretrained fetal-head model weights into the MVP checkpoint folder.")
    parser.add_argument("--variant", default="phase4a", choices=sorted(FILENAMES))
    parser.add_argument("--source", required=True, help="Path to the checkpoint already downloaded locally.")
    parser.add_argument("--out-dir", default="models/checkpoints")
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / FILENAMES[args.variant]
    shutil.copy2(src, dst)
    print(f"Prepared checkpoint: {dst}")


if __name__ == "__main__":
    main()
