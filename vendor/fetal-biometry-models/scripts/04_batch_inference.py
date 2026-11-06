import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.pipeline import run_case
from src.segmentation.inference import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--pixel-spacing-mm', type=float, default=None)
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    image_paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.dcm']:
        image_paths.extend(Path(args.input_dir).glob(ext))
    image_paths = [p for p in sorted(image_paths) if '_Annotation' not in p.stem]

    rows = []
    for p in tqdm(image_paths):
        payload = run_case(p, args.checkpoint, args.output_dir, pixel_spacing_mm=args.pixel_spacing_mm, model=model)
        m = payload.get('measurements', {})
        rows.append({
            'filename': p.name,
            'decision': payload['decision']['decision'],
            'reason_codes': '|'.join(payload['decision'].get('reason_codes', [])),
            'hc_mm': m.get('hc_mm'),
            'bpd_mm': m.get('bpd_mm'),
            'ofd_mm': m.get('ofd_mm'),
            'head_area_mm2': m.get('head_area_mm2'),
            'quality_score': payload['quality']['quality_score'],
            'confidence_score': payload['confidence']['confidence_score'],
            'overlay_path': payload['artifacts'].get('overlay_path'),
        })

    out = Path(args.output_dir) / 'reports' / 'batch_summary.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Saved batch summary: {out}')


if __name__ == '__main__':
    main()
