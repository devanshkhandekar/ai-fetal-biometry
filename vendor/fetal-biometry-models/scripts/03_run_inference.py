import argparse
import json
from src.pipeline import run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--pixel-spacing-mm', type=float, default=None)
    args = parser.parse_args()

    payload = run_case(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        pixel_spacing_mm=args.pixel_spacing_mm,
    )
    print(json.dumps({
        'decision': payload['decision'],
        'measurements': payload.get('measurements'),
        'quality': payload.get('quality', {}).get('quality_label'),
        'confidence': payload.get('confidence', {}).get('confidence_band'),
        'artifacts': payload.get('artifacts'),
    }, indent=2))


if __name__ == '__main__':
    main()
