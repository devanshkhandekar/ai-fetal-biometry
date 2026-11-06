import argparse
from pathlib import Path
import cv2
import pandas as pd

from src.segmentation.dataset import discover_hc18_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--out', default='outputs/logs/hc18_audit.csv')
    args = parser.parse_args()

    df = discover_hc18_pairs(args.dataset_root)
    rows = []
    for _, r in df.iterrows():
        img = cv2.imread(r['image_path'], cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2] if img is not None else (None, None)
        row = r.to_dict()
        row.update({
            'height': h,
            'width': w,
            'readable': img is not None,
            'usable_for_training': bool(r['has_annotation'] and img is not None),
        })
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(out_df[['filename', 'has_annotation', 'readable', 'height', 'width', 'usable_for_training']].head())
    print(f'Rows={len(out_df)} | usable={out_df.usable_for_training.sum()} | saved={out_path}')


if __name__ == '__main__':
    main()
