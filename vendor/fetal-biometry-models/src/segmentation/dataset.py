from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src.preprocessing.preprocess import normalize_grayscale


def annotation_to_filled_mask(annotation: np.ndarray) -> np.ndarray:
    """HC18 annotation PNG is usually an ellipse outline. Convert it to a filled binary mask."""
    gray = annotation if annotation.ndim == 2 else cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
    binary = (gray > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 255, thickness=-1)
    return (filled > 0).astype(np.float32)


def discover_hc18_pairs(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    train_dir = root / 'training_set'
    if not train_dir.exists():
        # allow user to pass training_set directly
        train_dir = root

    image_paths = sorted([p for p in train_dir.glob('*.png') if '_Annotation' not in p.stem])
    rows = []
    for img_path in image_paths:
        ann_path = img_path.with_name(img_path.stem + '_Annotation.png')
        rows.append({
            'filename': img_path.name,
            'image_path': str(img_path),
            'annotation_path': str(ann_path) if ann_path.exists() else None,
            'has_annotation': ann_path.exists(),
        })

    df = pd.DataFrame(rows)

    csv_path = root / 'training_set_pixel_size_and_HC.csv'
    if csv_path.exists() and len(df) > 0:
        meta = pd.read_csv(csv_path)
        # Normalize likely column names.
        meta.columns = [c.strip() for c in meta.columns]
        filename_col = [c for c in meta.columns if 'filename' in c.lower() or 'image' in c.lower()]
        if filename_col:
            meta = meta.rename(columns={filename_col[0]: 'filename'})
            df = df.merge(meta, on='filename', how='left')

    return df


def make_splits(df: pd.DataFrame, val_size=0.15, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable = df[df['has_annotation']].reset_index(drop=True)
    train_df, val_df = train_test_split(usable, test_size=val_size, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class HC18SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size=256, augment=False):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _augment(self, image, mask):
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))
            mask = np.ascontiguousarray(np.fliplr(mask))
        if np.random.rand() < 0.3:
            angle = float(np.random.uniform(-12, 12))
            h, w = image.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if np.random.rand() < 0.3:
            alpha = float(np.random.uniform(0.85, 1.15))
            beta = float(np.random.uniform(-0.05, 0.05))
            image = np.clip(image * alpha + beta, 0, 1)
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)
        return image, mask

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
        ann = cv2.imread(row['annotation_path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {row['image_path']}")
        if ann is None:
            raise ValueError(f"Could not read annotation: {row['annotation_path']}")

        img = normalize_grayscale(img)
        mask = annotation_to_filled_mask(ann)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        x = torch.from_numpy(img[None, ...]).float()
        y = torch.from_numpy(mask[None, ...]).float()
        return x, y, row['filename']
