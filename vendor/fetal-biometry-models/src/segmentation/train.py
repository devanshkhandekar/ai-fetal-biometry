from pathlib import Path
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.dataset import discover_hc18_pairs, make_splits, HC18SegmentationDataset
from src.segmentation.model import build_model
from src.segmentation.losses import BCEDiceLoss, dice_coeff_from_logits, iou_from_logits
from src.utils.config import load_yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for x, y, _ in tqdm(loader, desc='train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += dice_coeff_from_logits(logits.detach(), y)
        total_iou += iou_from_logits(logits.detach(), y)
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for x, y, _ in tqdm(loader, desc='val', leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item()
        total_dice += dice_coeff_from_logits(logits, y)
        total_iou += iou_from_logits(logits, y)
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True, help='Path containing training_set/ and training_set_pixel_size_and_HC.csv')
    parser.add_argument('--config', default='configs/model_config.yaml')
    parser.add_argument('--output', default='models/checkpoints/best_unet.pt')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get('seed', 42)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df = discover_hc18_pairs(args.dataset_root)
    train_df, val_df = make_splits(df, val_size=0.15, seed=int(cfg.get('seed', 42)))
    print(f'Discovered {len(df)} files | train={len(train_df)} | val={len(val_df)}')

    train_ds = HC18SegmentationDataset(train_df, image_size=cfg['image_size'], augment=True)
    val_ds = HC18SegmentationDataset(val_df, image_size=cfg['image_size'], augment=False)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg.get('num_workers', 2))
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg.get('num_workers', 2))

    model = build_model().to(device)
    loss_fn = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['learning_rate']))

    best_dice = -1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg['num_epochs']) + 1):
        tr_loss, tr_dice, tr_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_dice, va_iou = validate(model, val_loader, loss_fn, device)
        print(f'Epoch {epoch:03d} | train loss={tr_loss:.4f} dice={tr_dice:.4f} iou={tr_iou:.4f} | val loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}')

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_dice': va_dice,
                'config': cfg,
            }, output_path)
            print(f'  saved best checkpoint: {output_path} val_dice={va_dice:.4f}')


if __name__ == '__main__':
    main()
