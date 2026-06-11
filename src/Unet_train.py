"""
unet_train.py
=============
Online dataset generation + U-Net training loop.

Usage (local test):
    python unet_train.py --manifest training_data/manifest.csv \
                         --splits   training_data/splits.json  \
                         --output   results/ --epochs 5 --batch_size 4

Usage (Alliance Canada):
    sbatch train.sh
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

from Kspace_simulation import (
    load_slice, apply_motion_line_by_line,
    make_undersampling_mask, apply_undersampling,
    add_noise, kspace_to_image,
)
from Utils import compute_metrics, make_seed
from Unet_model import UNet2D

# ─────────────────────────────────────────────────────────────────────────────
# ONLINE DATASET
# ─────────────────────────────────────────────────────────────────────────────

class OnlineMRIDataset(Dataset):
    """
    Online k-space corruption dataset.

    Each __getitem__ call:
      1. Loads a clean NIfTI slice
      2. Samples random corruption parameters from the grid
      3. Applies corruption on-the-fly
      4. Returns (corrupted_kspace_2ch, clean_image) as tensors

    This means the model sees a different corrupted version
    of each slice at every epoch — better generalization than
    pre-generated data, zero storage overhead.

    Parameters
    ----------
    manifest : pd.DataFrame
        One row per (subject, slice) with columns: path, slice, TR, TE
    axis : int
        Slice orientation (0=sagittal for ds005616)
    augment : bool
        If True: sample random params each call (training)
        If False: use fixed params A=5, f=15, R=2, SNR=20 (val/test)
    """

    def __init__(self, manifest, data_root, axis=0):
        self.manifest  = manifest.reset_index(drop=True)
        self.data_root = data_root
        self.axis      = axis

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        # ── Load clean slice ───────────────────────────────────────────
        img_nib = nib.load(Path(self.data_root) / row['path'])
        volume  = img_nib.get_fdata(dtype=np.float32)
        image   = load_slice(volume, slice_idx=int(row['slice']), axis=self.axis)
        image   = image.copy()

        # # ── Pad ou crop à taille fixe ─────────────────────────────────
        TARGET_H, TARGET_W = 424, 736
        image = image[:TARGET_H, :TARGET_W]
        ph = TARGET_H - image.shape[0]
        pw = TARGET_W - image.shape[1]
        image = np.pad(image, ((0, ph), (0, pw)), mode='constant', constant_values=0)

        # ── TR et TE ──────────────────────────────────────────────────
        TR = float(row.get('TR', 2.5))
        TE = float(row.get('TE', 0.03))

        # ── Corruption parameters ─────────────────────────────────────
        A   = float(row["A"])
        f   = float(row["f"])
        R   = int(row["R"])
        snr = float(row["snr"])

        seed = make_seed(row['subject'], int(row['slice']), A, f, R, snr) % (2**31)

        # ── Apply corruption ───────────────────────────────────────────
        motion_kspace, clean_kspace, _, _ = apply_motion_line_by_line(
            image, A, TR=TR, TE=TE,
            respiratory_rate=f, method="breathmetrics", seed=seed,
        )
        mask             = make_undersampling_mask(image.shape, R=R, seed=seed)
        corrupted_kspace = apply_undersampling(motion_kspace, mask)
        corrupted_kspace = add_noise(corrupted_kspace, mask, snr, seed=seed)

        # ── Format for U-Net ───────────────────────────────────────────
        x = np.stack([
            corrupted_kspace.real.astype(np.float32),
            corrupted_kspace.imag.astype(np.float32),
        ])
        y = np.stack([
            clean_kspace.real.astype(np.float32),
            clean_kspace.imag.astype(np.float32),
        ])

        # ── Normalize ─────────────────────────────────────────────────
        scale = np.abs(clean_kspace).max() + 1e-8
        x = x / scale
        y = y / scale

        # À la fin, retourner aussi H et W originaux
        H_orig = int(row["H"])
        W_orig = int(row["W"])

        return torch.tensor(x), torch.tensor(y), H_orig, W_orig

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    # for x, y, H_orig, W_orig in loader:
    #     x, y = x.to(device), y.to(device)
    #     optimizer.zero_grad()
    #     pred = model(x)
    #     loss = nn.functional.l1_loss(pred, y)   # L1 — less blurry than MSE
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += loss.item()

    for x, y, H_orig, W_orig in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)

        # Masque padding
        mask = torch.zeros_like(y)
        for i in range(x.shape[0]):
            h, w = int(H_orig[i]), int(W_orig[i])
            mask[i, :, :h, :w] = 1.0

        loss = nn.functional.l1_loss(pred * mask, y * mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss, ssim_sum, psnr_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, y, H_orig, W_orig in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # loss = nn.functional.l1_loss(pred, y)
            
            mask_val = torch.zeros_like(y) # masked the padding for calculate the loss
            for i in range(x.shape[0]):
                h, w = int(H_orig[i]), int(W_orig[i])
                mask_val[i, :, :h, :w] = 1.0
            loss = nn.functional.l1_loss(pred * mask_val, y * mask_val)
            
            total_loss += loss.item()

        
            for i in range(pred.shape[0]):
                pred_np = pred[i].cpu().numpy()
                pred_kspace = pred_np[0] + 1j * pred_np[1]
                gt_np   = y[i].cpu().numpy()
                gt_kspace   = gt_np[0]   + 1j * gt_np[1]

                pred_img = np.abs(np.fft.ifft2(np.fft.ifftshift(pred_kspace)))
                gt_img   = np.abs(np.fft.ifft2(np.fft.ifftshift(gt_kspace)))

                norm     = gt_img.max() + 1e-8
                gt_img   /= norm
                pred_img /= norm

                # ── Masquer le padding ────────────────────────────────
                h = int(H_orig[i])
                w = int(W_orig[i])
                gt_crop   = gt_img[:h, :w]
                pred_crop = pred_img[:h, :w]

                m = compute_metrics(gt_crop, pred_crop)
                ssim_sum += m['SSIM']
                psnr_sum += m['PSNR (dB)']

    n = len(loader.dataset)
    return total_loss / len(loader), ssim_sum / n, psnr_sum / n


def train(args):
    # ── Setup ──────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = (
        torch.device("cuda")  if torch.cuda.is_available()  else
        torch.device("mps")   if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    torch.manual_seed(42) # fix the seed for reproductability
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # ── Load manifest and splits ───────────────────────────────────────
    manifest = pd.read_csv(args.manifest)
    with open(args.splits) as f:
        splits = json.load(f)

    train_df = manifest[manifest['subject'].isin(splits['train'])]
    val_df   = manifest[manifest['subject'].isin(splits['val'])]
    print(f"Train: {len(train_df)} slices | Val: {len(val_df)} slices")

    train_ds = OnlineMRIDataset(train_df, data_root=args.data_root)
    val_ds   = OnlineMRIDataset(val_df,   data_root=args.data_root)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=8, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=8, pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model     = UNet2D(in_channels=2, out_channels=2, base_channels=32).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6) # change the scheduler after bad result with 30 epochs

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Training ───────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss             = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_ssim, val_psnr = validate(model, val_loader, device)

        # scheduler.step(val_loss)
        scheduler.step()  # change the scheduler after bad result with 30 epochs
        elapsed = time.time() - t0

        row = {
            'epoch':      epoch,
            'train_loss': round(train_loss, 5),
            'val_loss':   round(val_loss,   5),
            'val_SSIM':   round(val_ssim,   4),
            'val_PSNR':   round(val_psnr,   2),
            'lr':         optimizer.param_groups[0]['lr'],
            'time_s':     round(elapsed, 1),
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_SSIM={val_ssim:.3f} | "
            f"val_PSNR={val_psnr:.1f}dB | "
            f"{elapsed:.0f}s"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'unet_best.pt')
            print(f"  → Best model saved (val_loss={val_loss:.5f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   val_loss,
            }, output_dir / f'checkpoint_epoch{epoch:03d}.pt')

    # ── Save training history ──────────────────────────────────────────
    pd.DataFrame(history).to_csv(output_dir / 'training_history.csv', index=False)
    print(f"\nTraining complete. Best val_loss = {best_val_loss:.5f}")
    print(f"Outputs saved to {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--manifest',   required=True)
    parser.add_argument('--splits',     required=True)
    parser.add_argument('--output',     default='results/')
    parser.add_argument('--epochs',     type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()
    train(args)