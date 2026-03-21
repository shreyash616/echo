"""
Echo — CnnMusicEncoder Training Script
GPU: RTX 3080 (CUDA 12.x, 10 GB VRAM)

Trains a CNN on 128-bin mel spectrograms from FMA audio files using online
hard-mining TripletLoss with FMA genre labels as class supervision.

Usage:
    python ml/training/train.py \
        --spec-dir   ml/data/spectrograms \
        --tracks-csv ml/data/fma_metadata/tracks.csv \
        --output     ml/checkpoints \
        --epochs     60 \
        --batch      64

After training, export with ml/inference/export_onnx.py, then build the FAISS
index with ml/inference/build_index.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.models.music_encoder import CnnMusicEncoder
from ml.training.dataset import FMASpectrogramDataset
from ml.training.losses import TripletLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CnnMusicEncoder on FMA mel spectrograms")
    p.add_argument("--spec-dir",   required=True, help="Directory of .npy spectrogram files")
    p.add_argument("--tracks-csv", required=True, help="FMA tracks.csv (multi-level header)")
    p.add_argument("--output",     default="ml/checkpoints", help="Checkpoint output directory")
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch",      type=int,   default=64,
                   help="Batch size (RTX 3080: 64-128 for spectrograms)")
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--margin",     type=float, default=0.2, help="Triplet loss margin")
    p.add_argument("--embedding-dim", type=int, default=512)
    p.add_argument("--crop-frames",   type=int, default=256,
                   help="Spectrogram time-axis crop width (~3 s)")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--val-split",     type=float, default=0.1)
    p.add_argument("--workers",       type=int, default=4)
    p.add_argument("--resume",        type=str, default=None,
                   help="Resume training from checkpoint path")
    return p.parse_args()


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: CnnMusicEncoder,
    loader: DataLoader,
    criterion: TripletLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    step: int,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_frac = 0.0

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)   # (B, 1, 128, T)
        labels = labels.to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type):  # fp16 — halves VRAM, 2× on RTX 3080
            embeddings = model(specs)                  # (B, embedding_dim)
            loss, frac_pos = criterion(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_frac += frac_pos.item()
        step += 1

    n = len(loader)
    return total_loss / n, total_frac / n, step


@torch.no_grad()
def val_epoch(
    model: CnnMusicEncoder,
    loader: DataLoader,
    criterion: TripletLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_frac = 0.0

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        embeddings = model(specs)
        loss, frac_pos = criterion(embeddings, labels)
        total_loss += loss.item()
        total_frac += frac_pos.item()

    n = len(loader)
    return total_loss / n, total_frac / n


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s  (%.1f GB VRAM)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "tb_logs"))

    # Dataset — two instances so train/val have independent augment flags
    logger.info("Loading dataset — spec_dir=%s  tracks_csv=%s", args.spec_dir, args.tracks_csv)
    train_dataset = FMASpectrogramDataset(
        spec_dir=args.spec_dir,
        tracks_csv=args.tracks_csv,
        crop_frames=args.crop_frames,
        augment=True,
    )
    val_dataset = FMASpectrogramDataset(
        spec_dir=args.spec_dir,
        tracks_csv=args.tracks_csv,
        crop_frames=args.crop_frames,
        augment=False,
    )
    logger.info("Dataset: %d samples, %d classes", len(train_dataset), train_dataset.num_classes)

    val_size   = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size

    # Shuffle indices with a fixed seed, then slice into train/val Subsets
    generator = torch.Generator().manual_seed(42)
    perm      = torch.randperm(len(train_dataset), generator=generator).tolist()
    train_ds  = Subset(train_dataset, perm[:train_size])
    val_ds    = Subset(val_dataset,   perm[train_size:])

    # Save crop_frames so inference knows the expected input width
    meta = {"crop_frames": args.crop_frames, "embedding_dim": args.embedding_dim}
    with open(output_dir / "train_meta.json", "w") as f:
        json.dump(meta, f)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Model
    model = CnnMusicEncoder(embedding_dim=args.embedding_dim).to(device)
    logger.info(
        "CnnMusicEncoder — %s parameters",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    criterion = TripletLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(device.type)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler    = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch   = 0
    best_val_loss = float("inf")
    global_step   = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        global_step   = ckpt.get("global_step", 0)
        logger.info("Resumed from epoch %d", start_epoch)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_frac, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, global_step
        )
        val_loss, val_frac = val_epoch(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d | train_loss=%.4f frac=%.2f | val_loss=%.4f frac=%.2f | %.1fs",
            epoch + 1, args.epochs,
            train_loss, train_frac,
            val_loss,   val_frac,
            elapsed,
        )

        writer.add_scalars("loss",          {"train": train_loss, "val": val_loss},   epoch)
        writer.add_scalars("frac_positive", {"train": train_frac, "val": val_frac},   epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / "best_model.pt"
            torch.save({
                "epoch":         epoch,
                "model":         model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scaler":        scaler.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "global_step":   global_step,
                "args":          vars(args),
            }, best_path)
            logger.info("  ↑ New best val_loss=%.4f — saved to %s", best_val_loss, best_path)

        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"ckpt_epoch_{epoch+1:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt_path)

    writer.close()
    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)
    logger.info(
        "Next steps:\n"
        "  1. python ml/inference/export_onnx.py --checkpoint %s/best_model.pt\n"
        "  2. python ml/inference/build_index.py --spec-dir %s --tracks-csv %s",
        args.output, args.spec_dir, args.tracks_csv,
    )


if __name__ == "__main__":
    main()
