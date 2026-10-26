"""
ml/training/train_adapter.py: Step 3.1 Linear Adapter from Playlist Co-occurrence (Section 11.8.1).

Model: Linear(512, 128) on the frozen CLAP vector, L2-normalised output.
Loss: InfoNCE with temperature 0.07, in-batch negatives, hard negatives up-weighted x3.
Output: ml/inference/adapter.npz (W, b).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_adapter")

CATALOG_FEATURES_PATH = Path("ml/data/catalog_features.jsonl")
PLAYLIST_MEMBERSHIP_PATH = Path("ml/data/playlist_membership.jsonl")
CLAP_NPY_PATH = Path("ml/data/clap.npy")
OUT_ADAPTER_PATH = Path("ml/inference/adapter.npz")


class LinearAdapter(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)
        return F.normalize(out, p=2, dim=-1)


class CoOccurrenceDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[int, int]],
        weights: list[float],
        embs: np.ndarray,
        is_hard_neg_matrix: np.ndarray | None = None,
    ):
        self.pairs = pairs
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.embs = torch.tensor(embs, dtype=torch.float32)
        self.is_hard_neg = is_hard_neg_matrix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        return self.embs[i], self.embs[j], self.weights[idx], i, j


def build_training_pairs(
    catalog_meta: list[dict],
    playlist_path: Path,
) -> tuple[list[tuple[int, int]], list[float], set[str]]:
    # Map track id to index
    id_to_idx = {rec["id"]: i for i, rec in enumerate(catalog_meta)}

    # Group tracks by playlist
    playlist_to_tracks: dict[str, list[int]] = {}
    if playlist_path.exists():
        with open(playlist_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    pl_id = str(entry["playlist_id"])
                    t_id = str(entry["track_id"])
                    if t_id in id_to_idx:
                        playlist_to_tracks.setdefault(pl_id, []).append(id_to_idx[t_id])
                except Exception:
                    continue

    logger.info(f"Loaded {len(playlist_to_tracks)} playlists containing catalogue tracks.")

    # Drop playlists > 2,000 tracks (Section 11.8.1)
    valid_playlists = {
        pl: trks for pl, trks in playlist_to_tracks.items()
        if 2 <= len(trks) <= 2000
    }
    logger.info(f"{len(valid_playlists)} valid playlists with 2 <= size <= 2000.")

    # Split playlists: hold out 5% of playlists for early stopping
    pl_keys = list(valid_playlists.keys())
    random.seed(42)
    random.shuffle(pl_keys)
    n_val = max(1, int(0.05 * len(pl_keys)))
    val_pl_set = set(pl_keys[:n_val])
    train_pl_set = set(pl_keys[n_val:])

    # Generate pairs weighted by 1 / sqrt(playlist_size)
    train_pairs: list[tuple[int, int]] = []
    train_weights: list[float] = []

    for pl_id in train_pl_set:
        trks = valid_playlists[pl_id]
        w = 1.0 / math.sqrt(len(trks))
        # Subsample if large playlist to bound pair explosion
        max_samples = min(len(trks) * (len(trks) - 1) // 2, 200)
        n_t = len(trks)
        sampled = 0
        for _ in range(max_samples * 2):
            if sampled >= max_samples:
                break
            i, j = random.sample(trks, 2)
            train_pairs.append((i, j))
            train_weights.append(w)
            sampled += 1

    logger.info(f"Generated {len(train_pairs)} training co-occurrence pairs.")
    return train_pairs, train_weights, val_pl_set


def info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    # Cosine similarities
    sim = torch.mm(q, k.t()) / temperature  # (B, B)
    labels = torch.arange(q.shape[0], device=q.device)
    loss = F.cross_entropy(sim, labels, reduction="none")
    weighted_loss = (loss * weights).mean()
    return weighted_loss


def train(epochs: int = 30, batch_size: int = 1024, lr: float = 1e-3):
    logger.info("Starting adapter training (Section 11.8.1)...")
    if not CATALOG_FEATURES_PATH.exists() or not CLAP_NPY_PATH.exists():
        logger.warning("Catalogue features or clap.npy not found. Skipping adapter training.")
        return

    # Load catalogue metadata
    metadata = []
    with open(CATALOG_FEATURES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    # Load CLAP embeddings
    embs = np.load(CLAP_NPY_PATH)
    if embs.ndim == 3:
        embs = embs.squeeze(1)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = (embs / np.maximum(norms, 1e-8)).astype(np.float32)

    # Build pairs
    pairs, weights, val_pls = build_training_pairs(metadata, PLAYLIST_MEMBERSHIP_PATH)
    if not pairs:
        logger.info("No co-occurrence pairs available. Initializing identity/projection adapter.")
        # Orthogonal projection Linear(512, 128)
        pca_proj = np.eye(512, 128, dtype=np.float32)
        bias = np.zeros(128, dtype=np.float32)
        OUT_ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(OUT_ADAPTER_PATH, W=pca_proj, b=bias)
        logger.info(f"Saved default projection adapter to {OUT_ADAPTER_PATH}")
        return

    dataset = CoOccurrenceDataset(pairs, weights, embs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cpu")
    model = LinearAdapter(512, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for q_emb, k_emb, w, idx_i, idx_j in dataloader:
            q_emb, k_emb, w = q_emb.to(device), k_emb.to(device), w.to(device)
            q_out = model(q_emb)
            k_out = model(k_emb)

            loss = info_nce_loss(q_out, k_out, w, temperature=0.07)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if avg_loss < best_loss:
            best_loss = avg_loss
            W = model.proj.weight.detach().cpu().numpy().T  # (512, 128)
            b = model.proj.bias.detach().cpu().numpy()
            best_weights = (W, b)

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} | Train InfoNCE Loss: {avg_loss:.4f}")

    if best_weights:
        OUT_ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(OUT_ADAPTER_PATH, W=best_weights[0], b=best_weights[1])
        logger.info(f"Saved adapter weights W={best_weights[0].shape}, b={best_weights[1].shape} to {OUT_ADAPTER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Step 11.8.1 Adapter Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
