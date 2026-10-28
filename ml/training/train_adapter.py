"""
ml/training/train_adapter.py: Step R7 Linear Adapter from Playlist Co-occurrence (Sections 11.8.1, 12.6).

Model: Linear(512, 128) on the frozen official LAION CLAP vector, L2-normalised output.
Loss: InfoNCE with temperature 0.07, in-batch negatives.
Output: ml/inference/adapter.npz (W, b), with MANIFEST entry: pairs used, epochs, final loss.
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

# Add backend to path
sys.path.insert(0, "backend")
from app.services.recommender import make_track_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_adapter")

CATALOG_FEATURES_PATH = Path("ml/data/catalog_features.jsonl")
PLAYLIST_MEMBERSHIP_PATH = Path("ml/data/playlist_membership.jsonl")
CLAP_NPY_PATH = Path("ml/data/clap.npy")
ID_MAP_PATH = Path("ml/inference/track_id_map.json")
OUT_ADAPTER_PATH = Path("ml/inference/adapter.npz")
MANIFEST_PATH = Path("ml/inference/ADAPTER_MANIFEST.json")


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
    ):
        self.pairs = pairs
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.embs = torch.tensor(embs, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        return self.embs[i], self.embs[j], self.weights[idx]


def build_training_pairs(
    catalog_meta: list[dict],
    playlist_path: Path,
    id_map_path: Path,
) -> tuple[list[tuple[int, int]], list[float], set[str], int]:
    # 1. Load id map (deezer_id, key -> catalogue index)
    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map: dict[str, int] = json.load(f)

    # 2. Group catalogue track indices by playlist
    playlist_to_tracks: dict[str, list[int]] = {}
    if playlist_path.exists():
        with open(playlist_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    pl_id = str(entry["playlist_id"])
                    did = str(entry.get("deezer_track_id", ""))
                    k = make_track_key(str(entry.get("artist", "")), str(entry.get("title", "")))

                    idx = id_map.get(k)
                    if idx is None and did:
                        idx = id_map.get(did)

                    if idx is not None and 0 <= idx < len(catalog_meta):
                        if pl_id not in playlist_to_tracks:
                            playlist_to_tracks[pl_id] = []
                        if idx not in playlist_to_tracks[pl_id]:
                            playlist_to_tracks[pl_id].append(idx)
                except Exception:
                    continue

    logger.info("Found %d playlists containing catalogue tracks.", len(playlist_to_tracks))

    # Drop playlists with < 2 or > 2,000 tracks (Section 11.8.1)
    valid_playlists = {
        pl: trks for pl, trks in playlist_to_tracks.items()
        if 2 <= len(trks) <= 2000
    }
    logger.info("%d valid playlists with 2 <= size <= 2000.", len(valid_playlists))

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
        max_samples = min(len(trks) * (len(trks) - 1) // 2, 100)
        n_t = len(trks)
        sampled = 0
        for _ in range(max_samples * 2):
            if sampled >= max_samples:
                break
            i, j = random.sample(trks, 2)
            train_pairs.append((i, j))
            train_weights.append(w)
            sampled += 1

    logger.info("Generated %d training co-occurrence pairs from %d playlists.", len(train_pairs), len(train_pl_set))
    return train_pairs, train_weights, val_pl_set, len(valid_playlists)


def info_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    sim = torch.mm(q, k.t()) / temperature  # (B, B)
    labels = torch.arange(q.shape[0], device=q.device)
    loss = F.cross_entropy(sim, labels, reduction="none")
    weighted_loss = (loss * weights).mean()
    return weighted_loss


def train(epochs: int = 30, batch_size: int = 1024, lr: float = 1e-3) -> dict:
    logger.info("=== Starting Step R7: Adapter Training (Section 11.8.1) ===")
    if not CATALOG_FEATURES_PATH.exists() or not CLAP_NPY_PATH.exists():
        raise FileNotFoundError(f"Missing {CATALOG_FEATURES_PATH} or {CLAP_NPY_PATH}")

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
    pairs, weights, val_pls, total_valid_pls = build_training_pairs(metadata, PLAYLIST_MEMBERSHIP_PATH, ID_MAP_PATH)
    if not pairs:
        raise RuntimeError("No co-occurrence pairs available! Check ID mapping.")

    dataset = CoOccurrenceDataset(pairs, weights, embs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)
    model = LinearAdapter(512, 128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for q_emb, k_emb, w in dataloader:
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
            logger.info("Epoch %02d/%02d | Train InfoNCE Loss: %.4f", epoch, epochs, avg_loss)

    assert best_weights is not None, "Failed to train adapter weights"

    OUT_ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_ADAPTER_PATH, W=best_weights[0], b=best_weights[1])
    logger.info("Saved adapter weights W=%s, b=%s to %s", best_weights[0].shape, best_weights[1].shape, OUT_ADAPTER_PATH)

    manifest_data = {
        "pairs_used": len(pairs),
        "playlists_used": total_valid_pls,
        "epochs": epochs,
        "final_loss": round(best_loss, 4),
        "device": str(device),
        "adapter_shape": list(best_weights[0].shape),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Wrote adapter manifest to %s", MANIFEST_PATH)

    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echo Step 11.8.1 Adapter Training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
