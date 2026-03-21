"""
Visualise trained CnnMusicEncoder embeddings with t-SNE and a confusion-style
genre similarity heatmap.

Reads pre-computed embeddings.npy + track_metadata.json from build_index.py,
so run this AFTER the full pipeline (train → export_onnx → build_index).

Usage:
    python ml/inference/visualize.py \
        --embeddings  ml/inference/embeddings.npy \
        --metadata    ml/inference/track_metadata.json \
        --output-dir  ml/inference/plots \
        --n-samples   3000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless — saves PNG files instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_data(
    embeddings_path: str,
    metadata_path: str,
    n_samples: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    emb = np.load(embeddings_path).astype(np.float32)   # (N, D)

    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)

    assert len(emb) == len(meta), "embeddings / metadata length mismatch"

    genres = [m.get("genre") or (m.get("vibes") or ["unknown"])[0] for m in meta]
    titles = [f"{m.get('title','?')} — {m.get('artist','?')}" for m in meta]

    if n_samples < len(emb):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(emb), size=n_samples, replace=False)
        emb    = emb[idx]
        genres = [genres[i] for i in idx]
        titles = [titles[i] for i in idx]

    logger.info("Loaded %d embeddings, %d unique genres", len(emb), len(set(genres)))
    return emb, genres, titles


def plot_tsne(
    emb: np.ndarray,
    genres: list[str],
    output_path: Path,
) -> None:
    logger.info("Running t-SNE on %d × %d embeddings…", *emb.shape)
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate="auto",
        init="pca",
        n_iter=1000,
        random_state=0,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(emb)    # (N, 2)

    le = LabelEncoder().fit(genres)
    labels = le.transform(genres)
    n_classes = len(le.classes_)
    cmap = cm.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    for cls_idx, genre in enumerate(le.classes_):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=6,
            alpha=0.65,
            color=cmap(cls_idx),
            label=genre,
            linewidths=0,
        )

    ax.legend(
        loc="upper right",
        fontsize=7,
        markerscale=3,
        framealpha=0.25,
        labelcolor="white",
        facecolor="#1a1a1a",
        edgecolor="#333",
    )
    ax.set_title(
        "CnnMusicEncoder — t-SNE of genre embeddings",
        color="white", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("t-SNE plot saved → %s", output_path)


def plot_genre_similarity(
    emb: np.ndarray,
    genres: list[str],
    output_path: Path,
) -> None:
    """Mean cosine similarity between every pair of genres — like a confusion matrix."""
    genre_arr   = np.array(genres)
    unique      = sorted(set(genres))
    n           = len(unique)
    sim_matrix  = np.zeros((n, n), dtype=np.float32)

    # Embeddings are already L2-normalised from build_index; dot product = cosine sim
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb_n = emb / norms

    for i, g_i in enumerate(unique):
        vecs_i = emb_n[genre_arr == g_i]
        mean_i = vecs_i.mean(axis=0)
        for j, g_j in enumerate(unique):
            vecs_j = emb_n[genre_arr == g_j]
            mean_j = vecs_j.mean(axis=0)
            sim_matrix[i, j] = float(np.dot(mean_i, mean_j))

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(sim_matrix, cmap="magma", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("cosine similarity", color="white", fontsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(unique, rotation=45, ha="right", fontsize=8, color="white")
    ax.set_yticklabels(unique, fontsize=8, color="white")
    ax.set_title(
        "Mean inter-genre cosine similarity",
        color="white", fontsize=12, pad=10,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Genre similarity heatmap saved → %s", output_path)


def plot_training_curves(output_dir: Path, checkpoints_dir: Path) -> None:
    """Read TensorBoard event files and plot loss curves as a static PNG."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("tensorboard not installed — skipping loss curve plot")
        return

    tb_dir = checkpoints_dir / "tb_logs"
    if not tb_dir.exists():
        logger.warning("TensorBoard log dir not found at %s", tb_dir)
        return

    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    if not tags:
        logger.warning("No scalar tags found in TensorBoard logs")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d0d")

    for ax, (tag_train, tag_val, title) in zip(
        axes,
        [
            ("loss/train",          "loss/val",          "Triplet Loss"),
            ("frac_positive/train", "frac_positive/val", "Fraction Positive Triplets"),
        ],
    ):
        ax.set_facecolor("#111")
        for tag, label, color in [
            (tag_train, "train", "#8B5CF6"),
            (tag_val,   "val",   "#2DD4BF"),
        ]:
            if tag in tags:
                events = ea.Scalars(tag)
                steps  = [e.step for e in events]
                values = [e.value for e in events]
                ax.plot(steps, values, label=label, color=color, linewidth=1.8)

        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("epoch", color="#888", fontsize=9)
        ax.tick_params(colors="#666")
        ax.legend(fontsize=9, labelcolor="white", facecolor="#1a1a1a",
                  framealpha=0.4, edgecolor="#333")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    fig.tight_layout()
    out = output_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Training curves saved → %s", out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings",      default="ml/inference/embeddings.npy")
    p.add_argument("--metadata",        default="ml/inference/track_metadata.json")
    p.add_argument("--output-dir",      default="ml/inference/plots")
    p.add_argument("--checkpoints-dir", default="ml/checkpoints",
                   help="Checkpoint dir containing tb_logs/ for training curves")
    p.add_argument("--n-samples",       type=int, default=3000,
                   help="Max samples for t-SNE (subsampled if larger)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb, genres, titles = load_data(args.embeddings, args.metadata, args.n_samples)

    plot_tsne(emb, genres, output_dir / "tsne.png")
    plot_genre_similarity(emb, genres, output_dir / "genre_similarity.png")
    plot_training_curves(output_dir, Path(args.checkpoints_dir))

    logger.info(
        "All plots written to %s\n"
        "  tsne.png              — 2-D cluster map of embeddings\n"
        "  genre_similarity.png  — mean cosine sim between genre centroids\n"
        "  training_curves.png   — loss + fraction-positive over epochs",
        output_dir,
    )


if __name__ == "__main__":
    main()
