"""
Build the FAISS similarity index from FMA mel spectrograms using the trained
CnnMusicEncoder exported to ONNX.

Pipeline:
  1. Load all .npy spectrogram files from spec_dir (output of preprocess.py).
  2. Match each file's track_id to FMA tracks.csv for metadata.
  3. Batch-encode spectrograms with the ONNX model → 512-d embeddings.
  4. L2-normalise and insert into a FAISS IndexFlatIP (cosine similarity).
  5. Write music_index.faiss + track_metadata.json.

Usage:
    python ml/inference/build_index.py \
        --spec-dir    ml/data/spectrograms \
        --tracks-csv  ml/data/fma_metadata/tracks.csv \
        --onnx-model  ml/inference/music_encoder.onnx \
        --output-dir  ml/inference \
        --batch        64
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import onnxruntime as ort
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

CROP_FRAMES = 256   # centre-crop width used at inference — match training


def centre_crop(mel: np.ndarray, crop_frames: int = CROP_FRAMES) -> np.ndarray:
    """Return a (128, crop_frames) centre-crop of the spectrogram."""
    T = mel.shape[1]
    if T <= crop_frames:
        pad = crop_frames - T
        return np.pad(mel, ((0, 0), (0, pad)), mode="constant")
    start = (T - crop_frames) // 2
    return mel[:, start: start + crop_frames]


def load_tracks(tracks_csv: str) -> pd.DataFrame:
    """Parse FMA multi-level tracks.csv and return a flat DataFrame."""
    df = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)
    df.columns = ["_".join(col).strip() for col in df.columns]
    df.index.name = "track_id"
    df = df.reset_index()
    df["track_id"] = df["track_id"].astype(int)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec-dir",    required=True,
                   help="Directory of .npy spectrogram files (from preprocess.py)")
    p.add_argument("--tracks-csv",  required=True,
                   help="FMA tracks.csv (multi-level header)")
    p.add_argument("--onnx-model",  default="ml/inference/music_encoder.onnx",
                   help="Exported ONNX model path")
    p.add_argument("--output-dir",  default="ml/inference")
    p.add_argument("--batch",       type=int, default=64, help="Encode batch size")
    p.add_argument("--crop-frames", type=int, default=CROP_FRAMES)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load track metadata
    # ------------------------------------------------------------------ #
    logger.info("Loading track metadata from %s", args.tracks_csv)
    tracks_df = load_tracks(args.tracks_csv)
    tracks_by_id: dict[int, dict] = {
        int(row["track_id"]): row.to_dict()
        for _, row in tracks_df.iterrows()
    }

    # ------------------------------------------------------------------ #
    # Collect spectrogram files that have metadata
    # ------------------------------------------------------------------ #
    spec_dir = Path(args.spec_dir)
    npy_files: list[tuple[int, Path]] = []
    for npy_path in sorted(spec_dir.glob("*.npy")):
        try:
            tid = int(npy_path.stem)
        except ValueError:
            continue
        if tid in tracks_by_id:
            npy_files.append((tid, npy_path))

    logger.info(
        "Found %d spectrogram files with metadata (out of %d total .npy files)",
        len(npy_files),
        len(list(spec_dir.glob("*.npy"))),
    )
    if not npy_files:
        raise RuntimeError(
            "No matching spectrogram files found — check --spec-dir and --tracks-csv"
        )

    # ------------------------------------------------------------------ #
    # Load ONNX model
    # ------------------------------------------------------------------ #
    logger.info("Loading ONNX model from %s", args.onnx_model)
    sess = ort.InferenceSession(
        args.onnx_model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name

    # ------------------------------------------------------------------ #
    # Batch-encode spectrograms
    # ------------------------------------------------------------------ #
    embeddings: list[np.ndarray] = []
    metadata:   list[dict]       = []
    batch_specs: list[np.ndarray] = []
    batch_metas: list[dict]       = []

    def flush_batch() -> None:
        if not batch_specs:
            return
        arr = np.stack(batch_specs, axis=0).astype(np.float32)
        arr = arr[:, np.newaxis, :, :]   # (B, 1, 128, T)
        out = sess.run(None, {input_name: arr})
        embeddings.extend(list(out[0]))
        metadata.extend(batch_metas)
        batch_specs.clear()
        batch_metas.clear()

    for tid, npy_path in tqdm(npy_files, desc="Encoding"):
        mel   = np.load(npy_path)                          # (128, T)
        patch = centre_crop(mel, args.crop_frames)         # (128, crop_frames)
        row   = tracks_by_id[tid]
        genre = str(row.get("track_genre_top", ""))

        batch_specs.append(patch)
        batch_metas.append({
            "id":          f"fma_{tid}",
            "title":       str(row.get("track_title", "")),
            "artist":      str(row.get("artist_name", "")),
            "album":       str(row.get("album_title", "")),
            "albumArtUrl": "",
            "previewUrl":  None,
            "durationMs":  int(float(row.get("track_duration", 0)) * 1000),
            "genre":       genre,
            "bpm":         0,
            "key":         "—",
            "energy":      0.5,
            "valence":     0.5,
            "vibes":       [genre.lower()] if genre else [],
        })

        if len(batch_specs) >= args.batch:
            flush_batch()

    flush_batch()

    logger.info("Encoded %d tracks", len(embeddings))

    # ------------------------------------------------------------------ #
    # Build FAISS index
    # ------------------------------------------------------------------ #
    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, 512)
    norms       = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    emb_matrix  = emb_matrix / norms   # L2-normalise → inner product == cosine sim

    # Save embeddings.npy for debugging / reuse
    emb_path = output_dir / "embeddings.npy"
    np.save(str(emb_path), emb_matrix)
    logger.info("Embeddings saved → %s  shape=%s", emb_path, emb_matrix.shape)

    dim   = emb_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_matrix)
    logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, dim)

    index_path = output_dir / "music_index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved → %s", index_path)

    # ------------------------------------------------------------------ #
    # Save metadata JSON
    # ------------------------------------------------------------------ #
    meta_path = output_dir / "track_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    logger.info("Track metadata saved → %s  (%d entries)", meta_path, len(metadata))

    # Sanity check: nearest neighbours of first track
    distances, indices = index.search(emb_matrix[:1], 6)
    logger.info(
        "Sanity check — top-5 neighbours of track[0] (%s — %s):",
        metadata[0]["title"], metadata[0]["artist"],
    )
    for dist, idx in zip(distances[0][1:6], indices[0][1:6]):
        m = metadata[idx]
        logger.info("  %.3f  %s — %s", dist, m["title"], m["artist"])

    logger.info("Done. Start the backend:  uvicorn main:app --reload")


if __name__ == "__main__":
    main()
