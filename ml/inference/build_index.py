"""
Build the FAISS similarity index from mel spectrograms using the trained
CnnMusicEncoder exported to ONNX, and compute self-generated per-track
musical & emotional analytics (BPM, Key, Energy, Valence, Vibes).

Pipeline:
  1. Load all .npy spectrogram files from spec_dir (output of preprocess.py
     or harvest_deezer_previews.py).
  2. Enrich each track with catalog metadata (FMA tracks.csv or harvested
     catalog_tracks.json) PLUS self-generated audio analytics computed directly
     from the track's mel spectrogram via `analyze_mel_spectrogram`.
  3. Extract multiple energy-weighted crops across each track, encode them with
     the ONNX model, and pool into a single L2-normalised 512-d embedding.
  4. Insert into a FAISS IndexFlatIP (cosine similarity).
  5. Write music_index.faiss + track_metadata.json + embeddings.npy.

Usage (with FMA metadata):
    python ml/inference/build_index.py \
        --spec-dir    ml/data/spectrograms \
        --tracks-csv  ml/data/fma_metadata/fma_metadata/tracks.csv \
        --onnx-model  ml/inference/music_encoder.onnx \
        --output-dir  ml/inference

Usage (with harvested Deezer preview catalog):
    python ml/inference/build_index.py \
        --spec-dir     ml/data/deezer_spectrograms \
        --catalog-json ml/data/deezer_catalog.json \
        --onnx-model   ml/inference/music_encoder.onnx \
        --output-dir   ml/inference
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.inference.audio_features import analyze_mel_spectrogram

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

CROP_FRAMES = 256   # crop width used at inference — matches training
N_CROPS     = 4     # number of energy-weighted crops pooled per track


def extract_energy_crops(
    mel: np.ndarray,
    crop_frames: int = CROP_FRAMES,
    n_crops: int = N_CROPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract up to `n_crops` deterministic windows of shape (n_crops, 128, crop_frames)
    across the spectrogram, alongside their relative energy weights.

    Unlike a single centre-crop (which misses choruses or lands on a quiet bridge),
    pooling multiple windows weighted by local energy captures the track's true
    hook and acoustic character.
    """
    mel = np.asarray(mel, dtype=np.float32)
    T = mel.shape[1]
    if T <= crop_frames:
        pad = crop_frames - T
        padded = np.pad(mel, ((0, 0), (0, pad)), mode="edge")
        return padded[np.newaxis, :, :], np.array([1.0], dtype=np.float32)

    if n_crops <= 1 or T <= crop_frames * 3 // 2:
        start = (T - crop_frames) // 2
        crop = mel[:, start : start + crop_frames]
        return crop[np.newaxis, :, :], np.array([1.0], dtype=np.float32)

    # Divide the track into `n_crops` segments and pick the highest-energy window in each
    crops: list[np.ndarray] = []
    energies: list[float] = []
    frame_e = mel.mean(axis=0)
    kernel = np.ones(crop_frames, dtype=np.float32) / float(crop_frames)
    win_e = np.convolve(frame_e, kernel, mode="valid")  # length: T - crop_frames + 1
    max_start = len(win_e) - 1

    seg_bounds = np.linspace(0, max_start, n_crops + 1, dtype=int)
    for i in range(n_crops):
        lo, hi = int(seg_bounds[i]), int(seg_bounds[i + 1])
        if hi <= lo:
            best_start = lo
        else:
            best_start = lo + int(np.argmax(win_e[lo : hi + 1]))
        crops.append(mel[:, best_start : best_start + crop_frames])
        energies.append(float(win_e[best_start]) + 1e-4)

    weights = np.array(energies, dtype=np.float32)
    weights = weights / weights.sum()
    return np.stack(crops, axis=0), weights


def load_tracks(tracks_csv: str) -> pd.DataFrame:
    """Parse FMA multi-level tracks.csv and return a flat DataFrame."""
    df = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)
    df.columns = ["_".join(col).strip() for col in df.columns]
    df.index.name = "track_id"
    df = df.reset_index()
    df["track_id"] = df["track_id"].astype(int)
    return df


def _clean_str(val: object, default: str = "") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    s = str(val).strip()
    return default if s.lower() == "nan" else s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec-dir",     required=True,
                   help="Directory of .npy spectrogram files")
    p.add_argument("--tracks-csv",   default=None,
                   help="FMA tracks.csv (multi-level header) — enriches FMA metadata")
    p.add_argument("--catalog-json", default=None,
                   help="Catalog JSON from harvest_deezer_previews.py — enriches real track metadata")
    p.add_argument("--onnx-model",   default="ml/inference/music_encoder.onnx",
                   help="Exported ONNX model path")
    p.add_argument("--output-dir",   default="ml/inference")
    p.add_argument("--batch",        type=int, default=64, help="Encode batch size (in tracks)")
    p.add_argument("--crop-frames",  type=int, default=CROP_FRAMES)
    p.add_argument("--n-crops",      type=int, default=N_CROPS,
                   help="Number of energy-weighted crops pooled per track")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load track metadata (FMA tracks.csv or harvested catalog.json)
    # ------------------------------------------------------------------ #
    tracks_by_key: dict[str, dict] = {}
    is_catalog_json = False

    if args.catalog_json and Path(args.catalog_json).exists():
        logger.info("Loading harvested catalog metadata from %s", args.catalog_json)
        with open(args.catalog_json, encoding="utf-8") as f:
            cat_list = json.load(f)
        for item in cat_list:
            tracks_by_key[str(item["id"])] = item
        is_catalog_json = True
    elif args.tracks_csv and Path(args.tracks_csv).exists():
        logger.info("Loading FMA track metadata from %s", args.tracks_csv)
        tracks_df = load_tracks(args.tracks_csv)
        for _, row in tracks_df.iterrows():
            tid = int(row["track_id"])
            tracks_by_key[str(tid)] = row.to_dict()
            tracks_by_key[f"{tid:06d}"] = row.to_dict()
    else:
        # Auto-detect FMA tracks.csv in standard location if present
        default_csv = Path("ml/data/fma_metadata/fma_metadata/tracks.csv")
        if default_csv.exists():
            logger.info("Auto-detected FMA metadata at %s", default_csv)
            tracks_df = load_tracks(str(default_csv))
            for _, row in tracks_df.iterrows():
                tid = int(row["track_id"])
                tracks_by_key[str(tid)] = row.to_dict()
                tracks_by_key[f"{tid:06d}"] = row.to_dict()
        else:
            logger.warning(
                "No --tracks-csv or --catalog-json found! Tracks will have stub titles "
                "('Track <id>') which cannot be resolved on Deezer by name."
            )

    # ------------------------------------------------------------------ #
    # Collect spectrogram files
    # ------------------------------------------------------------------ #
    spec_dir = Path(args.spec_dir)
    npy_files: list[tuple[str, Path]] = []
    for npy_path in sorted(spec_dir.glob("*.npy")):
        stem = npy_path.stem
        # Try exact stem or integer-stripped stem
        key = stem
        if key not in tracks_by_key:
            try:
                key = str(int(stem))
            except ValueError:
                pass
        if not tracks_by_key or key in tracks_by_key:
            npy_files.append((key, npy_path))

    logger.info(
        "Found %d usable spectrogram files (out of %d total .npy files)",
        len(npy_files),
        len(list(spec_dir.glob("*.npy"))),
    )
    if not npy_files:
        raise RuntimeError(
            "No matching spectrogram files found — check --spec-dir and metadata arguments"
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
    # Multi-crop encode + self-generated audio analytics
    # ------------------------------------------------------------------ #
    embeddings: list[np.ndarray] = []
    metadata:   list[dict]       = []

    batch_crops:   list[np.ndarray] = []
    batch_counts:  list[int]        = []
    batch_weights: list[np.ndarray] = []
    batch_metas:   list[dict]       = []

    def flush_batch() -> None:
        if not batch_crops:
            return
        all_patches = np.concatenate(batch_crops, axis=0).astype(np.float32)  # (sum_K, 128, T)
        all_patches = all_patches[:, np.newaxis, :, :]                        # (sum_K, 1, 128, T)
        out = sess.run(None, {input_name: all_patches})[0]                    # (sum_K, D)

        offset = 0
        for count, weights in zip(batch_counts, batch_weights):
            track_embs = out[offset : offset + count]                         # (K, D)
            offset += count
            pooled = (track_embs * weights[:, np.newaxis]).sum(axis=0)
            norm = np.linalg.norm(pooled) + 1e-8
            embeddings.append((pooled / norm).astype(np.float32))

        metadata.extend(batch_metas)
        batch_crops.clear()
        batch_counts.clear()
        batch_weights.clear()
        batch_metas.clear()

    for key, npy_path in tqdm(npy_files, desc="Encoding & Analyzing"):
        mel = np.load(npy_path).astype(np.float32)  # (128, T)
        if mel.min() < -1e-3 or mel.max() > 1.0 + 1e-3:
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        crops, weights = extract_energy_crops(
            mel, crop_frames=args.crop_frames, n_crops=args.n_crops
        )
        row = tracks_by_key.get(key, {})

        if is_catalog_json:
            genre = _clean_str(row.get("genre"), "")
            analytics = analyze_mel_spectrogram(mel, genre_hint=genre)
            track_meta = {
                "id":          str(row.get("id", key)),
                "title":       _clean_str(row.get("title"), f"Track {key}"),
                "artist":      _clean_str(row.get("artist"), ""),
                "album":       _clean_str(row.get("album"), ""),
                "albumArtUrl": _clean_str(row.get("albumArtUrl"), ""),
                "previewUrl":  row.get("previewUrl"),
                "durationMs":  int(row.get("durationMs", 30000)),
                "genre":       genre,
                "bpm":         analytics["bpm"],
                "key":         analytics["key"],
                "energy":      analytics["energy"],
                "valence":     analytics["valence"],
                "vibes":       analytics["vibes"],
            }
        else:
            genre = _clean_str(row.get("track_genre_top"), "")
            analytics = analyze_mel_spectrogram(mel, genre_hint=genre)
            dur_s = row.get("track_duration", 0)
            try:
                dur_ms = int(float(dur_s) * 1000) if dur_s and not pd.isna(dur_s) else 0
            except Exception:
                dur_ms = 0
            track_meta = {
                "id":          f"fma_{key}",
                "title":       _clean_str(row.get("track_title"), f"Track {key}"),
                "artist":      _clean_str(row.get("artist_name"), ""),
                "album":       _clean_str(row.get("album_title"), ""),
                "albumArtUrl": "",
                "previewUrl":  None,
                "durationMs":  dur_ms,
                "genre":       genre,
                "bpm":         analytics["bpm"],
                "key":         analytics["key"],
                "energy":      analytics["energy"],
                "valence":     analytics["valence"],
                "vibes":       analytics["vibes"],
            }

        batch_crops.append(crops)
        batch_counts.append(len(crops))
        batch_weights.append(weights)
        batch_metas.append(track_meta)

        if len(batch_counts) >= args.batch:
            flush_batch()

    flush_batch()

    logger.info("Encoded and analyzed %d tracks", len(embeddings))

    # ------------------------------------------------------------------ #
    # Build FAISS index
    # ------------------------------------------------------------------ #
    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)  # (N, 512)
    norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
    emb_matrix = emb_matrix / norms   # L2-normalise → inner product == cosine sim

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
    if len(metadata) > 1:
        k_check = min(6, len(metadata))
        distances, indices = index.search(emb_matrix[:1], k_check)
        m0 = metadata[0]
        logger.info(
            "Sanity check — top neighbours of track[0] (%s — %s | %d BPM | %s | E=%.2f V=%.2f):",
            m0["title"], m0["artist"], m0["bpm"], m0["key"], m0["energy"], m0["valence"],
        )
        for dist, idx in zip(distances[0][1:k_check], indices[0][1:k_check]):
            m = metadata[idx]
            logger.info(
                "  %.3f  %s — %s  [%d BPM, %s, vibes=%s]",
                dist, m["title"], m["artist"], m["bpm"], m["key"], m["vibes"],
            )

    logger.info("Done. Start the backend:  uvicorn main:app --reload")


if __name__ == "__main__":
    main()