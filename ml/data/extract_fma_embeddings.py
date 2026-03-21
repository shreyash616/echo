"""
Extract CLAP audio embeddings from FMA (Free Music Archive) audio files.

Replaces download_fma.py — no Spotify dependency whatsoever.

FMA datasets (choose one):
  fma_small  —  8,000 tracks, 30s clips, ~8 GB   ← recommended for dev
  fma_medium — 25,000 tracks, 30s clips, ~22 GB
  fma_large  — 106k  tracks, 30s clips, ~93 GB

Download from: https://github.com/mdeff/fma
  - Audio:    fma_small.zip / fma_medium.zip
  - Metadata: fma_metadata.zip  (contains tracks.csv, genres.csv, etc.)

Usage:
    python ml/data/extract_fma_embeddings.py \
        --audio-dir  ml/data/fma_small \
        --fma-tracks ml/data/fma_metadata/tracks.csv \
        --output-dir ml/data \
        --batch 32

Outputs:
    ml/data/embeddings.npy   — (N, 512) float32 CLAP embeddings
    ml/data/tracks.csv       — metadata for every embedded track
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.models.music_encoder import CLAPMusicEncoder, TARGET_SR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# FMA genre → vibe tags mapping
GENRE_VIBES: dict[str, list[str]] = {
    "Electronic":    ["energetic", "hypnotic"],
    "Hip-Hop":       ["groovy", "energetic"],
    "Rock":          ["aggressive", "energetic"],
    "Folk":          ["atmospheric", "melodic"],
    "Classical":     ["melodic", "hypnotic"],
    "Instrumental":  ["hypnotic", "melodic"],
    "Pop":           ["euphoric", "groovy"],
    "Jazz":          ["melodic", "groovy"],
    "International": ["atmospheric", "groovy"],
    "Experimental":  ["hypnotic", "dark"],
    "Soul-RnB":      ["euphoric", "groovy"],
    "Spoken":        ["melodic"],
    "Old-Time / Historic": ["atmospheric", "melodic"],
    "Blues":         ["dark", "melodic"],
    "Country":       ["atmospheric", "melodic"],
    "Easy Listening": ["chill", "melodic"],
}


def track_id_to_path(track_id: int, audio_dir: str) -> str:
    """FMA file layout: audio_dir/NNN/NNNNNN.mp3"""
    tid = f"{track_id:06d}"
    return os.path.join(audio_dir, tid[:3], f"{tid}.mp3")


CLIP_DURATION = 60  # seconds


def load_audio(path: str) -> np.ndarray | None:
    """
    Load a random 60-second clip from an mp3 file → mono float32 numpy array at TARGET_SR.
    Returns None if the file is unreadable or shorter than 60 seconds.
    """
    try:
        info = torchaudio.info(path)
        total_frames = info.num_frames
        file_sr = info.sample_rate
        min_frames = CLIP_DURATION * file_sr
        if total_frames < min_frames:
            return None

        max_offset = total_frames - min_frames
        frame_offset = random.randint(0, max_offset)
        num_frames = CLIP_DURATION * file_sr

        waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze().numpy().astype(np.float32)
    except Exception as e:
        logger.debug("Failed to load %s: %s", path, e)
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir",  required=True, help="Path to fma_small/ or fma_medium/")
    p.add_argument("--fma-tracks", required=True, help="Path to fma_metadata/tracks.csv")
    p.add_argument("--output-dir", default="ml/data")
    p.add_argument("--batch",      type=int, default=32, help="Tracks per CLAP forward pass")
    p.add_argument("--limit",      type=int, default=0,  help="Max tracks (0 = all)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load FMA metadata
    # ------------------------------------------------------------------ #
    logger.info("Loading FMA metadata from %s", args.fma_tracks)
    raw = pd.read_csv(args.fma_tracks, index_col=0, header=[0, 1])

    tracks = pd.DataFrame({
        "track_id":  raw.index.astype(int),
        "title":     raw[("track",  "title")].astype(str),
        "artist":    raw[("artist", "name")].astype(str),
        "album":     raw[("album",  "title")].astype(str),
        "genre":     raw[("track",  "genre_top")].astype(str).fillna(""),
        "duration":  pd.to_numeric(raw[("track", "duration")], errors="coerce").fillna(0),
    })

    if args.limit > 0:
        tracks = tracks.head(args.limit)

    # Keep only tracks whose audio file exists
    tracks["audio_path"] = tracks["track_id"].apply(
        lambda tid: track_id_to_path(tid, args.audio_dir)
    )
    tracks = tracks[tracks["audio_path"].apply(os.path.exists)].reset_index(drop=True)
    logger.info("%d tracks with audio files found", len(tracks))

    # ------------------------------------------------------------------ #
    # Load CLAP encoder
    # ------------------------------------------------------------------ #
    enc = CLAPMusicEncoder()
    enc.load()

    # ------------------------------------------------------------------ #
    # Extract embeddings in batches
    # ------------------------------------------------------------------ #
    all_embeddings: list[np.ndarray] = []
    valid_indices: list[int] = []

    batch_audio: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch_audio:
            return
        # Pad/trim all clips to the same length for batched processing
        max_len = max(a.shape[0] for a in batch_audio)
        padded = [
            np.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a[:max_len]
            for a in batch_audio
        ]
        batch_np = np.stack(padded, axis=0)  # (B, T)

        inputs = enc._processor(
            audios=list(batch_np),
            return_tensors="pt",
            sampling_rate=TARGET_SR,
        )
        inputs = {k: v.to(enc._device) for k, v in inputs.items()}

        with torch.no_grad():
            emb = enc._model.get_audio_features(**inputs)  # (B, 512)

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        emb_np = emb.cpu().numpy().astype(np.float32)

        for i, idx in enumerate(batch_indices):
            all_embeddings.append(emb_np[i])
            valid_indices.append(idx)

        batch_audio.clear()
        batch_indices.clear()

    for i, row in tqdm(tracks.iterrows(), total=len(tracks), desc="Extracting embeddings"):
        audio = load_audio(row["audio_path"])
        if audio is None:
            continue

        batch_audio.append(audio)
        batch_indices.append(i)

        if len(batch_audio) >= args.batch:
            flush_batch()

    flush_batch()   # remaining

    if not all_embeddings:
        logger.error("No embeddings extracted. Check --audio-dir path.")
        return

    embeddings = np.stack(all_embeddings, axis=0)   # (N, 512)
    valid_tracks = tracks.iloc[valid_indices].reset_index(drop=True)

    logger.info("Extracted %d embeddings, shape %s", len(embeddings), embeddings.shape)

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    emb_path = output_dir / "embeddings.npy"
    np.save(str(emb_path), embeddings)
    logger.info("Embeddings saved → %s", emb_path)

    # Add vibe tags from genre
    valid_tracks["vibes"] = valid_tracks["genre"].map(
        lambda g: GENRE_VIBES.get(g, ["melodic"])
    )

    meta_path = output_dir / "tracks.csv"
    valid_tracks.to_csv(meta_path, index=False)
    logger.info("Track metadata saved → %s  (%d tracks)", meta_path, len(valid_tracks))

    logger.info("Done. Next step:  python ml/inference/build_index.py")


if __name__ == "__main__":
    main()
