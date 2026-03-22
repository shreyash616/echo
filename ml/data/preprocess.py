"""
Mel spectrogram pre-processing for raw audio training.

This script:
  1. Reads a directory of .mp3/.wav files (FMA large/medium/small dataset).
  2. Extracts 128-bin mel spectrograms using librosa.
  3. Saves them as .npy files for fast loading during training.

Runs in parallel across all CPU cores by default.

Usage:
    python ml/data/preprocess.py \
        --audio-dir  ml/data/fma_large/fma_large \
        --output-dir ml/data/spectrograms \
        --workers    8 \
        --sr         22050
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

N_MELS     = 128
HOP_LENGTH = 512
N_FFT      = 2048


def audio_to_mel(
    path: str,
    sr: int = 22_050,
    n_mels: int = N_MELS,
) -> np.ndarray | None:
    """
    Load the full audio track, compute mel spectrogram,
    return (n_mels, T) float32 in [0, 1].
    Returns None if the file is unreadable or silent.
    """
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        mel    = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_db.astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Worker — runs in a subprocess, must be a module-level function to be
# picklable on Windows (spawn start method).
# ---------------------------------------------------------------------------

def _worker(args: tuple[str, str, int, bool]) -> tuple[str, bool]:
    """
    Process one audio file.
    Returns (stem, success).
    """
    src_path, out_path, sr, fp16 = args
    if os.path.exists(out_path):
        return src_path, True          # already done — count as success

    mel = audio_to_mel(src_path, sr=sr)
    if mel is None:
        return src_path, False

    np.save(out_path, mel.astype(np.float16) if fp16 else mel)
    return src_path, True


def main() -> None:
    p = argparse.ArgumentParser(
        description="Parallel mel spectrogram extraction from FMA audio files"
    )
    p.add_argument("--audio-dir",  required=True, help="Directory with .mp3/.wav files")
    p.add_argument("--output-dir", required=True, help="Directory to write .npy files")
    p.add_argument("--sr",         type=int,   default=22_050)
    p.add_argument("--workers",    type=int,   default=min(8, mp.cpu_count()),
                   help="Parallel worker processes (default: min(8, cpu_count))")
    p.add_argument("--fp16",      action="store_true",
                   help="Save spectrograms as float16 (halves disk usage, ~0.63 MB/track)")
    args = p.parse_args()

    audio_dir  = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        list(audio_dir.rglob("*.mp3")) + list(audio_dir.rglob("*.wav"))
    )
    logger.info("Found %d audio files in %s", len(audio_files), audio_dir)

    if not audio_files:
        logger.error("No audio files found. Check --audio-dir.")
        return

    if args.fp16:
        logger.info("Saving as float16 (~0.63 MB/track)")

    # Build work list — skip files already processed
    work = [
        (str(path), str(output_dir / (path.stem + ".npy")), args.sr, args.fp16)
        for path in audio_files
    ]
    already_done = sum(1 for _, out, _, _, _ in work if os.path.exists(out))
    if already_done:
        logger.info("Skipping %d already-processed files.", already_done)

    logger.info("Starting parallel preprocessing with %d workers…", args.workers)

    saved   = 0
    skipped = 0

    with mp.Pool(processes=args.workers) as pool:
        for _, success in tqdm(
            pool.imap_unordered(_worker, work, chunksize=16),
            total=len(work),
            desc="Preprocessing",
            unit="track",
        ):
            if success:
                saved += 1
            else:
                skipped += 1

    logger.info(
        "Done. Saved/existing: %d  |  Skipped (unreadable/short): %d  |  Total: %d",
        saved, skipped, len(audio_files),
    )


if __name__ == "__main__":
    # Required on Windows — prevents recursive subprocess spawning
    mp.freeze_support()
    main()
