"""
FMA spectrogram dataset for training CnnMusicEncoder.

Expected inputs:
  - spec_dir:   directory of .npy files (128, T) mel spectrograms,
                produced by ml/data/preprocess.py (one file per track).
  - tracks_csv: FMA tracks.csv with at minimum 'track_id' and 'genre_top' columns.

Each sample returns a (1, 128, T_fixed) mel spectrogram patch and an integer
genre label. A random time crop of fixed width is taken from the spectrogram
at training time; the crop is centred at validation/test time.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Fixed time-axis length fed to the CNN (~10 s clip at hop=512, sr=22050 → ~431 frames).
# Using 256 frames (~3 s) keeps batches small and acts as augmentation.
CROP_FRAMES = 256


def _load_genre_map(tracks_csv: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Parse FMA tracks.csv and return a cleaned DataFrame plus a genre→int map.

    FMA tracks.csv has multi-level headers — we flatten them.
    """
    df = pd.read_csv(tracks_csv, header=[0, 1], index_col=0)
    # Flatten column multi-index: ('track', 'genre_top') → 'track_genre_top'
    df.columns = ["_".join(col).strip() for col in df.columns]
    df.index.name = "track_id"
    df = df.reset_index()

    genre_col = "track_genre_top"
    if genre_col not in df.columns:
        raise ValueError(
            f"Column '{genre_col}' not found in {tracks_csv}. "
            f"Available: {list(df.columns[:20])}"
        )

    df = df[["track_id", genre_col]].dropna(subset=[genre_col])
    df["track_id"] = df["track_id"].astype(int)

    genres = sorted(df[genre_col].unique())
    genre_map: dict[str, int] = {g: i for i, g in enumerate(genres)}
    df["label"] = df[genre_col].map(genre_map)

    logger.info(
        "Genre map: %d genres — %s",
        len(genre_map),
        ", ".join(f"{g}({i})" for g, i in list(genre_map.items())[:5]) + "…",
    )
    return df, genre_map


class FMASpectrogramDataset(Dataset):
    """
    Dataset of mel spectrogram .npy files with FMA genre labels.

    Args:
        spec_dir:   Path to directory containing *.npy spectrogram files.
                    Each file is named '<track_id>.npy' and has shape (128, T).
        tracks_csv: Path to FMA tracks.csv (multi-level header format).
        crop_frames: Number of time frames to crop per sample.
        augment:    If True, use random crop + random amplitude scale.
                    If False (val/test), use centre crop.
    """

    def __init__(
        self,
        spec_dir: str,
        tracks_csv: str,
        crop_frames: int = CROP_FRAMES,
        augment: bool = True,
    ) -> None:
        self.spec_dir = Path(spec_dir)
        self.crop_frames = crop_frames
        self.augment = augment

        # Load genre labels
        df, self.genre_map = _load_genre_map(tracks_csv)

        # Match .npy files to track IDs that have genre labels
        available: dict[int, Path] = {}
        for p in self.spec_dir.glob("*.npy"):
            try:
                tid = int(p.stem)
                available[tid] = p
            except ValueError:
                pass

        merged = df[df["track_id"].isin(available)].copy()
        merged["npy_path"] = merged["track_id"].map(available)

        self.paths = merged["npy_path"].tolist()
        self.labels = merged["label"].values.astype(np.int64)

        logger.info(
            "FMASpectrogramDataset: %d samples, %d classes, spec_dir=%s",
            len(self.paths), self.num_classes, self.spec_dir,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel = np.load(self.paths[idx])          # (128, T), float32, in [0, 1]
        patch = self._crop(mel)                 # (128, crop_frames)

        if self.augment:
            # Random amplitude scale in [0.8, 1.2]
            patch = patch * np.random.uniform(0.8, 1.2)
            patch = np.clip(patch, 0.0, 1.0)

        # Add channel dim: (1, 128, crop_frames)
        x = torch.from_numpy(patch[np.newaxis].astype(np.float32))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def _crop(self, mel: np.ndarray) -> np.ndarray:
        """Return a (128, crop_frames) slice of the spectrogram."""
        T = mel.shape[1]
        if T <= self.crop_frames:
            # Pad with zeros if clip is shorter than crop_frames
            pad = self.crop_frames - T
            return np.pad(mel, ((0, 0), (0, pad)), mode="constant")

        if self.augment:
            start = np.random.randint(0, T - self.crop_frames)
        else:
            start = (T - self.crop_frames) // 2   # centre crop

        return mel[:, start: start + self.crop_frames]

    @property
    def num_classes(self) -> int:
        return len(self.genre_map)
