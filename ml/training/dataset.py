"""
Self-supervised contrastive dataset for training CnnMusicEncoder.

Each sample produces two independently augmented views of the same track's
mel spectrogram. The NT-Xent loss then pulls the two views together in
embedding space and pushes all other tracks apart — no genre labels required.

Augmentations applied independently to each view:
  - Random time crop (position varies between views)
  - Random amplitude scale  [0.8, 1.2]
  - Time masking: zero out up to 40 consecutive frames (SpecAugment-style)
  - Frequency masking: zero out up to 20 mel bins
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CROP_FRAMES  = 256   # time frames fed to the CNN (~3 s at hop=512, sr=22050)
TIME_MASK_F  = 40    # max consecutive frames to zero out
FREQ_MASK_F  = 20    # max consecutive mel bins to zero out


def _augment(patch: np.ndarray) -> np.ndarray:
    """Apply random augmentations to a (128, crop_frames) patch."""
    # Amplitude scale
    patch = patch * np.random.uniform(0.8, 1.2)
    patch = np.clip(patch, 0.0, 1.0)

    # Time masking
    t_start = np.random.randint(0, patch.shape[1])
    t_width = np.random.randint(0, TIME_MASK_F + 1)
    patch[:, t_start: t_start + t_width] = 0.0

    # Frequency masking
    f_start = np.random.randint(0, patch.shape[0])
    f_width = np.random.randint(0, FREQ_MASK_F + 1)
    patch[f_start: f_start + f_width, :] = 0.0

    return patch


class ContrastiveSpectrogramDataset(Dataset):
    """
    Dataset of mel spectrogram .npy files for self-supervised contrastive training.

    Each __getitem__ returns two differently augmented views of the same track:
        (view_a, view_b)  — both shape (1, 128, crop_frames), float32

    No genre labels or metadata CSV required.

    Args:
        spec_dir:    Path to directory containing *.npy spectrogram files.
                     Each file has shape (128, T), float32, values in [0, 1].
        crop_frames: Number of time frames per view.
    """

    def __init__(self, spec_dir: str, crop_frames: int = CROP_FRAMES) -> None:
        self.spec_dir    = Path(spec_dir)
        self.crop_frames = crop_frames
        self.paths       = sorted(self.spec_dir.glob("*.npy"))

        if not self.paths:
            raise RuntimeError(f"No .npy files found in {self.spec_dir}")

        logger.info(
            "ContrastiveSpectrogramDataset: %d tracks, spec_dir=%s",
            len(self.paths), self.spec_dir,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        mel = np.load(self.paths[idx])   # (128, T), float32

        view_a = _augment(self._random_crop(mel))
        view_b = _augment(self._random_crop(mel))

        # Add channel dim → (1, 128, crop_frames)
        t_a = torch.from_numpy(view_a[np.newaxis].astype(np.float32))
        t_b = torch.from_numpy(view_b[np.newaxis].astype(np.float32))
        return t_a, t_b

    def _random_crop(self, mel: np.ndarray) -> np.ndarray:
        """Return a (128, crop_frames) random crop of the spectrogram."""
        T = mel.shape[1]
        if T <= self.crop_frames:
            pad = self.crop_frames - T
            return np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        start = np.random.randint(0, T - self.crop_frames)
        return mel[:, start: start + self.crop_frames]
