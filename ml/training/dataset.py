"""
Self-supervised contrastive dataset for training CnnMusicEncoder.

Each sample produces two independently augmented views of the same track's
mel spectrogram. The NT-Xent loss then pulls the two views together and pushes
all other tracks apart — no genre labels required.

Crop selection — energy-weighted
---------------------------------
Crops are not sampled uniformly. We compute the average energy of every
possible crop-length window across the spectrogram and sample proportionally.
This biases training toward choruses, solos, and climactic sections — the
moments that define the emotional character of a song — rather than wasting
capacity on silent intros, sparse verses, or fade-outs.

A small probability floor (5% of the max window energy) ensures quiet sections
still have a chance to be seen.

Augmentations (applied independently to each view)
---------------------------------------------------
Each augmentation is designed to make two views of the same song hard to match
on superficial features, forcing the model to learn deep acoustic structure:

  1. Amplitude scale     [0.7, 1.3]  — ignore raw loudness, focus on shape
  2. Pitch shift         [−2, +2] mel bins  — same song in different key
  3. Time masking        up to 40 frames  — robustness to missing segments
  4. Frequency masking   up to 20 mel bins — robustness to EQ/mic variation
  5. Gaussian noise      σ=0.01  — robustness to recording quality

Crucially, none of these augmentations change spectral flux, dynamic range,
or energy entropy — the features that distinguish "adrenaline-y" from "soothing".
The model is therefore forced to preserve those distinctions in the embedding.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CROP_FRAMES  = 256   # time frames fed to the CNN (~3 s at hop=512, sr=22050)
TIME_MASK_F  = 40    # max consecutive frames to zero out per augmentation
FREQ_MASK_F  = 20    # max consecutive mel bins to zero out per augmentation
PITCH_MAX    = 2     # max mel-bin shift for pitch augmentation


def _energy_weighted_crop(mel: np.ndarray, crop_frames: int) -> np.ndarray:
    """
    Sample a (128, crop_frames) window proportional to local energy.

    A sliding window of width `crop_frames` is convolved across frame energies.
    Positions with higher average energy are more likely to be chosen, biasing
    training toward emotionally significant sections (choruses, solos, peaks).
    A 5% floor prevents any region from being completely excluded.
    """
    T = mel.shape[1]
    if T <= crop_frames:
        pad = crop_frames - T
        return np.pad(mel, ((0, 0), (0, pad)), mode="constant")

    # Frame-wise mean energy across mel bins  →  (T,)
    frame_e = mel.mean(axis=0)

    # Average energy of each crop-sized window  →  (T - crop_frames + 1,)
    kernel        = np.ones(crop_frames) / crop_frames
    window_energy = np.convolve(frame_e, kernel, mode="valid")

    # Add 5% floor so quiet regions are never fully excluded
    floor         = 0.05 * (window_energy.max() + 1e-8)
    window_energy = window_energy - window_energy.min() + floor

    probs = window_energy / window_energy.sum()
    start = np.random.choice(len(probs), p=probs)
    return mel[:, start : start + crop_frames]


def _augment(patch: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a (128, crop_frames) patch.

    All transforms preserve the spectral flux, dynamic range, and energy
    distribution that encode emotional character.
    """
    # 1. Amplitude scale
    patch = patch * np.random.uniform(0.7, 1.3)

    # 2. Pitch shift: roll mel bins up/down by up to PITCH_MAX bins
    shift = np.random.randint(-PITCH_MAX, PITCH_MAX + 1)
    if shift != 0:
        patch = np.roll(patch, shift, axis=0)
        # Zero out the wrapped edge so rolled content doesn't bleed in
        if shift > 0:
            patch[:shift, :] = 0.0
        else:
            patch[shift:, :] = 0.0

    # 3. Time masking
    t_width = np.random.randint(0, TIME_MASK_F + 1)
    if t_width > 0:
        t_start = np.random.randint(0, max(1, patch.shape[1] - t_width))
        patch[:, t_start : t_start + t_width] = 0.0

    # 4. Frequency masking
    f_width = np.random.randint(0, FREQ_MASK_F + 1)
    if f_width > 0:
        f_start = np.random.randint(0, max(1, patch.shape[0] - f_width))
        patch[f_start : f_start + f_width, :] = 0.0

    # 5. Gaussian noise
    patch = patch + np.random.normal(0.0, 0.01, patch.shape)

    return np.clip(patch, 0.0, 1.0).astype(np.float32)


class ContrastiveSpectrogramDataset(Dataset):
    """
    Dataset of mel spectrogram .npy files for self-supervised contrastive training.

    Each __getitem__ returns two independently augmented, energy-biased views
    of the same track:
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

        # Two independently cropped + augmented views of the same track
        view_a = _augment(_energy_weighted_crop(mel, self.crop_frames))
        view_b = _augment(_energy_weighted_crop(mel, self.crop_frames))

        # Add channel dim → (1, 128, crop_frames)
        t_a = torch.from_numpy(view_a[np.newaxis])
        t_b = torch.from_numpy(view_b[np.newaxis])
        return t_a, t_b
