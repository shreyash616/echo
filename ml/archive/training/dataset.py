"""
Self-supervised contrastive dataset for training CnnMusicEncoder.

Each sample produces two independently augmented views of the same track's
mel spectrogram. The NT-Xent loss pulls the two views together and pushes
all other tracks apart — no external labels required.

Energy-matched crop pairs
-------------------------
1. `view_a` is sampled proportional to local window energy, biasing training
   toward choruses, solos, and climactic sections rather than silent intros.
2. `view_b` is sampled from another window of the track weighted by BOTH high
   energy and similarity to `view_a`'s energy level (while discouraging identical
   overlap). This guarantees `view_a` and `view_b` share the same emotional
   intensity (avoiding pairing a quiet acoustic intro with a loud climax).

Emotion-preserving augmentations
--------------------------------
Unlike standard image augmentations or aggressive zero-masking (which create
artificial zero-energy cliffs that corrupt dynamic range, spectral flux, and
RMS energy), these augmentations preserve the track's true emotional and
rhythmic profile so `EmotionFeatureExtractor` is reinforced rather than
suppressed by the contrastive loss:
  1. Subtle gain jitter     [0.95, 1.05] — preserves loudness and dynamic range
  2. Micro pitch shift      [−1, +1] mel bin — preserves spectral centroid
  3. Mean-filled time mask  2 narrow bands (≤8 frames) filled with local mean
                            spectrum + micro-noise (no artificial 0.0 cliffs)
  4. Mean-filled freq mask  up to 8 mel bins filled with local band mean
  5. Micro Gaussian noise   σ=0.005 — robustness to mic/encoding artifacts
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

CROP_FRAMES = 256   # time frames fed to the CNN (~5.9 s at hop=512, sr=22050)
TIME_MASK_F = 8     # max consecutive frames per soft mask (applied twice)
FREQ_MASK_F = 8     # max consecutive mel bins for soft frequency mask
PITCH_MAX   = 1     # max mel-bin shift for pitch augmentation


def _compute_window_energies(mel: np.ndarray, crop_frames: int) -> np.ndarray:
    """Return average energy of each valid crop_frames window across mel."""
    frame_e = mel.mean(axis=0)
    kernel = np.ones(crop_frames, dtype=np.float32) / float(crop_frames)
    return np.convolve(frame_e, kernel, mode="valid")


def _energy_matched_crop_pair(
    mel: np.ndarray,
    crop_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample two (128, crop_frames) windows from `mel` that both emphasise
    high-energy sections AND share similar energy levels.
    """
    T = mel.shape[1]
    if T <= crop_frames:
        pad = crop_frames - T
        padded = np.pad(mel, ((0, 0), (0, pad)), mode="edge")
        return padded.copy(), padded.copy()

    win_e = _compute_window_energies(mel, crop_frames)
    n_wins = len(win_e)

    # 1. Sample view_a proportional to window energy (with 5% floor)
    floor = 0.05 * (float(win_e.max()) + 1e-8)
    weights_a = (win_e - float(win_e.min()) + floor).astype(np.float64)
    probs_a = weights_a / weights_a.sum()
    probs_a /= probs_a.sum()
    start_a = int(np.random.choice(n_wins, p=probs_a))

    # 2. Sample view_b from a window with similar energy to view_a,
    #    preferring non-identical positions when the track is long enough.
    target_e = win_e[start_a]
    e_std = float(win_e.std()) + 1e-4
    sim_weights = np.exp(-0.5 * ((win_e - target_e) / e_std) ** 2)

    # Down-weight immediate overlap with start_a so the two views are distinct
    positions = np.arange(n_wins)
    dist_frames = np.abs(positions - start_a)
    non_overlap = np.where(dist_frames < crop_frames // 3, 0.15, 1.0)

    weights_b = (weights_a * sim_weights * non_overlap).astype(np.float64)
    probs_b = weights_b / (weights_b.sum() + 1e-8)
    probs_b /= probs_b.sum()
    start_b = int(np.random.choice(n_wins, p=probs_b))

    crop_a = mel[:, start_a : start_a + crop_frames].copy()
    crop_b = mel[:, start_b : start_b + crop_frames].copy()
    return crop_a, crop_b


def _augment(patch: np.ndarray) -> np.ndarray:
    """
    Apply emotion-preserving augmentations to a (128, crop_frames) patch.
    """
    patch = patch.astype(np.float32, copy=True)

    # 1. Subtle gain jitter (±5%) — preserves RMS energy & dynamic range
    patch = patch * np.random.uniform(0.95, 1.05)

    # 2. Micro pitch shift: ±1 mel bin with edge replication (no zero-bleed)
    shift = int(np.random.randint(-PITCH_MAX, PITCH_MAX + 1))
    if shift > 0:
        patch = np.pad(patch[:-shift, :], ((shift, 0), (0, 0)), mode="edge")
    elif shift < 0:
        s_abs = -shift
        patch = np.pad(patch[s_abs:, :], ((0, s_abs), (0, 0)), mode="edge")

    # 3. Soft time masking (2 narrow bands filled with track mean spectrum, NOT 0.0)
    # Filling with the mean spectrum + micro-noise avoids artificial zero-energy
    # cliffs that would corrupt dynamic_range, flux_mean, and energy_entropy.
    mean_spectrum = patch.mean(axis=1, keepdims=True)  # (128, 1)
    for _ in range(2):
        t_width = int(np.random.randint(0, TIME_MASK_F + 1))
        if t_width > 0 and patch.shape[1] > t_width:
            t_start = int(np.random.randint(0, patch.shape[1] - t_width))
            noise = np.random.normal(0.0, 0.008, (patch.shape[0], t_width)).astype(np.float32)
            patch[:, t_start : t_start + t_width] = np.clip(mean_spectrum + noise, 0.0, 1.0)

    # 4. Soft frequency masking (filled with local band average, NOT 0.0)
    f_width = int(np.random.randint(0, FREQ_MASK_F + 1))
    if f_width > 0 and patch.shape[0] > f_width:
        f_start = int(np.random.randint(0, patch.shape[0] - f_width))
        band_mean = patch[f_start : f_start + f_width, :].mean(axis=0, keepdims=True)
        patch[f_start : f_start + f_width, :] = band_mean

    # 5. Subtle Gaussian noise
    patch = patch + np.random.normal(0.0, 0.005, patch.shape).astype(np.float32)

    return np.clip(patch, 0.0, 1.0).astype(np.float32)


class ContrastiveSpectrogramDataset(Dataset):
    """
    Dataset of mel spectrogram .npy files for self-supervised contrastive training.

    Each __getitem__ returns two independently augmented, energy-matched views
    of the same track:
        (view_a, view_b)  — both shape (1, 128, crop_frames), float32
    """

    def __init__(self, spec_dir: str, crop_frames: int = CROP_FRAMES, preload: bool = True) -> None:
        self.spec_dir    = Path(spec_dir)
        self.crop_frames = crop_frames
        self.paths       = sorted(self.spec_dir.glob("*.npy"))
        self._cache: dict[int, np.ndarray] = {}

        if not self.paths:
            raise RuntimeError(f"No .npy files found in {self.spec_dir}")

        logger.info(
            "ContrastiveSpectrogramDataset: %d tracks, spec_dir=%s (RAM caching enabled)",
            len(self.paths), self.spec_dir,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx in self._cache:
            mel = self._cache[idx]
        else:
            mel = np.load(self.paths[idx]).astype(np.float32)
            if mel.min() < -1e-3 or mel.max() > 1.0 + 1e-3:
                mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            self._cache[idx] = mel

        crop_a, crop_b = _energy_matched_crop_pair(mel, self.crop_frames)
        view_a = _augment(crop_a)
        view_b = _augment(crop_b)

        # Add channel dim → (1, 128, crop_frames)
        t_a = torch.from_numpy(view_a[np.newaxis])
        t_b = torch.from_numpy(view_b[np.newaxis])
        return t_a, t_b