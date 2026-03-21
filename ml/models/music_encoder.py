"""
CnnMusicEncoder — CNN acoustic backbone + EmotionFeatureExtractor → joint embedding.

Architecture
------------
Acoustic path:  (B, 1, 128, T) → ConvBlocks → AdaptiveAvgPool → 512-d
Emotion path:   (B, 1, 128, T) → 12 differentiable audio features → BN → MLP → 64-d (L2-norm)
Joint:          cat(512, 64) → Linear → embedding_dim → L2-norm

EmotionFeatureExtractor
-----------------------
Computes 12 features that correlate with emotional/energy qualities of audio.
These are computed differentiably so gradients flow through them during training.

  1.  rms_mean        — overall energy level (loud vs quiet)
  2.  rms_std         — energy variation over time (dynamic vs flat)
  3.  centroid_mean   — spectral brightness (harsh/bright vs warm/mellow)
  4.  centroid_std    — brightness consistency
  5.  flux_mean       — mean spectral flux (adrenaline/aggression vs smoothness)
  6.  flux_std        — variation in spectral flux (erratic vs steady)
  7.  dynamic_range   — peak-to-floor energy swing (emotional intensity)
  8.  rolloff_mean    — freq where 85% energy is contained (tonal weight)
  9.  contrast_mean   — high vs low mel band energy difference (fullness)
  10. contrast_std    — variation in spectral contrast
  11. hf_ratio        — high-frequency energy fraction (presence/brightness)
  12. energy_entropy  — spread of energy over time (sustained vs burst)

These guarantee that "adrenaline-y" (high flux, high rms_std, high dynamic_range)
and "soothing" (low flux, low rms_std, smooth) are captured even if the CNN
backbone maps both as "guitar solo" textures.

ProjectionHead
--------------
Used ONLY during SimCLR training (not exported to ONNX).
NT-Xent loss is computed on projection outputs; the encoder output (pre-projection)
is what gets stored in FAISS and used at inference — standard SimCLR practice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU, with optional stride for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionFeatureExtractor(nn.Module):
    """
    Differentiable extractor of 12 emotion-correlated audio features from a mel spectrogram.
    Projects them to a normalised emotion embedding via BN → MLP → L2-norm.
    """

    N_FEATS = 12

    def __init__(self, n_mels: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        # Mel bin indices (1-indexed) used for spectral centroid and rolloff
        self.register_buffer(
            "mel_idx",
            torch.arange(1, n_mels + 1, dtype=torch.float32),  # (n_mels,)
        )
        # Normalise raw features before MLP (handles large scale differences)
        self.bn  = nn.BatchNorm1d(self.N_FEATS)
        self.mlp = nn.Sequential(
            nn.Linear(self.N_FEATS, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, T) mel spectrogram, values in [0, 1].
        Returns:
            (B, out_dim) L2-normalised emotion embedding.
        """
        s   = x.squeeze(1)   # (B, 128, T)
        eps = 1e-8

        # ── 1-2: RMS energy per frame ──────────────────────────────────────
        frame_e  = s.mean(dim=1)           # (B, T)
        rms_mean = frame_e.mean(dim=1)     # (B,)
        rms_std  = frame_e.std(dim=1)      # (B,)

        # ── 3-4: Spectral centroid ─────────────────────────────────────────
        # Weighted average mel bin index per frame
        idx      = self.mel_idx.view(1, -1, 1)                          # (1, 128, 1)
        centroid = (s * idx).sum(dim=1) / (s.sum(dim=1) + eps)          # (B, T)
        centroid_mean = centroid.mean(dim=1)
        centroid_std  = centroid.std(dim=1)

        # ── 5-6: Spectral flux ─────────────────────────────────────────────
        # L1 frame-to-frame change — high = adrenaline-y, low = soothing
        flux      = (s[:, :, 1:] - s[:, :, :-1]).abs().mean(dim=1)      # (B, T-1)
        flux_mean = flux.mean(dim=1)
        flux_std  = flux.std(dim=1)

        # ── 7: Dynamic range ───────────────────────────────────────────────
        # Max minus min frame energy — captures emotional intensity
        dyn_range = (
            frame_e.max(dim=1).values - frame_e.min(dim=1).values       # (B,)
        )

        # ── 8: Spectral rolloff ────────────────────────────────────────────
        # Mean mel bin where 85% of cumulative energy is reached per frame
        cumsum       = s.cumsum(dim=1)                                   # (B, 128, T)
        total        = cumsum[:, -1:, :] + eps                           # (B, 1, T)
        rolloff_mask = (cumsum / total) >= 0.85                          # (B, 128, T)
        rolloff_bin  = rolloff_mask.float().argmax(dim=1).float()        # (B, T)
        rolloff_mean = rolloff_bin.mean(dim=1)                           # (B,)

        # ── 9-10: Spectral contrast ────────────────────────────────────────
        # Energy difference between top-16 and bottom-16 mel bins per frame
        contrast      = s[:, -16:, :].mean(dim=1) - s[:, :16, :].mean(dim=1)  # (B, T)
        contrast_mean = contrast.mean(dim=1)
        contrast_std  = contrast.std(dim=1)

        # ── 11: High-frequency energy ratio ───────────────────────────────
        # Fraction of energy in the upper half of mel bins (presence/brightness)
        hf_ratio = (
            s[:, 64:, :].sum(dim=1) / (s.sum(dim=1) + eps)
        ).mean(dim=1)                                                    # (B,)

        # ── 12: Energy entropy ─────────────────────────────────────────────
        # Entropy of the normalised frame-energy distribution.
        # Low = energy concentrated in bursts; high = sustained/uniform energy.
        e_probs = frame_e / (frame_e.sum(dim=1, keepdim=True) + eps)    # (B, T)
        e_probs = e_probs.clamp(min=1e-8)
        entropy = -(e_probs * e_probs.log()).sum(dim=1)                  # (B,)

        # ── Stack → BN → MLP → L2-norm ────────────────────────────────────
        feats = torch.stack([
            rms_mean, rms_std,
            centroid_mean, centroid_std,
            flux_mean, flux_std,
            dyn_range,
            rolloff_mean,
            contrast_mean, contrast_std,
            hf_ratio,
            entropy,
        ], dim=1)                                                         # (B, 12)

        feats = self.bn(feats)
        return F.normalize(self.mlp(feats), p=2, dim=-1)                 # (B, out_dim)


class ProjectionHead(nn.Module):
    """
    Non-linear projection head used ONLY during SimCLR training — not exported to ONNX.

    SimCLR v2 design: Linear → BN → ReLU → Linear → BN → L2-norm.
    The NT-Xent loss is computed on these projections. After training, the
    encoder output (pre-projection) is used for FAISS and inference.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class CnnMusicEncoder(nn.Module):
    """
    Joint acoustic + emotion encoder for mel spectrograms.

    Input:  (B, 1, 128, T)
    Output: (B, embedding_dim) — L2-normalised joint embedding
    """

    def __init__(self, embedding_dim: int = 512, emotion_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64,  stride=2),
            ConvBlock(64,  128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
        )
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.emotion = EmotionFeatureExtractor(n_mels=128, out_dim=emotion_dim)
        # Project concatenated [acoustic | emotion] to final embedding
        self.proj    = nn.Linear(512 + emotion_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, T)
        Returns:
            (B, embedding_dim) L2-normalised joint embedding.
        """
        h = self.pool(self.features(x)).flatten(1)          # (B, 512)
        e = self.emotion(x)                                  # (B, emotion_dim), L2-normed
        return F.normalize(self.proj(torch.cat([h, e], dim=1)), p=2, dim=-1)
