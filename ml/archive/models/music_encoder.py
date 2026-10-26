"""
CnnMusicEncoder — CNN acoustic backbone + EmotionFeatureExtractor → joint embedding.

Architecture
------------
Acoustic path:  (B, 1, 128, T) → ConvBlocks → Temporal Mean+Std Pool → 1024-d
Emotion path:   (B, 1, 128, T) → 16 differentiable emotion & rhythm features → BN → MLP → 64-d (L2-norm)
Joint:          cat(1024, 64) → Linear → embedding_dim → L2-norm

EmotionFeatureExtractor
-----------------------
Computes 16 differentiable features that capture emotional intensity, tonal brightness,
and rhythmic beat/tempo qualities directly from the mel spectrogram:

  1.  rms_mean         — overall energy level (loud vs quiet)
  2.  rms_std          — energy variation over time (dynamic vs flat)
  3.  centroid_mean    — spectral brightness (harsh/bright vs warm/mellow)
  4.  centroid_std     — brightness consistency
  5.  flux_mean        — mean spectral flux (adrenaline/aggression vs smoothness)
  6.  flux_std         — variation in spectral flux (erratic vs steady)
  7.  dynamic_range    — robust 90th-to-10th percentile energy swing (emotional intensity)
  8.  rolloff_mean     — normalised freq where 85% energy is contained (tonal weight)
  9.  contrast_mean    — high vs low mel band energy difference (fullness)
  10. contrast_std     — variation in spectral contrast
  11. hf_ratio         — high-frequency energy fraction (presence/brightness)
  12. energy_entropy   — length-normalised entropy of frame energy in [0, 1] (sustained vs burst)
  13. onset_density    — fraction of frames with active rhythmic onsets (busyness)
  14. pulse_clarity    — onset autocorrelation peak strength (driving beat vs ambient)
  15. tempo_fast_ratio — fast-beat (120–200 BPM) vs slow-beat (65–115 BPM) rhythmic ratio
  16. bass_ratio       — sub-bass & kick energy fraction in lowest 24 mel bins

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


class ResidualBlock(nn.Module):
    """
    Residual acoustic block with pre-activation shortcut:
      Conv2d(3x3, stride) -> BN -> GELU -> Conv2d(3x3) -> BN + shortcut -> GELU
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class EmotionFeatureExtractor(nn.Module):
    """
    Differentiable extractor of 16 emotion, tonal, and rhythm/beat features from a mel spectrogram.
    Projects them to a normalised emotion embedding via BN → MLP → L2-norm.
    """

    N_FEATS = 16

    def __init__(self, n_mels: int = 128, out_dim: int = 64) -> None:
        super().__init__()
        self.n_mels = n_mels
        # Mel bin indices normalised to [0, 1] for scale consistency
        self.register_buffer(
            "mel_idx",
            torch.arange(1, n_mels + 1, dtype=torch.float32) / float(n_mels),  # (n_mels,)
        )
        # Normalise raw features before MLP (handles scale differences across features)
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
            (B, out_dim) L2-normalised emotion & rhythm embedding.
        """
        s   = x.squeeze(1)   # (B, 128, T)
        eps = 1e-8
        T   = s.shape[2]

        # ── 1-2: RMS energy per frame ──────────────────────────────────────
        frame_e  = s.mean(dim=1)           # (B, T)
        rms_mean = frame_e.mean(dim=1)     # (B,)
        rms_std  = frame_e.std(dim=1)      # (B,)

        # ── 3-4: Spectral centroid (normalised to [0, 1]) ──────────────────
        idx      = self.mel_idx.view(1, -1, 1)                          # (1, 128, 1)
        centroid = (s * idx).sum(dim=1) / (s.sum(dim=1) + eps)          # (B, T)
        centroid_mean = centroid.mean(dim=1)
        centroid_std  = centroid.std(dim=1)

        # ── 5-6: Spectral flux ─────────────────────────────────────────────
        # L1 frame-to-frame change — high = adrenaline-y, low = soothing
        diff      = s[:, :, 1:] - s[:, :, :-1]                          # (B, 128, T-1)
        flux      = diff.abs().mean(dim=1)                              # (B, T-1)
        flux_mean = flux.mean(dim=1)
        flux_std  = flux.std(dim=1)

        # ── 7: Robust dynamic range (top 10% minus bottom 10% frame energy) ─
        # Immune to single-frame dropouts or masks unlike raw max - min
        k_pct     = max(1, T // 10)
        top_e     = torch.topk(frame_e, k=k_pct, dim=1, largest=True).values.mean(dim=1)
        bot_e     = torch.topk(frame_e, k=k_pct, dim=1, largest=False).values.mean(dim=1)
        dyn_range = top_e - bot_e                                       # (B,)

        # ── 8: Spectral rolloff (normalised to [0, 1]) ─────────────────────
        cumsum       = s.cumsum(dim=1)                                  # (B, 128, T)
        total        = cumsum[:, -1:, :] + eps                          # (B, 1, T)
        rolloff_mask = (cumsum / total) >= 0.85                         # (B, 128, T)
        rolloff_bin  = rolloff_mask.float().argmax(dim=1).float() / float(self.n_mels)
        rolloff_mean = rolloff_bin.mean(dim=1)                          # (B,)

        # ── 9-10: Spectral contrast ────────────────────────────────────────
        contrast      = s[:, -16:, :].mean(dim=1) - s[:, :16, :].mean(dim=1)  # (B, T)
        contrast_mean = contrast.mean(dim=1)
        contrast_std  = contrast.std(dim=1)

        # ── 11: High-frequency energy ratio ───────────────────────────────
        hf_ratio = (
            s[:, 64:, :].sum(dim=1) / (s.sum(dim=1) + eps)
        ).mean(dim=1)                                                   # (B,)

        # ── 12: Length-normalised energy entropy in [0, 1] ─────────────────
        # Dividing by log(T) makes entropy invariant to clip duration T.
        e_probs = frame_e / (frame_e.sum(dim=1, keepdim=True) + eps)   # (B, T)
        e_probs = e_probs.clamp(min=1e-8)
        log_T   = torch.log(
            torch.tensor(T, dtype=s.dtype, device=s.device).clamp(min=2.0)
        )
        entropy = -(e_probs * e_probs.log()).sum(dim=1) / log_T         # (B,)

        # ── 13-15: Rhythm, Beat & Tempo features ───────────────────────────
        # Half-wave rectified onset envelope from lower 80 rhythmic mel bands
        onset_env = F.relu(diff[:, :80, :]).mean(dim=1)                 # (B, T-1)
        onset_thresh = onset_env.mean(dim=1, keepdim=True) + 0.5 * onset_env.std(dim=1, keepdim=True)
        onset_density = (onset_env > onset_thresh).float().mean(dim=1)  # (B,)

        # Multi-lag onset autocorrelation for beat pulse clarity and fast/slow tempo ratio
        onset_c = onset_env - onset_env.mean(dim=1, keepdim=True)
        onset_var = (onset_c * onset_c).mean(dim=1) + eps               # (B,)

        # Fast-tempo lags (~125–195 BPM at 43.07 fps -> lags 13, 16, 19, 21)
        fast_lags = [13, 16, 19, 21]
        # Slow-tempo lags (~70–115 BPM at 43.07 fps -> lags 24, 28, 32, 36)
        slow_lags = [24, 28, 32, 36]

        fast_corrs = torch.stack(
            [(onset_c[:, :-L] * onset_c[:, L:]).mean(dim=1) / onset_var for L in fast_lags],
            dim=1,
        )  # (B, 4)
        slow_corrs = torch.stack(
            [(onset_c[:, :-L] * onset_c[:, L:]).mean(dim=1) / onset_var for L in slow_lags],
            dim=1,
        )  # (B, 4)

        all_corrs = torch.cat([fast_corrs, slow_corrs], dim=1)          # (B, 8)
        pulse_clarity = F.relu(all_corrs.max(dim=1).values)             # (B,)

        fast_e = F.relu(fast_corrs).mean(dim=1)
        slow_e = F.relu(slow_corrs).mean(dim=1)
        tempo_fast_ratio = fast_e / (fast_e + slow_e + eps)             # (B,)

        # ── 16: Sub-bass & kick energy fraction (lowest 24 mel bins) ───────
        bass_ratio = (
            s[:, :24, :].sum(dim=1) / (s.sum(dim=1) + eps)
        ).mean(dim=1)                                                   # (B,)

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
            onset_density,
            pulse_clarity,
            tempo_fast_ratio,
            bass_ratio,
        ], dim=1)                                                       # (B, 16)

        feats = self.bn(feats)
        return F.normalize(self.mlp(feats), p=2, dim=-1)                # (B, out_dim)


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
    Joint acoustic ResNet + emotion/rhythm encoder for mel spectrograms.

    Input:  (B, 1, 128, T)
    Output: (B, embedding_dim) — L2-normalised joint embedding
    """

    def __init__(self, embedding_dim: int = 512, emotion_dim: int = 64) -> None:
        super().__init__()
        # Initial 7x7 wide receptive field stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        # Deep Residual acoustic stages with skip connections
        self.features = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128, stride=1),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256, stride=1),
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512, stride=1),
        )
        self.emotion = EmotionFeatureExtractor(n_mels=128, out_dim=emotion_dim)
        # Pool both mean and temporal std of residual feature maps (512 + 512) + emotion_dim
        self.proj    = nn.Linear(512 * 2 + emotion_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, T)
        Returns:
            (B, embedding_dim) L2-normalised joint embedding.
        """
        out      = self.stem(x)
        feat_map = self.features(out)                       # (B, 512, H', T')
        # Mean over frequency, then capture both mean and temporal std over time
        freq_mean = feat_map.mean(dim=2)                    # (B, 512, T')
        h_mean    = freq_mean.mean(dim=2)                   # (B, 512)
        h_std     = freq_mean.std(dim=2)                    # (B, 512)
        e         = self.emotion(x)                         # (B, emotion_dim), L2-normed
        joint     = torch.cat([h_mean, h_std, e], dim=1)    # (B, 1024 + emotion_dim)
        return F.normalize(self.proj(joint), p=2, dim=-1)