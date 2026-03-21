"""
CnnMusicEncoder — custom CNN that maps mel spectrograms to 512-d embeddings.

Architecture:
  Input:  (batch, 1, 128, T) mel spectrogram — T varies with clip length.
  Conv blocks with increasing channels: 1→32→64→128→256→512.
  AdaptiveAvgPool2d((1,1)) collapses the spatial dimensions regardless of T.
  Linear projection → 512-d L2-normalised embedding.

Training:  TripletLoss with online hard mining on FMA genre labels.
Inference: exported to ONNX for fast CPU inference in the FastAPI backend.
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


class CnnMusicEncoder(nn.Module):
    """
    CNN encoder for mel spectrograms.

    Input:  (B, 1, 128, T)  — single-channel mel spectrogram
    Output: (B, embedding_dim)  — L2-normalised embedding
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
            ConvBlock(32,  64,  stride=2),
            ConvBlock(64,  128, stride=2),
            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 512, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 128, T) mel spectrogram — T can be any length.
        Returns:
            (B, embedding_dim) L2-normalised embedding.
        """
        h = self.features(x)           # (B, 512, H', T')
        h = self.pool(h).flatten(1)    # (B, 512)
        return F.normalize(self.proj(h), p=2, dim=-1)
