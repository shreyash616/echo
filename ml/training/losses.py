"""
NT-Xent (Normalised Temperature-scaled Cross Entropy) loss for SimCLR training.

Each batch contains B tracks, each represented by two augmented views (z_a, z_b).
The loss pulls the two views of the same track together and treats every other
track in the batch as a negative — so larger batches give stronger training signal.

temperature: controls hardness of negatives (lower = harder, try 0.07–0.2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss (SimCLR-style).

    Args:
        z_a, z_b: (B, D) L2-normalised embeddings of two augmented views.
    Returns:
        scalar NT-Xent loss.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temp = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)          # (2B, D)
        sim = torch.mm(z, z.T) / self.temp         # (2B, 2B)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        return F.cross_entropy(sim, labels)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class RegularizedNTXentLoss(nn.Module):
    """
    Hybrid NT-Xent + Covariance/Variance Regularization Loss (VICReg + SimCLR).

    Combines:
      1. NT-Xent contrastive loss (pulls views of same track together, pushes negatives apart)
      2. Covariance decorrelation loss (penalizes redundant dimensions, forcing orthogonal acoustic features)
      3. Variance regularization (prevents dimensional collapse)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        cov_coeff: float = 0.05,
        var_coeff: float = 0.02,
    ) -> None:
        super().__init__()
        self.nt_xent = NTXentLoss(temperature=temperature)
        self.cov_coeff = cov_coeff
        self.var_coeff = var_coeff

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        loss_contrast = self.nt_xent(z_a, z_b)

        z = torch.cat([z_a, z_b], dim=0)  # (2B, D)
        N, D = z.shape

        # Variance loss: standard deviation of each feature dimension should stay active
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - std_z))

        # Covariance loss: off-diagonal elements of covariance matrix should be 0
        z_centered = z - z.mean(dim=0)
        cov_z = (z_centered.T @ z_centered) / (N - 1)
        cov_loss = _off_diagonal(cov_z).pow_(2).sum() / D

        total_loss = loss_contrast + (self.cov_coeff * cov_loss) + (self.var_coeff * var_loss)
        return total_loss
