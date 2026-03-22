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
