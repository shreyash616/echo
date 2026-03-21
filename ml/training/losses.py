"""
Triplet loss with online hard mining for metric learning.

Given a batch of (anchor, positive, negative) triplets:
  - Anchor and positive share the same genre cluster.
  - Negative is from a different cluster.

We use online hard mining to select the hardest negatives within
each batch — this dramatically speeds up convergence compared to
random triplet selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Online hard triplet loss with cosine distance.

    For each anchor i:
      - Hard positive:  sample j in the same class with the LOWEST similarity.
      - Hard negative:  sample k in a different class with the HIGHEST similarity.

    Loss = max(d(a, p) - d(a, n) + margin, 0)
    where d is 1 - cosine_similarity (lower = more similar).
    """

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) L2-normalised embeddings.
            labels:     (B,) integer class labels (e.g. genre cluster id).

        Returns:
            loss:           scalar mean triplet loss.
            frac_positive:  fraction of triplets with loss > 0 (training signal).
        """
        # Pairwise cosine similarity matrix  (B, B)
        sim = torch.mm(embeddings, embeddings.T)  # (B, B)
        # Cosine distance: 1 - sim
        dist = 1.0 - sim  # (B, B), in [0, 2]

        # Build masks
        labels_row = labels.unsqueeze(1)  # (B, 1)
        labels_col = labels.unsqueeze(0)  # (1, B)
        same_class = labels_row == labels_col          # (B, B)
        diff_class = ~same_class
        eye = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        valid_positive = same_class & ~eye             # exclude self
        valid_negative = diff_class

        # Hard positive: furthest same-class sample
        # Mask out invalid positions with 0 (smallest distance → won't be picked as max)
        pos_dist = dist.masked_fill(~valid_positive, 0.0)
        hard_pos, _ = pos_dist.max(dim=1)  # (B,)

        # Hard negative: closest different-class sample
        # Mask out invalid positions with large distance → won't be picked as min
        neg_dist = dist.masked_fill(~valid_negative, 1e9)
        hard_neg, _ = neg_dist.min(dim=1)  # (B,)

        raw_loss = hard_pos - hard_neg + self.margin
        loss = F.relu(raw_loss).mean()
        frac_positive = (raw_loss > 0).float().mean()

        return loss, frac_positive


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy loss (SimCLR-style).
    Treats each pair (augmented_a, augmented_b) of the same track as positive.
    Useful when you have two augmented views of each song snippet.

    temperature: lower = harder negatives (try 0.07–0.3).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temp = temperature

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_a, z_b: (B, D) L2-normalised embeddings of two augmented views.
        Returns:
            scalar NT-Xent loss.
        """
        B = z_a.size(0)
        z = torch.cat([z_a, z_b], dim=0)          # (2B, D)
        sim = torch.mm(z, z.T) / self.temp        # (2B, 2B)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(z.device)

        # Exclude self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        return F.cross_entropy(sim, labels)
