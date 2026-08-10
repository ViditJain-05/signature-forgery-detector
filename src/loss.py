"""
Contrastive Loss Module for Siamese Signature Verification.
Pulls embeddings of genuine pairs close together (distance -> 0)
and pushes embeddings of forged/different pairs beyond a specified distance margin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Margin-based Contrastive Loss Function.
    Formula: Loss = 0.5 * y * dist^2 + 0.5 * (1 - y) * max(margin - dist, 0)^2
    where y=1 for same/genuine pairs, and y=0 for different/forged pairs.
    """

    def __init__(self, margin: float = 2.0):
        """
        Args:
            margin (float): Separation margin distance for negative (different/forged) pairs.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Calculates contrastive loss over a batch.

        Args:
            emb1 (torch.Tensor): Feature embeddings for first image batch (B, D).
            emb2 (torch.Tensor): Feature embeddings for second image batch (B, D).
            label (torch.Tensor): Binary tensor (B,) where 1=genuine/same, 0=forged/different.

        Returns:
            torch.Tensor: Mean contrastive loss scalar over the batch.
        """
        # Pairwise Euclidean distance
        euclidean_distance = F.pairwise_distance(emb1, emb2, p=2)

        # Contrastive loss component calculation
        loss_genuine = label * torch.pow(euclidean_distance, 2)
        loss_forged = (1.0 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        loss_contrastive = torch.mean(0.5 * (loss_genuine + loss_forged))
        return loss_contrastive
