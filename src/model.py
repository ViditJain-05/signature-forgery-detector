"""
Siamese Neural Network & Embedding Network Architecture for Signature Verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class EmbeddingNet(nn.Module):
    """
    Convolutional Neural Network that projects input signature images (1, 155, 220)
    into a dense 128-dimensional L2-normalized embedding space.
    """

    def __init__(self, embedding_dim: int = 128):
        super(EmbeddingNet, self).__init__()

        # Conv Block 1: Input (1, 155, 220) -> (32, 77, 110)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 2: (32, 77, 110) -> (64, 38, 55)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv Block 3: (64, 38, 55) -> (128, 19, 27)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully Connected Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(128 * 19 * 27, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim)
        )

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image tensor (B, 1, 155, 220).

        Args:
            x (torch.Tensor): Image tensor batch.

        Returns:
            torch.Tensor: L2-normalized embedding tensor of shape (B, embedding_dim).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten
        embedding = self.fc(x)
        # Normalize embeddings to unit length (L2 norm)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_single(x)


class SiameseNet(nn.Module):
    """
    Siamese Network wrapper that feeds two images through shared EmbeddingNet branches
    and returns their feature embeddings for distance computation.
    """

    def __init__(self, embedding_net: nn.Module = None):
        super(SiameseNet, self).__init__()
        if embedding_net is None:
            self.embedding_net = EmbeddingNet()
        else:
            self.embedding_net = embedding_net

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for two signature images.

        Args:
            img1 (torch.Tensor): Tensor batch of first signature (B, 1, H, W).
            img2 (torch.Tensor): Tensor batch of second signature (B, 1, H, W).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pair of feature embeddings (embedding1, embedding2).
        """
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        return emb1, emb2
