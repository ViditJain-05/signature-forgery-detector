"""
Training Script for Signature Forgery Detector Siamese Neural Network.
"""

import os
import sys
import argparse
from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import create_sample_dataset_if_missing, load_signature_paths, SignaturePairDataset
from src.model import SiameseNet, EmbeddingNet
from src.loss import ContrastiveLoss


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Executes one training epoch over the training dataloader.

    Returns:
        Tuple[float, float, float]: (average_loss, avg_dist_genuine, avg_dist_forged)
    """
    model.train()
    running_loss = 0.0
    dist_genuine_list: List[float] = []
    dist_forged_list: List[float] = []

    for img1, img2, label in dataloader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)

        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img1.size(0)
        distances = F.pairwise_distance(emb1, emb2).detach().cpu().numpy()
        labels = label.cpu().numpy()

        dist_genuine_list.extend(distances[labels == 1])
        dist_forged_list.extend(distances[labels == 0])

    epoch_loss = running_loss / len(dataloader.dataset)
    avg_dist_gen = float(np.mean(dist_genuine_list)) if dist_genuine_list else 0.0
    avg_dist_forg = float(np.mean(dist_forged_list)) if dist_forged_list else 0.0

    return epoch_loss, avg_dist_gen, avg_dist_forg


def validate_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluates model performance over the validation dataloader.

    Returns:
        Tuple[float, float, float]: (val_loss, avg_dist_genuine, avg_dist_forged)
    """
    model.eval()
    running_loss = 0.0
    dist_genuine_list: List[float] = []
    dist_forged_list: List[float] = []

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)

            running_loss += loss.item() * img1.size(0)
            distances = F.pairwise_distance(emb1, emb2).cpu().numpy()
            labels = label.cpu().numpy()

            dist_genuine_list.extend(distances[labels == 1])
            dist_forged_list.extend(distances[labels == 0])

    val_loss = running_loss / len(dataloader.dataset)
    avg_dist_gen = float(np.mean(dist_genuine_list)) if dist_genuine_list else 0.0
    avg_dist_forg = float(np.mean(dist_forged_list)) if dist_forged_list else 0.0

    return val_loss, avg_dist_gen, avg_dist_forg


def main():
    parser = argparse.ArgumentParser(description="Train Siamese Network for Signature Verification")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--margin", type=float, default=2.0, help="Contrastive loss margin")
    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # Check or create dataset
    dataset_dir = create_sample_dataset_if_missing("data")
    genuine_dict, _ = load_signature_paths(dataset_dir)
    all_person_ids = sorted(list(genuine_dict.keys()))

    if not all_person_ids:
        print("ERROR: No signature images found in dataset directory!")
        sys.exit(1)

    # Stratified Person ID split (80% train, 20% val)
    split_idx = max(1, int(len(all_person_ids) * 0.8))
    train_person_ids = all_person_ids[:split_idx]
    val_person_ids = all_person_ids[split_idx:] if split_idx < len(all_person_ids) else train_person_ids

    print(f"Dataset: '{dataset_dir}' | Total Persons: {len(all_person_ids)}")
    print(f"Train Persons: {len(train_person_ids)} | Val Persons: {len(val_person_ids)}")

    train_dataset = SignaturePairDataset(dataset_dir, train_person_ids, is_train=True, pairs_per_person=30)
    val_dataset = SignaturePairDataset(dataset_dir, val_person_ids, is_train=False, pairs_per_person=20)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SiameseNet(EmbeddingNet()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = ContrastiveLoss(margin=args.margin)

    best_val_loss = float("inf")
    model_save_path = os.path.join("models", "siamese_model.pt")

    print("\nStarting Training Loop...")
    print("-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Gen Dist (Val)':<15} | {'Forg Dist (Val)':<15}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_g_dist, train_f_dist = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_g_dist, val_f_dist = validate_epoch(model, val_loader, criterion, device)

        print(f"{epoch:<6d} | {train_loss:<10.4f} | {val_loss:<10.4f} | {val_g_dist:<15.4f} | {val_f_dist:<15.4f}")

        # Checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)

    print("-" * 75)
    print(f"SUCCESS: Training complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Best model weights saved to '{model_save_path}'.")


if __name__ == "__main__":
    main()
