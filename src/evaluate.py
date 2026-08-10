"""
Evaluation & ROC / EER Analysis Script for Signature Forgery Detector.
Computes verification distances, ROC curves, Equal Error Rate (EER), optimal threshold, and distance distribution plots.
"""

import os
import sys
import argparse
from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import create_sample_dataset_if_missing, load_signature_paths, SignaturePairDataset
from src.model import SiameseNet, EmbeddingNet


def calculate_eer(labels: np.ndarray, distances: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Equal Error Rate (EER) and optimal distance threshold.
    """
    scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    optimal_threshold = float(-thresholds[eer_idx])

    return eer, optimal_threshold, fpr, tpr, thresholds


def plot_evaluation_results(
    labels: np.ndarray,
    distances: np.ndarray,
    eer: float,
    optimal_threshold: float,
    fpr: np.ndarray,
    tpr: np.ndarray,
    save_dir: str = "assets"
):
    """
    Generates and saves evaluation charts: Distance Distribution Histogram and ROC Curve.
    """
    if not HAS_MATPLOTLIB:
        print("INFO: matplotlib is not installed. Skipping chart PNG generation.")
        return

    os.makedirs(save_dir, exist_ok=True)
    gen_distances = distances[labels == 1]
    forg_distances = distances[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))


    # 1. Distance Distribution Histogram
    ax1.hist(gen_distances, bins=30, alpha=0.6, color="green", label="Genuine Pairs (Label=1)", density=True)
    ax1.hist(forg_distances, bins=30, alpha=0.6, color="red", label="Forged / Diff Pairs (Label=0)", density=True)
    ax1.axvline(optimal_threshold, color="black", linestyle="--", linewidth=2, label=f"Optimal Thresh ({optimal_threshold:.2f})")
    ax1.set_xlabel("Euclidean Distance")
    ax1.set_ylabel("Density")
    ax1.set_title("Signature Pair Embedding Distance Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. ROC Curve
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Baseline")
    ax2.scatter(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))], tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))],
                color="red", s=100, zorder=5, label=f"EER = {eer*100:.1f}%")
    ax2.set_xlabel("False Positive Rate (FPR)")
    ax2.set_ylabel("True Positive Rate (TPR)")
    ax2.set_title("Verification Receiver Operating Characteristic (ROC)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "distance_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Evaluation charts saved to '{plot_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Siamese Signature Verification Model")
    parser.add_argument("--model-path", type=str, default="models/siamese_model.pt", help="Path to trained model weights")
    args = parser.parse_args()

    dataset_dir = create_sample_dataset_if_missing("data")
    genuine_dict, _ = load_signature_paths(dataset_dir)
    all_person_ids = sorted(list(genuine_dict.keys()))

    # Use validation / test split person IDs
    split_idx = max(1, int(len(all_person_ids) * 0.8))
    test_person_ids = all_person_ids[split_idx:] if split_idx < len(all_person_ids) else all_person_ids

    test_dataset = SignaturePairDataset(dataset_dir, test_person_ids, is_train=False, pairs_per_person=30)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNet(EmbeddingNet()).to(device)

    if not os.path.exists(args.model_path):
        print(f"WARNING: Model file '{args.model_path}' not found. Initializing un-trained baseline model for evaluation.")
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded trained model weights from '{args.model_path}'.")

    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            emb1, emb2 = model(img1, img2)
            dists = F.pairwise_distance(emb1, emb2).cpu().numpy()
            all_distances.extend(dists)
            all_labels.extend(label.numpy())

    distances = np.array(all_distances)
    labels = np.array(all_labels)

    eer, optimal_threshold, fpr, tpr, _ = calculate_eer(labels, distances)
    acc_at_threshold = np.mean((distances <= optimal_threshold) == (labels == 1))

    print("\n" + "=" * 50)
    print("      SIAMESE MODEL EVALUATION METRICS       ")
    print("=" * 50)
    print(f"Total Test Pair Samples     : {len(labels)}")
    print(f"Genuine Pairs Count         : {np.sum(labels == 1)}")
    print(f"Forged / Diff Pairs Count   : {np.sum(labels == 0)}")
    print(f"Equal Error Rate (EER)      : {eer * 100:.2f}%")
    print(f"Optimal Distance Threshold  : {optimal_threshold:.4f}")
    print(f"Accuracy at Threshold       : {acc_at_threshold * 100:.2f}%")
    print("=" * 50)

    plot_evaluation_results(labels, distances, eer, optimal_threshold, fpr, tpr)


if __name__ == "__main__":
    main()
