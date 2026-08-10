"""
Dataset and Pair Generator Module for Signature Forgery Detector.
Supports loading genuine/forged signatures, generating balanced pairs, and fallback synthetic dataset creation.
"""

import os
import random
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import get_transform


def create_sample_dataset_if_missing(data_dir: str = "data") -> str:
    """
    Creates a synthetic sample dataset in data/sample_signatures if no CEDAR dataset is found.

    Args:
        data_dir (str): Base data directory.

    Returns:
        str: Path to the usable dataset directory.
    """
    cedar_dir = os.path.join(data_dir, "CEDAR")
    if os.path.exists(os.path.join(cedar_dir, "full_org")) and os.path.exists(os.path.join(cedar_dir, "full_forg")):
        return cedar_dir

    sample_dir = os.path.join(data_dir, "sample_signatures")
    org_dir = os.path.join(sample_dir, "full_org")
    forg_dir = os.path.join(sample_dir, "full_forg")

    if os.path.exists(org_dir) and os.path.exists(forg_dir):
        return sample_dir

    os.makedirs(org_dir, exist_ok=True)
    os.makedirs(forg_dir, exist_ok=True)

    print("INFO: CEDAR dataset not found. Generating synthetic sample signature images for demonstration...")
    
    # Generate synthetic images for 10 persons
    random.seed(42)
    for person_id in range(1, 11):
        # 10 genuine signatures per person
        for sig_idx in range(1, 11):
            img = Image.new("L", (220, 155), color=255)
            draw = ImageDraw.Draw(img)
            # Draw synthetic cursive stroke specific to person_id
            points = [(20 + i*15, 75 + int(np.sin(i + person_id) * 30)) for i in range(12)]
            draw.line(points, fill=0, width=3)
            img.save(os.path.join(org_dir, f"original_{person_id}_{sig_idx}.png"))

        # 10 forged signatures per person
        for sig_idx in range(1, 11):
            img = Image.new("L", (220, 155), color=255)
            draw = ImageDraw.Draw(img)
            # Draw altered stroke pattern
            points = [(20 + i*15, 75 + int(np.cos(i * 0.5 + person_id) * 40)) for i in range(12)]
            draw.line(points, fill=50, width=2)
            img.save(os.path.join(forg_dir, f"forgeries_{person_id}_{sig_idx}.png"))

    return sample_dir


def load_signature_paths(dataset_dir: str) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Scans the dataset folder and maps Person ID to lists of genuine and forged image filepaths.

    Args:
        dataset_dir (str): Path to dataset directory containing full_org and full_forg.

    Returns:
        Tuple[Dict[int, List[str]], Dict[int, List[str]]]: (genuine_dict, forged_dict)
    """
    org_dir = os.path.join(dataset_dir, "full_org")
    forg_dir = os.path.join(dataset_dir, "full_forg")

    genuine_dict: Dict[int, List[str]] = {}
    forged_dict: Dict[int, List[str]] = {}

    if os.path.exists(org_dir):
        for fname in os.listdir(org_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                # Format: original_X_Y.png
                parts = fname.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    person_id = int(parts[1])
                    genuine_dict.setdefault(person_id, []).append(os.path.join(org_dir, fname))

    if os.path.exists(forg_dir):
        for fname in os.listdir(forg_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                # Format: forgeries_X_Y.png
                parts = fname.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    person_id = int(parts[1])
                    forged_dict.setdefault(person_id, []).append(os.path.join(forg_dir, fname))

    return genuine_dict, forged_dict


class SignaturePairDataset(Dataset):
    """
    PyTorch Dataset class producing balanced pairs of signature images.
    Returns: (img1_tensor, img2_tensor, label) where label=1 (same person genuine) and label=0 (different/forged).
    """

    def __init__(
        self,
        dataset_dir: str,
        person_ids: List[int],
        is_train: bool = True,
        pairs_per_person: int = 40,
        seed: int = 42
    ):
        """
        Args:
            dataset_dir (str): Path to root signature folder.
            person_ids (List[int]): List of person IDs assigned to this split.
            is_train (bool): Apply data augmentation if True.
            pairs_per_person (int): Target number of pairs generated per person.
            seed (int): Random seed for reproducible pair generation.
        """
        self.dataset_dir = dataset_dir
        self.person_ids = person_ids
        self.transform = get_transform(is_train=is_train)

        genuine_dict, forged_dict = load_signature_paths(dataset_dir)
        self.pairs: List[Tuple[str, str, float]] = []

        random.seed(seed)
        for pid in person_ids:
            gen_list = genuine_dict.get(pid, [])
            forg_list = forged_dict.get(pid, [])

            if len(gen_list) < 2:
                continue

            # 1. Genuine - Genuine pairs (Same person, Label = 1.0)
            num_same = pairs_per_person // 2
            for _ in range(num_same):
                img1, img2 = random.sample(gen_list, 2)
                self.pairs.append((img1, img2, 1.0))

            # 2. Genuine - Forged pairs (Same person ID, forged image, Label = 0.0)
            num_diff_forg = pairs_per_person // 4
            if forg_list:
                for _ in range(num_diff_forg):
                    img1 = random.choice(gen_list)
                    img2 = random.choice(forg_list)
                    self.pairs.append((img1, img2, 0.0))

            # 3. Genuine - Genuine cross-person pairs (Different person ID, Label = 0.0)
            other_pids = [p for p in person_ids if p != pid]
            if other_pids:
                num_diff_cross = pairs_per_person - num_same - num_diff_forg
                for _ in range(num_diff_cross):
                    other_pid = random.choice(other_pids)
                    other_gen_list = genuine_dict.get(other_pid, [])
                    if other_gen_list:
                        img1 = random.choice(gen_list)
                        img2 = random.choice(other_gen_list)
                        self.pairs.append((img1, img2, 0.0))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img1_tensor, img2_tensor, label_tensor
