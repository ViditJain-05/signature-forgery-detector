"""
Image Preprocessing Utilities for Signature Forgery Detector.
Provides standard image loading, grayscaling, resizing, normalization, and tensor conversions.
"""

from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch

# Target image size for CEDAR signature models: (Height, Width) = (155, 220)
IMAGE_HEIGHT: int = 155
IMAGE_WIDTH: int = 220
IMAGE_SIZE: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)

try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class FallbackTransform:
    """
    Lightweight transform pipeline fallback when torchvision is not installed.
    """
    def __init__(self, target_size: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)):
        self.target_size = (target_size[1], target_size[0])  # PIL uses (width, height)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. Grayscale
        img_gray = img.convert("L")
        # 2. Resize
        img_resized = img_gray.resize(self.target_size, Image.Resampling.BILINEAR)
        # 3. Convert to numpy array & scale to [0, 1]
        arr = np.array(img_resized, dtype=np.float32) / 255.0
        # 4. Convert to Tensor (1, H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        # 5. Normalize mean=[0.5], std=[0.5] -> [-1, 1]
        tensor = (tensor - 0.5) / 0.5
        return tensor


def get_transform(is_train: bool = False):
    """
    Returns Image preprocessing transformations.
    Uses torchvision transforms if available, else FallbackTransform.
    """
    if HAS_TORCHVISION:
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]
        return transforms.Compose(transform_list)
    else:
        return FallbackTransform(IMAGE_SIZE)


def preprocess_image(image_input: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
    """
    Preprocesses a single signature image into a 4D Tensor batch (1, 1, 155, 220).

    Args:
        image_input (Union[str, Image.Image, np.ndarray]): Input image path or image object.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 1, 155, 220).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("L")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("L")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    transform = get_transform(is_train=False)
    tensor = transform(image)
    return tensor.unsqueeze(0) if tensor.dim() == 3 else tensor
