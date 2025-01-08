from PIL import Image
from typing import List, Dict, Optional, Union, Tuple, Iterable
import torch
import numpy as np


def resize(image: Image.Image, size: Tuple[int, int], resample: Image.Resampling = None, reducing_gap: Optional[int] = None):
    H, W = size
    resized_image = image.resize((W, H), resample=resample, reducing_gap=reducing_gap)
    return resized_image


def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32):
    return (image * scale).astype(dtype)


def normalize(image: np.ndarray, std: Union[float, Iterable [float]], mean: Union[float, Iterable[float]]):
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    return (image - mean) / std


def preprocess_image(
    images: List[Image.Image],
    image_size: Dict[str, int],
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
):
    H, W = image_size["height"], image_size["width"]
    # Resize the image to the desired size
    images = [resize(image, (H, W), resample=resample) for image in images]
    # Convert the image to a numpy array
    images = [np.array(image) for image in images]
    # Rescale the image
    images = [rescale(image, scale=rescale_factor) for image in images]
    # Normalize the image
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    # (H, W, C) -> (C, H, W)
    images = [image.transpose(2, 0, 1) for image in images]
    return images
