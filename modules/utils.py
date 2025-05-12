import os
import numpy as np
from PIL import Image
import io
from typing import Union, Optional, Tuple

def load_image(path: str) -> Optional[Image.Image]:
    """
    Load an image from file path

    Args:
        path: Path to the image file

    Returns:
        PIL Image object or None if loading fails
    """
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return None

def save_image(image: Image.Image, path: str) -> bool:
    """
    Save PIL Image to a file

    Args:
        image: PIL Image object
        path: Path to save the image

    Returns:
        True if saving successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save image
        image.save(path)
        return True
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return False

def convert_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes

    Args:
        image: PIL Image object
        format: Image format (default: PNG)

    Returns:
        Image as bytes
    """
    img_buffer = io.BytesIO()
    image.save(img_buffer, format=format)
    return img_buffer.getvalue()

def bytes_to_image(image_bytes: bytes) -> Optional[Image.Image]:
    """
    Convert bytes to PIL Image

    Args:
        image_bytes: Image as bytes

    Returns:
        PIL Image object or None if conversion fails
    """
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to 0-1 range

    Args:
        image: Image as numpy array

    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image

def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from 0-1 range to 0-255 range

    Args:
        image: Normalized image as numpy array

    Returns:
        Denormalized image
    """
    if image.max() <= 1.0:
        return (image * 255.0).astype(np.uint8)
    return image.astype(np.uint8)

def ensure_valid_pixel_values(image: np.ndarray) -> np.ndarray:
    """
    Ensure pixel values are within valid range (0-255)

    Args:
        image: Image as numpy array

    Returns:
        Image with valid pixel values
    """
    return np.clip(image, 0, 255).astype(np.uint8)

def is_image_valid(image: Optional[Union[Image.Image, np.ndarray]]) -> bool:
    """
    Check if image is valid (not None and has proper dimensions)

    Args:
        image: PIL Image or numpy array

    Returns:
        True if image is valid, False otherwise
    """
    if image is None:
        return False

    if isinstance(image, Image.Image):
        return image.width > 0 and image.height > 0

    elif isinstance(image, np.ndarray):
        return image.size > 0 and len(image.shape) >= 2

    return False