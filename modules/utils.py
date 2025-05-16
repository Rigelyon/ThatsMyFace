import io
import os
import pickle
from typing import Union, Optional, Dict, Any

import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace

from modules.constants import FACE_DETECTION_MODEL


def load_image(path: str) -> Optional[Image.Image]:
    """
    Load an image from file path

    Args:
        path: Path to the image file

    Returns:
        PIL Image object or None if loading fails
    """
    try:
        return Image.open(path).convert("RGB")
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


def convert_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
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


def has_face(image):
    """
    Analyzes the given image and detects if it contains any human faces.

    Args:
        image: Input image to be analyzed. It should be compatible with image processing libraries like a PIL Image or NumPy array.

    Returns:
        True if one or more human faces are detected, False otherwise.
    """
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        faces = DeepFace.extract_faces(
            image, detector_backend=FACE_DETECTION_MODEL, enforce_detection=False
        )
        return len(faces) > 0
    except Exception:
        st.error("Error detecting faces.")
        return False


def serialize_embedding(embedding):
    """
    Convert a numpy array embedding to downloadable bytes

    Args:
        embedding: Numpy array containing the face embedding

    Returns:
        Bytes representation of the embedding
    """
    buffer = io.BytesIO()
    np.save(buffer, embedding)
    return buffer.getvalue()


def serialize_helper_data(helper_dict: Dict[str, Any]) -> bytes:
    """
    Serialize helper data to bytes for storage.

    Args:
        helper_dict: Helper data dictionary

    Returns:
        Serialized helper data as bytes
    """
    buffer = io.BytesIO()
    pickle.dump(helper_dict, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def deserialize_helper_data(helper_bytes: bytes) -> Dict[str, Any]:
    """
    Deserialize helper data from bytes.

    Args:
        helper_bytes: Serialized helper data

    Returns:
        Helper data dictionary
    """
    buffer = io.BytesIO(helper_bytes)
    buffer.seek(0)
    return pickle.load(buffer)


def save_helper_data(helper_dict: Dict[str, Any], path: str) -> bool:
    """
    Save helper data to a file.

    Args:
        helper_dict: Helper data dictionary
        path: Path to save the helper data

    Returns:
        True if saving was successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Serialize and save
        helper_bytes = serialize_helper_data(helper_dict)
        with open(path, "wb") as f:
            f.write(helper_bytes)
        return True
    except Exception as e:
        print(f"Error saving helper data: {str(e)}")
        return False


def load_helper_data(path: str) -> Optional[Dict[str, Any]]:
    """
    Load helper data from a file.

    Args:
        path: Path to the helper data file

    Returns:
        Helper data dictionary or None if loading fails
    """
    try:
        with open(path, "rb") as f:
            helper_bytes = f.read()
        return deserialize_helper_data(helper_bytes)
    except Exception as e:
        print(f"Error loading helper data: {str(e)}")
        return None


def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Convert bytes to bit array.

    Args:
        data: Bytes to convert

    Returns:
        Numpy array of bits (0s and 1s)
    """
    result = []
    for byte in data:
        # Convert each byte to 8 bits
        for i in range(8):
            result.append((byte >> (7 - i)) & 1)
    return np.array(result, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array to bytes.

    Args:
        bits: Numpy array of bits (0s and 1s)

    Returns:
        Bytes representation
    """
    # Ensure length is multiple of 8
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - (len(bits) % 8)))

    result = bytearray()
    for i in range(0, len(bits), 8):
        byte_val = 0
        for bit_idx, bit in enumerate(bits[i : i + 8]):
            byte_val |= bit << (7 - bit_idx)
        result.append(byte_val)

    return bytes(result)
