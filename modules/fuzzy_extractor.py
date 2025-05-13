import numpy as np
import pickle
import os
import hashlib
import io
from typing import Tuple, Optional, Dict, Any
from fuzzy_extractor import FuzzyExtractor

def vector_to_binary(embedding: np.ndarray, threshold: float = 0.0) -> bytes:
    """
    Convert a face embedding vector to binary representation
    by thresholding the values.

    Args:
        embedding: Face embedding vector
        threshold: Threshold value (default: mean of the vector)

    Returns:
        Binary representation as bytes
    """
    if threshold == 0.0:
        threshold = np.mean(embedding)

    # Create binary representation (1 for values >= threshold, 0 otherwise)
    binary = (embedding >= threshold).astype(int)

    # Convert to bytes (8 bits per byte)
    result = bytearray()
    for i in range(0, len(binary), 8):
        chunk = binary[i:i+8]
        # Pad with zeros if necessary
        if len(chunk) < 8:
            chunk = np.pad(chunk, (0, 8 - len(chunk)))
        byte_val = 0
        for bit_idx, bit in enumerate(chunk):
            byte_val |= (bit << (7 - bit_idx))
        result.append(byte_val)

    return bytes(result)

def create_fuzzy_extractor(embedding: np.ndarray,
                           error_tolerance: int = 50) -> Tuple[bytes, Dict[str, Any]]:
    """
    Create a fuzzy extractor from a face embedding vector.

    Args:
        embedding: Face embedding vector
        error_tolerance: Number of bit errors to tolerate

    Returns:
        Tuple of (key, helper_data)
    """
    # Normalize embedding vector
    norm_embedding = embedding / np.linalg.norm(embedding)

    # Convert to binary representation
    binary_data = vector_to_binary(norm_embedding)

    # Create fuzzy extractor
    extractor = FuzzyExtractor(n_bits=len(binary_data) * 8,
                               t_error=error_tolerance)

    # Generate key and helper data
    key, helper_data = extractor.generate(binary_data)

    # Hash the key to get a fixed-length encryption key
    key_hash = hashlib.sha256(key).digest()

    # Prepare helper data dictionary
    helper_dict = {
        'helper_data': helper_data,
        'error_tolerance': error_tolerance,
        'vector_shape': embedding.shape,
        'vector_mean': float(np.mean(embedding)),
        'vector_std': float(np.std(embedding))
    }

    return key_hash, helper_dict

def reproduce_key_from_embedding(embedding: np.ndarray,
                                 helper_dict: Dict[str, Any]) -> Optional[bytes]:
    """
    Reproduce the key from a face embedding and helper data.

    Args:
        embedding: Face embedding vector
        helper_dict: Helper data dictionary from create_fuzzy_extractor

    Returns:
        Reproduced key or None if reproduction fails
    """
    try:
        # Normalize embedding vector
        norm_embedding = embedding / np.linalg.norm(embedding)

        # Convert to binary representation
        binary_data = vector_to_binary(norm_embedding)

        # Create fuzzy extractor with same parameters
        extractor = FuzzyExtractor(n_bits=len(binary_data) * 8,
                                   t_error=helper_dict['error_tolerance'])

        # Reproduce key from helper data
        key = extractor.reproduce(binary_data, helper_dict['helper_data'])

        if key is not None:
            # Hash the key to get consistent encryption key
            key_hash = hashlib.sha256(key).digest()
            return key_hash

        return None
    except Exception as e:
        print(f"Key reproduction error: {str(e)}")
        return None

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
        with open(path, 'wb') as f:
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
        with open(path, 'rb') as f:
            helper_bytes = f.read()
        return deserialize_helper_data(helper_bytes)
    except Exception as e:
        print(f"Error loading helper data: {str(e)}")
        return None