import numpy as np
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from typing import Union, Optional, Dict, Any
from modules.fuzzy_extractor import create_fuzzy_extractor, reproduce_key_from_embedding

def generate_key_from_embedding(embedding: np.ndarray) -> bytes:
    """
    Generate an AES encryption key from a face embedding vector

    Args:
        embedding: Face embedding vector

    Returns:
        32-byte key for AES-256 encryption
    """
    # Normalize embedding vector
    norm_embedding = embedding / np.linalg.norm(embedding)

    # Convert to bytes and hash to get a fixed-length key
    embedding_bytes = norm_embedding.tobytes()
    key = hashlib.sha256(embedding_bytes).digest()

    return key

def generate_key_with_helper(embedding: np.ndarray, error_tolerance: int = 50) -> tuple[bytes, Dict[str, Any]]:
    """
    Generate an AES encryption key from a face embedding vector using fuzzy extractor

    Args:
        embedding: Face embedding vector
        error_tolerance: Number of bit errors to tolerate

    Returns:
        tuple of (32-byte key for AES-256 encryption, helper data)
    """
    return create_fuzzy_extractor(embedding, error_tolerance)

def regenerate_key_from_helper(embedding: np.ndarray, helper_data: Dict[str, Any]) -> Optional[bytes]:
    """
    Regenerate the AES encryption key from a face embedding and helper data

    Args:
        embedding: Face embedding vector
        helper_data: Helper data from generate_key_with_helper

    Returns:
        32-byte key for AES-256 encryption or None if regeneration fails
    """
    return reproduce_key_from_embedding(embedding, helper_data)

def encrypt_watermark(watermark_data: Union[str, bytes], key: bytes) -> bytes:
    """
    Encrypt watermark data using AES with the given key

    Args:
        watermark_data: Watermark data as string or bytes
        key: 32-byte encryption key

    Returns:
        Encrypted watermark data
    """
    # Ensure watermark_data is bytes
    if isinstance(watermark_data, str):
        watermark_bytes = watermark_data.encode('utf-8')
    else:
        watermark_bytes = watermark_data

    # Generate IV (initialization vector)
    iv = get_random_bytes(16)

    # Create AES cipher in CBC mode
    cipher = AES.new(key, AES.MODE_CBC, iv)

    # Pad the data to be a multiple of 16 bytes (AES block size)
    padded_data = pad(watermark_bytes, AES.block_size)

    # Encrypt the data
    encrypted_data = cipher.encrypt(padded_data)

    # Combine IV and encrypted data
    result = iv + encrypted_data

    return result

def decrypt_watermark(encrypted_data: bytes, key: bytes) -> Optional[bytes]:
    """
    Decrypt watermark data using AES with the given key

    Args:
        encrypted_data: Encrypted watermark data
        key: 32-byte encryption key

    Returns:
        Decrypted watermark data or None if decryption fails
    """
    try:
        # Extract IV (first 16 bytes)
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        # Create AES cipher in CBC mode
        cipher = AES.new(key, AES.MODE_CBC, iv)

        # Decrypt and unpad
        decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)

        return decrypted_data
    except Exception as e:
        print(f"Decryption error: {str(e)}")
        return None
