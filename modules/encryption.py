from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from typing import Union, Optional

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
