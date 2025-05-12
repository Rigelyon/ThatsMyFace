import numpy as np
from PIL import Image
import cv2
from typing import Union, Optional, Tuple, List
import base64
import io

# Constants for watermarking
BLOCK_SIZE = 8
ALPHA = 0.1  # Strength of watermark (lower means less visible)
MAX_SVD_COEFFICIENTS = 10  # Number of singular values to modify

def dct_transform(block: np.ndarray) -> np.ndarray:
    """
    Apply Discrete Cosine Transform to an image block

    Args:
        block: Image block as numpy array

    Returns:
        DCT coefficients
    """
    return cv2.dct(np.float32(block))

def idct_transform(dct_block: np.ndarray) -> np.ndarray:
    """
    Apply inverse Discrete Cosine Transform

    Args:
        dct_block: DCT coefficients

    Returns:
        Image block
    """
    return cv2.idct(dct_block)

def embed_watermark(image: Image.Image, watermark_data: bytes) -> Image.Image:
    """
    Embed encrypted watermark data into an image using DCT and SVD

    Args:
        image: PIL Image to watermark
        watermark_data: Encrypted watermark data as bytes

    Returns:
        Watermarked PIL Image
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Convert to YCrCb color space if image is RGB (we'll only modify Y channel)
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # Convert RGB to YCrCb
        img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0].copy()
    else:
        # Grayscale image
        y_channel = img_array.copy()

    # Ensure dimensions are multiples of block size
    height, width = y_channel.shape
    height_crop = height - (height % BLOCK_SIZE)
    width_crop = width - (width % BLOCK_SIZE)
    y_channel_cropped = y_channel[:height_crop, :width_crop]

    # Encode watermark data as base64 string for reliable extraction
    watermark_str = base64.b64encode(watermark_data).decode('utf-8')
    watermark_bits = ''.join(format(ord(c), '08b') for c in watermark_str)

    # Store watermark length at the beginning to know how many bits to extract later
    watermark_length_bits = format(len(watermark_bits), '032b')
    watermark_bits = watermark_length_bits + watermark_bits

    bit_index = 0
    total_bits = len(watermark_bits)

    # Process each block
    for y in range(0, height_crop, BLOCK_SIZE):
        for x in range(0, width_crop, BLOCK_SIZE):
            if bit_index >= total_bits:
                break

            # Get current block
            block = y_channel_cropped[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # Apply DCT
            dct_block = dct_transform(block)

            # Apply SVD to DCT coefficients
            U, S, Vt = np.linalg.svd(dct_block, full_matrices=True)

            # Modify singular values based on watermark bit
            if bit_index < total_bits:
                bit = int(watermark_bits[bit_index])

                # Modify the mid-range singular value(s)
                # This affects perceptibility and robustness
                mod_index = min(1, len(S) - 1)  # Use second singular value if available

                if bit == 1:
                    S[mod_index] = np.ceil(S[mod_index] / ALPHA) * ALPHA
                else:
                    S[mod_index] = np.floor(S[mod_index] / ALPHA) * ALPHA

                bit_index += 1

            # Reconstruct block
            dct_block = np.matmul(U, np.matmul(np.diag(S), Vt))

            # Apply inverse DCT
            modified_block = idct_transform(dct_block)

            # Update image with watermarked block
            y_channel_cropped[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE] = modified_block

    # Update Y channel in original image
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        img_ycrcb[:height_crop, :width_crop, 0] = y_channel_cropped
        watermarked_img = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
    else:
        watermarked_img = y_channel_cropped.copy()

    # Ensure output has same dimensions as input by padding if needed
    if height_crop < height or width_crop < width:
        if len(img_array.shape) == 3:
            out_img = img_array.copy()
            out_img[:height_crop, :width_crop, :] = watermarked_img[:height_crop, :width_crop, :]
            watermarked_img = out_img
        else:
            out_img = img_array.copy()
            out_img[:height_crop, :width_crop] = watermarked_img[:height_crop, :width_crop]
            watermarked_img = out_img

    # Convert back to PIL Image
    return Image.fromarray(watermarked_img.astype(np.uint8))

def extract_watermark(image: Image.Image) -> Optional[bytes]:
    """
    Extract watermark from a watermarked image

    Args:
        image: PIL Image with watermark

    Returns:
        Extracted watermark data as bytes, or None if extraction failed
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Convert to YCrCb color space if image is RGB
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        img_ycrcb = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
        y_channel = img_ycrcb[:, :, 0].copy()
    else:
        y_channel = img_array.copy()

    # Ensure dimensions are multiples of block size
    height, width = y_channel.shape
    height_crop = height - (height % BLOCK_SIZE)
    width_crop = width - (width % BLOCK_SIZE)
    y_channel_cropped = y_channel[:height_crop, :width_crop]

    # First extract 32 bits to determine watermark length
    extracted_bits = ""
    length_bits_needed = 32
    bit_index = 0

    # Extract length bits
    for y in range(0, height_crop, BLOCK_SIZE):
        for x in range(0, width_crop, BLOCK_SIZE):
            if bit_index >= length_bits_needed:
                break

            # Get current block
            block = y_channel_cropped[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # Apply DCT
            dct_block = dct_transform(block)

            # Apply SVD
            U, S, Vt = np.linalg.svd(dct_block, full_matrices=True)

            # Extract bit from singular value
            mod_index = min(1, len(S) - 1)
            bit = 1 if (S[mod_index] % ALPHA) >= (ALPHA / 2) else 0

            extracted_bits += str(bit)
            bit_index += 1

            if len(extracted_bits) >= length_bits_needed:
                break

        if len(extracted_bits) >= length_bits_needed:
            break

    # Parse watermark length
    try:
        watermark_length = int(extracted_bits[:length_bits_needed], 2)
    except ValueError:
        return None

    if watermark_length <= 0 or watermark_length > 1000000:  # Sanity check
        return None

    # Continue extracting watermark bits
    total_bits_needed = length_bits_needed + watermark_length

    # Reset for full extraction
    extracted_bits = ""
    bit_index = 0

    # Extract all bits
    for y in range(0, height_crop, BLOCK_SIZE):
        for x in range(0, width_crop, BLOCK_SIZE):
            if bit_index >= total_bits_needed:
                break

            # Get current block
            block = y_channel_cropped[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]

            # Apply DCT
            dct_block = dct_transform(block)

            # Apply SVD
            U, S, Vt = np.linalg.svd(dct_block, full_matrices=True)

            # Extract bit from singular value
            mod_index = min(1, len(S) - 1)
            bit = 1 if (S[mod_index] % ALPHA) >= (ALPHA / 2) else 0

            extracted_bits += str(bit)
            bit_index += 1

            if len(extracted_bits) >= total_bits_needed:
                break

        if len(extracted_bits) >= total_bits_needed:
            break

    # Parse watermark length again and extract watermark bits
    watermark_length = int(extracted_bits[:length_bits_needed], 2)
    watermark_bits = extracted_bits[length_bits_needed:length_bits_needed+watermark_length]

    # Convert bits back to base64 string
    watermark_bytes = bytearray()
    for i in range(0, len(watermark_bits), 8):
        if i + 8 <= len(watermark_bits):
            byte = int(watermark_bits[i:i+8], 2)
            watermark_bytes.append(byte)

    watermark_b64 = bytes(watermark_bytes).decode('utf-8', errors='ignore')

    try:
        # Decode base64 to get original watermark data
        watermark_data = base64.b64decode(watermark_b64)
        return watermark_data
    except Exception:
        return None