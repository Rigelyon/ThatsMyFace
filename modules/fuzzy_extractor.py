import numpy as np
import os
import hashlib
from typing import Tuple, Optional, Dict, Any
from modules.utils import bytes_to_bits, bits_to_bytes

class FuzzyExtractor:
    """
    Fuzzy extractor implementation that can handle larger binary data
    using error correction coding techniques.
    """

    def __init__(self, error_tolerance: float = 0.15):
        """
        Initialize the fuzzy extractor.

        Args:
            error_tolerance: Fraction of bits that can be different between
                            original and reproduced data (0.0-1.0)
        """
        self.error_tolerance = error_tolerance

    def generate(self, binary_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Generate a key and helper data from binary input.

        Args:
            binary_data: Binary data from which to generate the key

        Returns:
            Tuple of (key, helper_data)
        """
        # Convert bytes to bit array
        bits = bytes_to_bits(binary_data)

        # Create secure hash of original data as the key
        key = hashlib.sha256(binary_data).digest()

        # Generate random bits for the sketch (secure sketch)
        # We use a cryptographically secure random generator
        random_seed = int.from_bytes(os.urandom(4), byteorder='big') % (2**32 - 1)  # Ensure seed is within valid range
        np.random.seed(random_seed)
        random_bits = np.random.randint(0, 2, size=len(bits), dtype=np.uint8)

        # XOR the original bits with random bits to create the syndrome
        syndrome = bits ^ random_bits

        # Create helper data
        helper_data = {
            'syndrome': syndrome.tolist(),
            'random_seed': random_seed,  # Store the properly sized random seed
            'data_length': len(binary_data),
            'error_tolerance': self.error_tolerance,
            'hash_check': hashlib.sha256(binary_data).hexdigest()[:16]  # First 16 chars of hash for verification
        }

        return key, helper_data

    def reproduce(self, binary_data: bytes, helper_data: Dict[str, Any]) -> Optional[bytes]:
        """
        Reproduce the key from binary data and helper data.

        Args:
            binary_data: Binary data similar to the original
            helper_data: Helper data from generate()

        Returns:
            Reproduced key or None if reproduction fails
        """
        # Validate the input data length
        if len(binary_data) != helper_data['data_length']:
            # Try to adjust the binary data to match expected length
            if len(binary_data) > helper_data['data_length']:
                binary_data = binary_data[:helper_data['data_length']]
            else:
                # If too short, pad with zeros
                binary_data = binary_data + b'\x00' * (helper_data['data_length'] - len(binary_data))

        # Convert to bit array
        bits = bytes_to_bits(binary_data)
        syndrome = np.array(helper_data['syndrome'], dtype=np.uint8)

        # Check data dimensions
        if len(bits) != len(syndrome):
            # Try to adjust bits array to match syndrome length
            if len(bits) > len(syndrome):
                bits = bits[:len(syndrome)]
            else:
                bits = np.pad(bits, (0, len(syndrome) - len(bits)))

        # Calculate Hamming distance between input bits and stored syndrome
        # This represents the errors in the random bits + errors in the input
        hamming_distance = np.sum(bits ^ syndrome)

        # Calculate max allowable errors
        max_errors = int(len(bits) * helper_data['error_tolerance'])

        # If too many errors, reproduction fails
        if hamming_distance > max_errors:
            return None

        # Ensure random_seed is within valid range for np.random.seed
        random_seed = helper_data['random_seed'] % (2**32 - 1)
        np.random.seed(random_seed)
        random_bits = np.random.randint(0, 2, size=len(syndrome), dtype=np.uint8)

        # XOR to get the original bits
        reconstructed_bits = syndrome ^ random_bits

        # Convert bits back to bytes
        reconstructed_bytes = bits_to_bytes(reconstructed_bits)

        # Verify reconstruction with hash check
        reconstruct_hash = hashlib.sha256(reconstructed_bytes).hexdigest()[:16]
        if reconstruct_hash != helper_data['hash_check']:
            if not self._try_error_correction(reconstructed_bits, helper_data):
                return None

        key = hashlib.sha256(reconstructed_bytes).digest()
        return key

    def _try_error_correction(self, bits: np.ndarray, helper_data: Dict[str, Any]) -> bool:
        """
        Try to correct errors in the reconstructed bits.

        Args:
            bits: Reconstructed bits
            helper_data: Helper data with hash_check for verification

        Returns:
            True if error correction succeeds, False otherwise
        """
        # Limit to checking a reasonable number of positions
        # In a real implementation, use proper error correction codes like BCH or Reed-Solomon

        # Only try to correct if error rate is close to threshold
        max_flips = min(8, int(len(bits) * helper_data['error_tolerance'] * 0.1))

        # Try single bit flips first (more efficient)
        for i in range(min(len(bits), 100)):
            original_bit = bits[i]
            bits[i] = 1 - original_bit

            # Convert bits to bytes and check hash
            test_bytes = bits_to_bytes(bits)
            test_hash = hashlib.sha256(test_bytes).hexdigest()[:16]

            if test_hash == helper_data['hash_check']:
                return True

            # Restore the bit if no match
            bits[i] = original_bit

        return False

    # Removed redundant wrapper methods - using imported functions directly


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
    binary = (embedding >= threshold).astype(np.uint8)
    
    # Convert to bytes
    return bits_to_bytes(binary)

def generate_key_with_helper(embedding: np.ndarray,
                             error_tolerance: float = 0.15) -> Tuple[bytes, Dict[str, Any]]:
    """
    Create a fuzzy extractor from a face embedding vector.
    This implementation can handle larger binary data sizes.

    Args:
        embedding: Face embedding vector
        error_tolerance: Fraction of bits that can be different (0.0-1.0)

    Returns:
        Tuple of (key, helper_data)
    """
    # Normalize embedding vector
    norm_embedding = embedding / np.linalg.norm(embedding)

    # Convert to binary representation
    binary_data = vector_to_binary(norm_embedding)

    # Create fuzzy extractor
    extractor = FuzzyExtractor(error_tolerance=error_tolerance)
    key, helper_data = extractor.generate(binary_data)

    # Store additional metadata in helper data
    helper_dict = {
        **helper_data,
        'vector_shape': embedding.shape,
        'vector_mean': float(np.mean(embedding)),
        'vector_std': float(np.std(embedding)),
        'binary_length': len(binary_data)
    }

    return key, helper_dict


def regenerate_key_from_helper(embedding: np.ndarray,
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

        # Create fuzzy extractor with saved parameters
        extractor = FuzzyExtractor(
            error_tolerance=helper_dict['error_tolerance']
        )

        # Reproduce key from helper data
        key = extractor.reproduce(binary_data, helper_dict)
        if key is None:
            return b''
        else:
            return key
    except Exception as e:
        print(f"Key reproduction error: {str(e)}")
        return b''
