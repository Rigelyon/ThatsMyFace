from typing import Optional, List

import numpy as np
from deepface import DeepFace
from scipy import spatial

from modules.constants import (
    FACE_DETECTION_MODEL,
    EMBEDDING_MODEL,
    DISTANCE_METRIC,
    SIMILARITY_THRESHOLD,
)


def detect_faces(image: np.ndarray, anti_spoofing: bool = False) -> List[dict]:
    """
    Detect faces in an image

    Args:
        image: numpy array containing the image
        anti_spoofing: Whether to perform anti-spoofing check (detect fake faces)

    Returns:
        List of detected face information dictionaries with "is_real" field when anti_spoofing=True
    """
    try:
        faces = DeepFace.extract_faces(
            image,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=False,
            anti_spoofing=anti_spoofing,
        )
        return faces
    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return []


def get_face_embedding(
    image: np.ndarray, anti_spoofing: bool = False
) -> Optional[np.ndarray]:
    """
    Extract face embedding vector from an image with a face

    Args:
        image: numpy array containing the image
        anti_spoofing: Whether to perform anti-spoofing check (detect fake faces)

    Returns:
        Face embedding vector or None if no face detected or spoofing detected
    """
    try:
        # Get face embedding
        embedding_obj = DeepFace.represent(
            image,
            model_name=EMBEDDING_MODEL,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=True,
            align=True,
            normalization="base",
            anti_spoofing=anti_spoofing,
        )

        # Return the embedding vector
        if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
            # If anti_spoofing is enabled, check if face is real
            if anti_spoofing and not embedding_obj[0].get("is_real", True):
                print("Spoofing detected: The face appears to be fake")
                return None
            return np.array(embedding_obj[0]["embedding"])
        return None
    except Exception as e:
        print(f"Error getting face embedding: {str(e)}")
        return None


def check_face_match(
    image: np.ndarray, reference_image: np.ndarray, anti_spoofing: bool = False
) -> tuple:
    """
    Check if the image contains a face that matches the reference image

    Args:
        image: numpy array containing the target image
        reference_image: numpy array containing the reference face image
        anti_spoofing: Whether to perform anti-spoofing check (detect fake faces)

    Returns:
        Tuple of (is_match, is_real, message)
        - is_match: True if faces match, False otherwise
        - is_real: True if face is real, False if spoofing detected, None if anti_spoofing disabled
        - message: Explanation message if there's an issue, empty string otherwise
    """
    try:
        # Use DeepFace.verify to directly compare faces
        verification_result = DeepFace.verify(
            img1_path=reference_image,
            img2_path=image,
            model_name=EMBEDDING_MODEL,
            detector_backend=FACE_DETECTION_MODEL,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False,
            align=True,
            normalization="base",
            anti_spoofing=anti_spoofing,
        )

        is_match = verification_result["verified"]
        is_real = None
        message = ""

        # Check anti-spoofing results if enabled
        if anti_spoofing:
            # Check if faces are real (note: the API returns "facial_areas" with is_real)
            img1_real = verification_result.get("img1_facial_areas", [{}])[0].get(
                "is_real", True
            )
            img2_real = verification_result.get("img2_facial_areas", [{}])[0].get(
                "is_real", True
            )

            is_real = img1_real and img2_real

            if not img1_real:
                message = (
                    "Reference face appears to be fake (possible spoofing attempt)"
                )
            elif not img2_real:
                message = "Target face appears to be fake (possible spoofing attempt)"

        # Return verification result with anti-spoofing info
        return (is_match, is_real, message)
    except Exception as e:
        message = f"Error checking face match: {str(e)}"
        print(message)
        return (False, None, message)


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate similarity between two face embeddings

    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector

    Returns:
        Similarity score (higher means more similar)
    """
    if DISTANCE_METRIC == "cosine":
        # Cosine similarity
        similarity = 1 - spatial.distance.cosine(embedding1, embedding2)
    elif DISTANCE_METRIC == "euclidean":
        # Euclidean distance (convert to similarity)
        distance = np.linalg.norm(embedding1 - embedding2)
        similarity = 1 / (1 + distance)
    elif DISTANCE_METRIC == "euclidean_l2":
        # Normalized Euclidean distance (convert to similarity)
        distance = np.linalg.norm(embedding1 - embedding2)
        similarity = 1 / (1 + distance)
    else:
        # Default to cosine similarity
        similarity = 1 - spatial.distance.cosine(embedding1, embedding2)

    return similarity


def verify_face_similarity(
    embedding1: np.ndarray, embedding2: np.ndarray, custom_threshold: float = None
) -> bool:
    """
    Verify if two face embeddings belong to the same person.

    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector
        custom_threshold: Optional custom similarity threshold (overrides SIMILARITY_THRESHOLD)

    Returns:
        Boolean indicating whether faces are similar enough (same person)
    """
    similarity = calculate_similarity(embedding1, embedding2)
    threshold = (
        custom_threshold if custom_threshold is not None else SIMILARITY_THRESHOLD
    )

    return similarity >= threshold
