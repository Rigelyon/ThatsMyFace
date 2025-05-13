import numpy as np
from deepface import DeepFace
from typing import Optional, List
from scipy import spatial

from modules.constants import FACE_DETECTION_MODEL, EMBEDDING_MODEL, DISTANCE_METRIC

def detect_faces(image: np.ndarray) -> List[dict]:
    """
    Detect faces in an image

    Args:
        image: numpy array containing the image

    Returns:
        List of detected face information dictionaries
    """
    try:
        faces = DeepFace.extract_faces(
            image,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=False
        )
        return faces
    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return []

def get_face_embedding(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract face embedding vector from an image with a face

    Args:
        image: numpy array containing the image

    Returns:
        Face embedding vector or None if no face detected
    """
    try:
        # Get face embedding
        embedding_obj = DeepFace.represent(
            image,
            model_name=EMBEDDING_MODEL,
            detector_backend=FACE_DETECTION_MODEL,
            enforce_detection=True,
            align=True,
            normalization="base"
        )

        # Return the embedding vector
        if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
            return np.array(embedding_obj[0]["embedding"])
        return None
    except Exception as e:
        print(f"Error getting face embedding: {str(e)}")
        return None

def check_face_match(image: np.ndarray, reference_image: np.ndarray) -> bool:
    """
    Check if the image contains a face that matches the reference image

    Args:
        image: numpy array containing the target image
        reference_image: numpy array containing the reference face image

    Returns:
        True if matching face found, False otherwise
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
            normalization="base"
        )
        
        # Return verification result
        return verification_result["verified"]
    except Exception as e:
        print(f"Error checking face match: {str(e)}")
        return False

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