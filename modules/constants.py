# Constants
MAX_IMAGES = 10
MAX_WATERMARK_SIZE = 1024  # Maximum watermark size in bytes

# Constants for watermarking
BLOCK_SIZE = 8
ALPHA = 0.1  # Strength of watermark (lower means less visible)
MAX_SVD_COEFFICIENTS = 10  # Number of singular values to modify

# Settings for face recognition
FACE_DETECTION_MODEL = "opencv"  # Options: opencv, ssd, dlib, mtcnn, retinaface, mediapipe
EMBEDDING_MODEL = "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID
DISTANCE_METRIC = "cosine"  # Options: cosine, euclidean, euclidean_l2

# Thresholds for face similarity
MIN_SIMILARITY_THRESHOLD = 0.7  # Minimum threshold - if similarity is below this, always reject
SIMILARITY_THRESHOLD = MIN_SIMILARITY_THRESHOLD  # Default threshold for face similarity (higher means more strict)
MAX_SIMILARITY_THRESHOLD = 0.95  # Maximum threshold - fully confident it's the same person