# Constants
MAX_IMAGES = 30
MAX_WATERMARK_CHARACTERS = 100
QRCODE_SIZE = 1000  # Ukuran default QR Code dalam pixel

# Constants for watermarking
BLOCK_SIZE = 8  # Ukuran blok 8x8 untuk DCT
ALPHA = 0.1  # Faktor kekuatan watermark
MAX_SVD_COEFFICIENTS = 10  # Number of singular values to modify

ERROR_TOLERANCE = 60  # Default error tolerance for face matching

# Settings for face recognition
FACE_DETECTION_MODEL = (
    "opencv"  # Options: opencv, ssd, dlib, mtcnn, retinaface, mediapipe
)
EMBEDDING_MODEL = (
    "Facenet512"  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID
)
DISTANCE_METRIC = "cosine"  # Options: cosine, euclidean, euclidean_l2

# Thresholds for face similarity
MIN_SIMILARITY_THRESHOLD = (
    0.7  # Minimum threshold - if similarity is below this, always reject
)
SIMILARITY_THRESHOLD = MIN_SIMILARITY_THRESHOLD  # Default threshold for face similarity (higher means more strict)
MAX_SIMILARITY_THRESHOLD = (
    0.95  # Maximum threshold - fully confident it's the same person
)
