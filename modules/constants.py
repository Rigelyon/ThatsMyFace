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
SIMILARITY_THRESHOLD = 0.5  # Threshold for face similarity (lower means more strict)