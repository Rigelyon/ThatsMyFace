# That's My Face - Secure Image Watermarking with Facial Authentication

A Streamlit application for watermarking images using facial recognition, DCT (Discrete Cosine Transform), and SVD (Singular Value Decomposition) methods. The watermark is encrypted using AES with the encryption key derived from facial recognition embeddings.

## Features

- Facial recognition-based authentication using DeepFace
- DCT and SVD-based watermarking
- AES encryption for watermark security
- Option to watermark only images containing a specific face
- Support for text and image watermarks
- Watermark extraction capability

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/thats-my-face.git
cd thats-my-face
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. The application offers two main functions:
    - **Embed Watermark**: Add a watermark to your images
    - **Extract Watermark**: Retrieve a watermark from previously watermarked images

### Embedding a Watermark

1. Upload an authentication face image (a clear photo of a face)
2. Upload a watermark (text file or image)
3. Upload images you want to watermark (maximum 10 files)
4. Choose whether to watermark only images containing the authentication face
5. Click "Process Images"

### Extracting a Watermark

1. Upload the same authentication face used during embedding
2. Upload the watermarked image
3. Click "Extract Watermark"

## How It Works

1. **Facial Authentication**:
    - DeepFace is used to extract facial embeddings from the authentication image
    - These embeddings serve as a unique identifier for the person

2. **Encryption**:
    - The facial embeddings are processed to generate an AES encryption key
    - The watermark is encrypted using this key

3. **Watermarking**:
    - The image is divided into blocks
    - DCT is applied to transform the blocks to frequency domain
    - SVD is applied to the DCT coefficients
    - The singular values are modified based on the encrypted watermark bits
    - Inverse transforms reconstruct the watermarked image

4. **Face Matching**:
    - If authentication is required, DeepFace checks if the target images contain the same face
    - Only matching images receive the watermark

## Technical Details

- **Face Recognition Model**: Facenet512
- **Watermarking Method**: DCT-SVD
- **Encryption Algorithm**: AES-256 in CBC mode
- **Maximum Files**: 10 images per process
- **Maximum Watermark Size**: 1KB

## Limitations

- Face recognition accuracy depends on image quality
- Watermark may degrade under heavy image manipulation
- Maximum of 10 images can be processed at once
- Watermark size is limited to 1KB