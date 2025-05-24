# That's My Face - Secure Image Watermarking with Facial Authentication

A Streamlit application for watermarking images using facial recognition, DCT (Discrete Cosine Transform), and SVD (Singular Value Decomposition) methods. The watermark is encrypted using AES with the encryption key derived from facial recognition embeddings.

## Features

- Facial recognition-based authentication using DeepFace
- Fuzzy extractor technology for consistent key generation from different face images
- DCT and SVD-based watermarking
- QR code AES encryption for watermark security
- Option to watermark only images containing a specific face
- Use text as a watermark
- Watermark extraction capability with helper data

## Installation

1. Clone this repository:
```
git clone https://github.com/Rigelyon/ThatsMyFace.git
cd ThatsMyFace
```

2. Create Virtual Environment:
```
python -m venv .venv
```

3. Activate the Virtual Environment:
```
# On Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

4. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

## Contribution

Contribution are not allowed as this is a personal project