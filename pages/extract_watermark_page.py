import streamlit as st
from PIL import Image
import numpy as np
import io
from modules.face_recognition import get_face_embedding
from modules.watermarking import extract_watermark, detect_watermark
from modules.encryption import generate_key_from_embedding, decrypt_watermark

def display_extract_watermark_page(debug_mode=False):
    st.header("Extract Watermark")

    # Upload authentication face
    st.subheader("1. Upload Authentication Face")
    auth_face = st.file_uploader("Upload the same face used for watermark embedding",
                                 type=["jpg", "jpeg", "png"])

    # Upload watermarked image
    st.subheader("2. Upload Watermarked Image")
    watermarked_file = st.file_uploader("Upload the watermarked image",
                                        type=["jpg", "jpeg", "png"])

    if st.button("Extract Watermark", disabled=not (auth_face and watermarked_file)):
        try:
            # Process the authentication face
            auth_image = Image.open(auth_face)
            auth_image_array = np.array(auth_image)

            # Get embedding for encryption key generation
            embedding = get_face_embedding(auth_image_array)

            if embedding is None:
                st.error("Could not detect a face in the authentication image.")
            else:
                # Generate encryption key from face embedding
                encryption_key = generate_key_from_embedding(embedding)

                # Debug: Show key information
                if debug_mode:
                    st.write("DEBUG: Authentication key (first 8 bytes hex):", encryption_key[:8].hex())
                    st.write("DEBUG: Authentication face embedding stats:", {
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding)),
                        'shape': embedding.shape
                    })

                # Extract watermark
                watermarked_img = Image.open(watermarked_file)

                # First detect if image likely contains a watermark
                has_watermark = detect_watermark(watermarked_img)
                if has_watermark:
                    st.write("Image appears to contain a watermark.")

                if debug_mode:
                    st.write("DEBUG: Attempting to extract watermark...")
                encrypted_watermark = extract_watermark(watermarked_img)

                if encrypted_watermark:
                    if debug_mode:
                        st.write("DEBUG: Watermark extraction successful! Length:", len(encrypted_watermark))

                    # Decrypt watermark
                    try:
                        watermark_data = decrypt_watermark(encrypted_watermark, encryption_key)

                        # Determine if watermark is text or image
                        try:
                            # Try to decode as text
                            watermark_text = watermark_data.decode('utf-8')
                            st.success("Watermark extracted successfully!")
                            st.text_area("Extracted Watermark (Text)", watermark_text, height=150)
                        except UnicodeDecodeError:
                            # If not text, treat as image
                            try:
                                watermark_img = Image.open(io.BytesIO(watermark_data))
                                st.success("Watermark extracted successfully!")
                                st.image(watermark_img, caption="Extracted Watermark (Image)", use_container_width=True)
                            except Exception:
                                st.error("Could not interpret the extracted watermark as text or image.")
                    except Exception:
                        st.error("Failed to decrypt the watermark. The authentication face may not match the one used for embedding.")
                else:
                    st.error("Could not extract a watermark from this image. This usually means either:")
                    st.markdown("""
                    - The image doesn't contain a watermark
                    - The image wasn't watermarked with this application
                    - The image was modified after watermarking (resized, cropped, compressed)
                    - You're not using the same authentication face that was used for embedding
                    """)

        except Exception as e:
            st.error(f"An error occurred during extraction: {str(e)}")
