import streamlit as st
from PIL import Image
import numpy as np
import io
from modules.face_recognition import get_face_embedding
from modules.watermarking import extract_watermark, detect_watermark
from modules.encryption import regenerate_key_from_helper, decrypt_watermark
from modules.fuzzy_extractor import load_helper_data, deserialize_helper_data

def display_extract_watermark_page(debug_mode=False):
    st.header("Extract Watermark")

    # Upload authentication face
    st.subheader("1. Upload Authentication Face")
    auth_face = st.file_uploader("Upload a face image for authentication (can be different from the one used for embedding)",
                                 type=["jpg", "jpeg", "png"])

    # Upload helper data file
    st.subheader("2. Upload Helper Data File")
    helper_file = st.file_uploader("Upload the helper data file generated during watermark embedding",
                                   type=["bin"])

    # Upload watermarked image
    st.subheader("3. Upload Watermarked Image")
    watermarked_file = st.file_uploader("Upload the watermarked image",
                                        type=["jpg", "jpeg", "png"])

    if st.button("Extract Watermark", disabled=not (auth_face and helper_file and watermarked_file)):
        try:
            # Process the authentication face
            auth_image = Image.open(auth_face)
            auth_image_array = np.array(auth_image)

            # Get embedding for encryption key generation
            embedding = get_face_embedding(auth_image_array)

            if embedding is None:
                st.error("Could not detect a face in the authentication image.")
            else:
                # Load helper data
                helper_data = deserialize_helper_data(helper_file.getvalue())

                # Debug: Show helper data information
                if debug_mode:
                    st.write("DEBUG: Helper data loaded:", {
                        'error_tolerance': helper_data.get('error_tolerance'),
                        'original_vector_shape': helper_data.get('vector_shape'),
                        'original_mean': helper_data.get('vector_mean'),
                        'original_std': helper_data.get('vector_std')
                    })
                    st.write("DEBUG: Current face embedding stats:", {
                        'mean': float(np.mean(embedding)),
                        'std': float(np.std(embedding)),
                        'min': float(np.min(embedding)),
                        'max': float(np.max(embedding)),
                        'shape': embedding.shape
                    })

                # Generate encryption key from face embedding using helper data
                encryption_key = regenerate_key_from_helper(embedding, helper_data)

                if encryption_key is None:
                    st.error("Failed to regenerate key. The face may be too different from the original one.")
                    st.info("Try using a clearer photo of the same person or adjust error tolerance during embedding.")
                else:
                    # Debug: Show key information
                    if debug_mode:
                        st.write("DEBUG: Regenerated key (first 8 bytes hex):", encryption_key[:8].hex())

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

                            if watermark_data:
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
                            else:
                                st.error("Failed to decrypt the watermark. The authentication face may not match closely enough with the one used for embedding.")
                                if debug_mode:
                                    st.info("DEBUG: Decryption returned None, suggesting the key is incorrect")
                        except Exception as e:
                            st.error(f"Failed to decrypt the watermark: {str(e)}")
                    else:
                        st.error("Could not extract a watermark from this image. This usually means either:")
                        st.markdown("""
                        - The image doesn't contain a watermark
                        - The image wasn't watermarked with this application
                        - The image was modified after watermarking (resized, cropped, compressed)
                        """)

        except Exception as e:
            st.error(f"An error occurred during extraction: {str(e)}")