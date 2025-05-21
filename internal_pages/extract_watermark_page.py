import io

import numpy as np
import streamlit as st
from PIL import Image

from modules.encryption import decrypt_watermark
from modules.face_recognition import get_face_embedding
from modules.fuzzy_extractor import regenerate_key_from_helper
from modules.utils import deserialize_helper_data, has_face
from modules.watermarking import extract_watermark


def display_extract_watermark_page(debug_mode=False):
    st.header("Extract Watermark")

    # Upload authentication face
    st.subheader("1. Upload Authentication Face")
    auth_face = st.file_uploader(
        "Upload a face image for authentication (can be different from the one used for embedding)",
        type=["jpg", "jpeg", "png"])
    if auth_face:
        auth_img = Image.open(auth_face)
        st.image(auth_img, caption="Original Image", use_container_width=True)

    # Upload helper data file
    st.subheader("2. Upload Helper Data File")
    helper_file = st.file_uploader("Upload the helper data file generated during watermark embedding",
                                   type=["bin"])

    st.subheader("3. Upload Images File")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Original Image")
        original_file = st.file_uploader(
            "Upload original image",
            type=["jpg", "jpeg", "png"],
            key="extract_test_original_image",
        )

        original_img = None
        if original_file:
            original_img = Image.open(original_file)
            st.image(original_img, caption="Original Image", use_container_width=True)

    with col2:
        st.markdown("### Watermarked Image")
        watermarked_file = st.file_uploader(
            "Upload watermarked image",
            type=["jpg", "jpeg", "png"],
            key="extract_test_image",
        )

        watermarked_img = None
        if watermarked_file:
            watermarked_img = Image.open(watermarked_file)
            st.image(
                watermarked_img, caption="Watermarked Image", use_container_width=True
            )

    if st.button("Extract Watermark", disabled=not (auth_face and helper_file and watermarked_file)):
        if not auth_face or not has_face(Image.open(auth_face)):
            st.error("Authentication image must contain a clearly visible face.")
        elif not original_file:
            st.error("Please upload the original image.")
        elif not watermarked_file:
            st.error("Please upload the watermarked image.")
        elif not helper_file:
            st.error("Please upload the helper data file.")
        else:
            with st.spinner("Extracting watermark..."):
                try:
                    # Process the authentication face
                    auth_image_array = np.array(auth_img)

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
                                'vector_shape': helper_data.get('vector_shape'),
                                'vector_mean': helper_data.get('vector_mean'),
                                'vector_std': helper_data.get('vector_std')
                            })
                            st.write("DEBUG: Current face embedding stats:", {
                                'mean': float(np.mean(embedding)),
                                'std': float(np.std(embedding)),
                                'min': float(np.min(embedding)),
                                'max': float(np.max(embedding)),
                                'shape': embedding.shape
                            })

                        # Generate encryption key from face embedding using helper data
                        decryption_key = regenerate_key_from_helper(embedding, helper_data)

                        if decryption_key is b"":
                            st.error("Failed to regenerate key. The face may be too different from the original one.")
                            st.info(
                                "Try using a clearer photo of the same person or adjust error tolerance during embedding.")
                        else:
                            # Debug: Show key information
                            if debug_mode:
                                st.write("DEBUG: Regenerated key (first 8 bytes hex):", decryption_key[:8].hex())

                            if debug_mode:
                                st.write("DEBUG: Attempting to extract watermark...")
                            encrypted_watermark = extract_watermark(watermarked_img)

                            if encrypted_watermark:
                                if debug_mode:
                                    st.write("DEBUG: Watermark extraction successful! Length:", len(encrypted_watermark))

                                # Decrypt watermark
                                try:
                                    decrypted_watermark_img = decrypt_watermark(encrypted_watermark, decryption_key)
                                    if decrypted_watermark_img:
                                        st.success("Watermark extracted successfully!")
                                        st.subheader("Watermark Extraction Results")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.markdown("**Original Image**")
                                            st.image(original_img, use_container_width=True)

                                        with col2:
                                            st.markdown("**Watermarked Image**")
                                            st.image(watermarked_img, use_container_width=True)

                                        with col3:
                                            st.markdown("**Extracted Watermark**")
                                            st.image(
                                                decrypted_watermark_img,
                                                caption="Extracted Watermark",
                                                use_container_width=True,
                                            )

                                        img_bytes = io.BytesIO()
                                        decrypted_watermark_img.save(img_bytes, format="PNG")

                                        st.download_button(
                                            label="Download Watermark Image",
                                            data=img_bytes.getvalue(),
                                            file_name="extracted_watermark.png",
                                            mime="image/png",
                                        )
                                    else:
                                        st.error(
                                            "Failed to decrypt the watermark. The authentication face may not match closely enough with the one used for embedding or the error tolerance are too low.")
                                        if debug_mode:
                                            st.info("DEBUG: Decryption returned None, suggesting the key is incorrect")
                                except Exception as e:
                                    st.error(f"Failed to decrypt the watermark: {str(e)}")
                            else:
                                st.error("Could not extract a watermark from this image. This usually means either:")
                                st.markdown("""
                                - The image doesn't contain a watermark
                                - The image wasn't watermarked with this application
                                - The original image was modified after (resized, cropped, compressed)
                                """)

                except Exception as e:
                    st.error(f"An error occurred during extraction: {str(e)}")
