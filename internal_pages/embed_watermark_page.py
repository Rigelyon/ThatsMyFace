import io
import os
import time

import numpy as np
import streamlit as st
from PIL import Image

from modules.constants import MAX_IMAGES, MAX_WATERMARK_SIZE, MAX_WATERMARK_RESOLUTION
from modules.encryption import encrypt_watermark
from modules.face_recognition import (
    get_face_embedding,
    check_face_match,
    calculate_similarity,
)
from modules.fuzzy_extractor import generate_key_with_helper
from modules.utils import has_face, convert_to_bytes, save_helper_data
from modules.watermarking import embed_watermark


def display_embed_watermark_page(debug_mode=False):
    st.header("Embed Watermark")

    # File uploaders
    st.subheader("1. Upload Authentication Face")
    auth_face = st.file_uploader(
        "Upload a clear image of the face for authentication",
        type=["jpg", "jpeg", "png"],
    )
    if auth_face:
        auth_img = Image.open(auth_face)
        st.image(auth_img, caption="Original Image", width=300)

    # Added error tolerance slider
    st.subheader("2. Authentication Settings")
    st.markdown(
        """
    Adjust this slider to control the similarity tolerance. **This is important when you want to use different authentication image to extract the watermark later**. Balance between security (lower values) and flexibility (higher values).
    - **Low values** (1-30): Stricter matching - Requires very similar face images but more secure.
    - **Medium values** (31-70): Balanced - Good for most use cases.
    - **High values** (71-95): Flexible matching - Works with more varied face images but potentially less secure.
    """
    )
    error_tolerance = st.slider(
        "Error Tolerance",
        min_value=1,
        max_value=95,
        value=60,
        help="Higher values allow more variation in face images but may reduce security",
    )

    st.subheader("3. Upload Watermark")
    st.markdown(
        f"""
    **Watermark image requirements:**
    - Format: JPG, JPEG, or PNG
    - File size: Maximum {MAX_WATERMARK_SIZE / (1024 * 1024)} MB
    - Resolution: Not more than {MAX_WATERMARK_RESOLUTION} x {MAX_WATERMARK_RESOLUTION} pixels
    """
    )
    watermark_file = st.file_uploader("Upload watermark", type=["jpg", "jpeg", "png"])
    if watermark_file:
        watermark_img = Image.open(watermark_file)
        st.image(
            watermark_img,
            caption=f"Watermark Image ({watermark_img.width}x{watermark_img.height})",
            width=200,
        )
        img_byte_arr = io.BytesIO()
        watermark_img.save(
            img_byte_arr,
            format=watermark_img.format if watermark_img.format else "PNG",
        )
        watermark_data = img_byte_arr.getvalue()

    st.subheader("4. Upload Images to Watermark")
    uploaded_files = st.file_uploader(
        f"Upload images (max {MAX_IMAGES} files)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Authentication option
    st.subheader("5. Watermarking Options")
    auth_required = st.checkbox(
        "Only watermark images containing the authentication face"
    )

    if st.button(
        "Process Images", disabled=not (auth_face and watermark_file and uploaded_files)
    ):
        # Validate inputs
        if not auth_face or not has_face(Image.open(auth_face)):
            st.error("Authentication image must contain a clearly visible face.")
        elif not watermark_file:
            st.error("Please upload a watermark image.")
        elif not uploaded_files:
            st.error("Please upload at least one image to watermark.")
        elif len(uploaded_files) > MAX_IMAGES:
            st.error(
                f"Maximum {MAX_IMAGES} images allowed. Please reduce the number of uploads."
            )
        elif watermark_file.size > MAX_WATERMARK_SIZE:
            st.error(
                f"Watermark size exceeds the maximum allowed size ({MAX_WATERMARK_SIZE / (1024 * 1024)} MB)."
            )
        elif (
                watermark_img.width > MAX_WATERMARK_RESOLUTION
                or watermark_img.height > MAX_WATERMARK_RESOLUTION
            ):
                st.error(
                    f"Watermark image resolution too large ({watermark_img.width}x{watermark_img.height})! Maximum {MAX_WATERMARK_RESOLUTION}x{MAX_WATERMARK_RESOLUTION} pixels."
                )
        else:
            try:
                # Process the authentication face
                auth_image_array = np.array(auth_img)

                # Get embedding for encryption key generation
                embedding = get_face_embedding(auth_image_array)

                if embedding is None:
                    st.error("Could not detect a face in the authentication image.")
                else:
                    # Generate encryption key and helper data using fuzzy extractor
                    norm_tolerance = error_tolerance / 100.0
                    encryption_key, helper_data = generate_key_with_helper(
                        embedding, norm_tolerance
                    )

                    # Encrypt watermark
                    # encrypted_watermark = encrypt_watermark(
                    #     watermark_img, encryption_key
                    # )

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    watermarked = []
                    not_watermarked = []

                    # Create directory for helper files if it doesn't exist
                    if not os.path.exists("output/helpers"):
                        os.makedirs("output/helpers")

                    # Save helper data to a file
                    helper_filename = (
                        f"output/helpers/helper_data_{int(time.time())}.bin"
                    )
                    save_helper_data(helper_data, helper_filename)

                    # TODO: Results layout
                    for i, uploaded_file in enumerate(uploaded_files[:MAX_IMAGES]):
                        status_text.text(
                            f"Processing image {i + 1}/{len(uploaded_files[:MAX_IMAGES])}..."
                        )
                        progress_value = (i + 1) / len(uploaded_files[:MAX_IMAGES])
                        progress_bar.progress(progress_value)

                        # Load image
                        img = Image.open(uploaded_file)

                        # Check if authentication is required and image contains matching face
                        should_watermark = True
                        if auth_required:
                            should_watermark = check_face_match(
                                np.array(img), auth_image_array
                            )

                        # Apply watermark if conditions are met
                        if should_watermark:
                            watermarked_img = embed_watermark(img, watermark_img)
                            watermarked.append((uploaded_file.name, watermarked_img))
                        else:
                            not_watermarked.append(uploaded_file.name)

                        time.sleep(0.1)  # Small delay for UI feedback

                    # Show results
                    status_text.text("Processing complete!")

                    with results_container:
                        st.subheader("Results")

                        # Download helper data file
                        with open(helper_filename, "rb") as file:
                            helper_data_bytes = file.read()

                        st.download_button(
                            label="⬇️ Download Helper Data File (IMPORTANT)",
                            data=helper_data_bytes,
                            file_name=os.path.basename(helper_filename),
                            mime="application/octet-stream",
                            help="This file is required to extract watermarks later. Keep it safe!",
                            key="download_helper",
                        )

                        st.info(
                            "⚠️ **IMPORTANT:** Download and keep the helper data file. You will need it to extract watermarks later, even from different photos of the same person."
                        )

                        # Display watermarked images
                        if watermarked:
                            st.write(f"**{len(watermarked)} images were watermarked:**")
                            cols = st.columns(min(3, len(watermarked)))

                            for i, (name, img) in enumerate(watermarked):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    st.image(
                                        img, caption=name, width=200
                                    )

                                    # Save button for each image
                                    img_bytes = io.BytesIO()
                                    img.save(img_bytes, format="PNG")
                                    st.download_button(
                                        label="Download",
                                        data=img_bytes.getvalue(),
                                        file_name=f"watermarked_{name}",
                                        mime="image/png",
                                        key=f"download_btn_{i}",
                                    )

                        # List images that were not watermarked
                        if not_watermarked:
                            st.write(
                                f"**{len(not_watermarked)} images were not watermarked** (no matching face found):"
                            )
                            st.write(", ".join(not_watermarked))

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
