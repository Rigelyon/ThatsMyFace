import streamlit as st
from PIL import Image
import numpy as np
import io
import time
import os
from modules.watermarking import embed_watermark
from modules.face_recognition import get_face_embedding, check_face_match
from modules.encryption import generate_key_with_helper, encrypt_watermark
from modules.fuzzy_extractor import save_helper_data, serialize_helper_data
from modules.utils import has_face, convert_to_bytes
from modules.constants import MAX_IMAGES, MAX_WATERMARK_SIZE

def display_embed_watermark_page(debug_mode=False):
    st.header("Embed Watermark")

    # File uploaders
    st.subheader("1. Upload Authentication Face")
    auth_face = st.file_uploader("Upload a clear image of the face for authentication",
                                 type=["jpg", "jpeg", "png"])

    # Added error tolerance slider
    st.subheader("2. Authentication Settings")
    error_tolerance = st.slider(
        "Error Tolerance (higher values allow more variation in face images)",
        min_value=10,
        max_value=200,
        value=50,
        help="Higher values make authentication more flexible but potentially less secure"
    )

    st.subheader("3. Upload Watermark")
    watermark_file = st.file_uploader("Upload watermark (text file recommended)",
                                      type=["txt", "jpg", "jpeg", "png"])

    st.subheader("4. Upload Images to Watermark")
    uploaded_files = st.file_uploader("Upload images (max 10 files)",
                                      type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)

    # Authentication option
    st.subheader("5. Watermarking Options")
    auth_required = st.checkbox("Only watermark images containing the authentication face")

    if st.button("Process Images", disabled=not (auth_face and watermark_file and uploaded_files)):
        # Validate inputs
        if not auth_face or not has_face(Image.open(auth_face)):
            st.error("Authentication image must contain a clearly visible face.")
        elif not watermark_file:
            st.error("Please upload a watermark file.")
        elif not uploaded_files:
            st.error("Please upload at least one image to watermark.")
        elif len(uploaded_files) > MAX_IMAGES:
            st.error(f"Maximum {MAX_IMAGES} images allowed. Please reduce the number of uploads.")
        elif watermark_file.size > MAX_WATERMARK_SIZE:
            st.error(f"Watermark size exceeds the maximum allowed size ({MAX_WATERMARK_SIZE/1024:.1f} KB).")
        else:
            try:
                # Process the authentication face
                auth_image = Image.open(auth_face)
                auth_image_array = np.array(auth_image)

                # Get embedding for encryption key generation
                embedding = get_face_embedding(auth_image_array)

                if embedding is None:
                    st.error("Could not detect a face in the authentication image.")
                else:
                    # Generate encryption key and helper data using fuzzy extractor
                    encryption_key, helper_data = generate_key_with_helper(embedding, error_tolerance)

                    # Debug: Show key information if debug mode is enabled
                    if debug_mode:
                        st.write("DEBUG: Embedding key (first 8 bytes hex):", encryption_key[:8].hex())
                        st.write("DEBUG: Embedding face stats:", {
                            'mean': float(np.mean(embedding)),
                            'std': float(np.std(embedding)),
                            'min': float(np.min(embedding)),
                            'max': float(np.max(embedding)),
                            'shape': embedding.shape,
                            'error_tolerance': error_tolerance
                        })

                    # Prepare watermark data
                    if watermark_file.type.startswith('text'):
                        watermark_data = watermark_file.getvalue().decode('utf-8')
                    else:
                        watermark_image = Image.open(watermark_file)
                        # Convert image watermark to bytes
                        watermark_data = convert_to_bytes(watermark_image)

                    # Encrypt watermark
                    encrypted_watermark = encrypt_watermark(watermark_data, encryption_key)

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    watermarked = []
                    not_watermarked = []

                    # Create directory for helper files if it doesn't exist
                    if not os.path.exists("output/helpers"):
                        os.makedirs("output/helpers")

                    # Save helper data to a file
                    helper_filename = f"output/helpers/helper_data_{int(time.time())}.bin"
                    save_helper_data(helper_data, helper_filename)

                    for i, uploaded_file in enumerate(uploaded_files[:MAX_IMAGES]):
                        status_text.text(f"Processing image {i+1}/{len(uploaded_files[:MAX_IMAGES])}...")
                        progress_value = (i + 1) / len(uploaded_files[:MAX_IMAGES])
                        progress_bar.progress(progress_value)

                        # Load image
                        img = Image.open(uploaded_file)

                        # Check if authentication is required and image contains matching face
                        should_watermark = True
                        if auth_required:
                            should_watermark = check_face_match(np.array(img), auth_image_array)

                        # Apply watermark if conditions are met
                        if should_watermark:
                            watermarked_img = embed_watermark(img, encrypted_watermark)
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
                            key="download_helper"
                        )

                        st.info("⚠️ **IMPORTANT:** Download and keep the helper data file. You will need it to extract watermarks later, even from different photos of the same person.")

                        # Display watermarked images
                        if watermarked:
                            st.write(f"**{len(watermarked)} images were watermarked:**")
                            cols = st.columns(min(3, len(watermarked)))

                            for i, (name, img) in enumerate(watermarked):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    st.image(img, caption=name, use_container_width=True)

                                    # Save button for each image
                                    img_bytes = io.BytesIO()
                                    img.save(img_bytes, format="PNG")
                                    st.download_button(
                                        label="Download",
                                        data=img_bytes.getvalue(),
                                        file_name=f"watermarked_{name}",
                                        mime="image/png",
                                        key=f"download_btn_{i}"
                                    )

                        # List images that were not watermarked
                        if not_watermarked:
                            st.write(f"**{len(not_watermarked)} images were not watermarked** (no matching face found):")
                            st.write(", ".join(not_watermarked))

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")