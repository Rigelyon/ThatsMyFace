import io
import os
import time
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

from modules.constants import (
    MAX_IMAGES,
    MAX_WATERMARK_CHARACTERS,
    QRCODE_SIZE,
    BLOCK_SIZE,
    ALPHA,
    ERROR_TOLERANCE,
)
from modules.encryption import encrypt_watermark
from modules.face_recognition import (
    get_face_embedding,
    check_face_match,
)
from modules.fuzzy_extractor import generate_key_with_helper
from modules.utils import has_face, save_helper_data
from modules.watermarking import embed_watermark
from modules.qrcode_generator import text_to_qrcode


def display_embed_watermark_page(debug_mode=False):
    # Header
    st.title("✨ Embed Watermark")
    st.markdown(
        "Secure your images with invisible watermarks linked to your facial identity"
    )

    # Menggunakan struktur yang lebih baik untuk penanganan state
    if "embed_state" not in st.session_state:
        st.session_state.embed_state = {
            "processing_completed": False,
            "helper_filename": None,
            "helper_data_bytes": None,
            "watermarked_images": [],
            "not_watermarked_images": [],
        }

    # Store custom values in session state
    if "custom_settings" not in st.session_state:
        st.session_state.custom_settings = {}

    # Use default values from constants.py when advanced settings are hidden
    if "custom_settings" not in st.session_state:
        st.session_state.custom_settings = {
            "qrcode_size": QRCODE_SIZE,
            "block_size": BLOCK_SIZE,
            "alpha": ALPHA,
            "max_images": MAX_IMAGES,
            "error_tolerance": ERROR_TOLERANCE,
        }


    # Menampilkan hasil jika proses sudah selesai
    if st.session_state.embed_state["processing_completed"]:
        st.success("Watermarking process completed successfully!")

        # Tampilkan tombol untuk memulai proses baru
        if st.button("Start New Watermarking Process", key="new_process"):
            # Hapus file helper jika ada
            if st.session_state.embed_state["helper_filename"] and os.path.exists(
                st.session_state.embed_state["helper_filename"]
            ):
                try:
                    os.remove(st.session_state.embed_state["helper_filename"])
                except Exception:
                    pass

            # Reset state
            st.session_state.embed_state = {
                "processing_completed": False,
                "helper_filename": None,
                "helper_data_bytes": None,
                "watermarked_images": [],
                "not_watermarked_images": [],
            }
            st.rerun()

        # Tampilkan tombol download helper data
        st.header("Helper Data File")
        st.warning(
            "⚠️ IMPORTANT: Download Your Helper Data File\n\n"
            "This file is required to extract watermarks later. Without it, your watermarks cannot be recovered!"
        )

        if st.session_state.embed_state["helper_data_bytes"]:
            st.download_button(
                label="⬇️ DOWNLOAD HELPER DATA FILE",
                data=st.session_state.embed_state["helper_data_bytes"],
                file_name=(
                    os.path.basename(st.session_state.embed_state["helper_filename"])
                    if st.session_state.embed_state["helper_filename"]
                    else "helper_data.bin"
                ),
                mime="application/octet-stream",
                key="download_helper_state",
            )

            # Tampilkan informasi tentang file
            st.caption(
                f"File: {os.path.basename(st.session_state.embed_state['helper_filename'])}"
            )

        # Tampilkan gambar-gambar yang berhasil di-watermark
        watermarked_images = st.session_state.embed_state["watermarked_images"]
        if watermarked_images:
            st.header(f"✅ {len(watermarked_images)} images were watermarked")

            # Tombol download semua hasil sebagai ZIP
            if (
                "zip_data" in st.session_state.embed_state
                and st.session_state.embed_state["zip_data"]
            ):
                st.download_button(
                    label="⬇️ Download All Watermarked Images",
                    data=st.session_state.embed_state["zip_data"],
                    file_name="watermarked_images.zip",
                    mime="application/zip",
                    key="download_all_zip",
                )

            # Tampilkan semua gambar yang berhasil di-watermark
            cols = st.columns(min(3, len(watermarked_images)))
            for i, (name, img_bytes) in enumerate(watermarked_images):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    # Buat gambar dari bytes
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption=name, width=300)

                    # Tombol download untuk setiap gambar
                    st.download_button(
                        label="Download",
                        data=img_bytes,
                        file_name=f"watermarked_{name}",
                        mime="image/png",
                        key=f"download_img_{i}",
                    )

        # Tampilkan gambar-gambar yang tidak berhasil di-watermark
        not_watermarked_images = st.session_state.embed_state["not_watermarked_images"]
        if not_watermarked_images:
            st.header(f"❎ {len(not_watermarked_images)} images were not watermarked")
            st.caption("No matching face was found in these images")

            cols = st.columns(min(3, len(not_watermarked_images)))
            for i, (name, img_bytes) in enumerate(not_watermarked_images):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    # Buat gambar dari bytes
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption=name, width=300)

        # Keluar dari fungsi
        return

    # Step 1
    st.header("Step 1: Upload Authentication Face")
    st.markdown(
        "This is the face that will be used to secure your watermark. Only someone with a similar face can extract the watermark later."
    )

    auth_face = st.file_uploader(
        "Upload a clear image of your face",
        type=["jpg", "jpeg", "png"],
    )

    if auth_face:
        auth_img = Image.open(auth_face)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(auth_img, caption="Authentication Face", width=250)
        with col2:
            st.success("Face image uploaded successfully!")
            st.info(
                "Tips for best results:\n"
                "- Ensure your face is clearly visible and well-lit\n"
                "- Avoid wearing sunglasses or heavy makeup\n"
                "- Look directly at the camera"
            )

    # Step 2
    st.header("Step 2: Enter Watermark Text")
    st.markdown(
        "Enter the text you want to use as a watermark. This text will be encrypted and converted to a QR code."
    )

    watermark_text = st.text_area(
        "Enter text for watermark",
        max_chars=MAX_WATERMARK_CHARACTERS,
        height=100,
        key="watermark_text_input",
        placeholder="Enter your watermark text here (maximum 100 characters)",
    )

    # Tampilkan preview QR code jika teks telah dimasukkan
    if watermark_text:
        # Buat QR code preview
        qr_image = text_to_qrcode(
            watermark_text,
            (
                st.session_state.custom_settings.get("qrcode_size", QRCODE_SIZE),
                st.session_state.custom_settings.get("qrcode_size", QRCODE_SIZE),
            ),
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(qr_image, caption="QR Code Preview (before encryption)", width=200)
        with col2:
            st.success("Watermark text ready for embedding!")
            st.info(
                f"Character count: {len(watermark_text)}/{MAX_WATERMARK_CHARACTERS}\n"
                "The text will be encrypted and embedded as a QR code."
            )

    # Step 3
    st.header("Step 3: Upload Images to Watermark")
    st.markdown(
        "These are the images that will contain your invisible watermark.\n"
        "You can upload multiple images at once!"
    )

    uploaded_files = st.file_uploader(
        f"Select images to watermark (max {st.session_state.custom_settings.get(
            "max_images", MAX_IMAGES
        )} files)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded successfully!")
        if len(uploaded_files) > 10:
            st.warning(f"Using more than 10 images may take a while to process.")

        # Preview first few images
        if len(uploaded_files) > 0:
            st.subheader("Image Preview")
            preview_cols = st.columns(min(4, len(uploaded_files)))
            for i, file in enumerate(uploaded_files[:4]):
                with preview_cols[i]:
                    img = Image.open(file)
                    st.image(img, caption=file.name, width=200)
            if len(uploaded_files) > 4:
                st.caption(f"... and {len(uploaded_files)-4} more images")

    # Step 4
    st.header("Step 4: Watermarking Options")
    auth_required = st.checkbox(
        "Only watermark images that contain faces matching your authentication face",
        value=True,
    )

    if auth_required:
        st.info(
            "Smart mode activated: Only images containing a face matching your authentication face will be watermarked."
        )
    # Advanced Settings
    st.header("Advanced Settings")
    show_advanced = st.toggle("Show Advanced Settings", value=False)

    if show_advanced:
        st.warning(
            "⚠️ WARNING: Changing these settings may affect the watermark quality and extraction success rate. Only modify if you understand the implications."
        )

        with st.expander("Advanced Configuration"):
            st.subheader("1. Authentication Face Matching Tolerance")
            st.caption(
                "Adjust the slider below to control security vs. convenience:\n"
                "- Strict (1-30): High security - Requires nearly identical face images\n"
                "- Balanced (31-70): Good for most uses - Recommended for most people\n"
                "- Flexible (71-95): More convenient - Works with varied face images"
            )
            # Error Tolerance Setting
            error_tolerance = st.slider(
                "Authentication Face Matching Tolerance",
                min_value=1,
                max_value=95,
                value=ERROR_TOLERANCE,
                help=f"Higher values allow more variation in face images but may reduce security. Default: {ERROR_TOLERANCE}",
            )
            st.markdown("---")
            # QR Code Size Setting
            st.subheader("2. QR Code Size")
            st.caption(
                "Adjust the QR code size for the watermark:\n"
                "- Small (32-300): Less visible but may be harder to extract\n"
                "- Medium (301-700): Good balance of visibility and robustness\n"
                "- Large (701-1024): More robust but more visible in image"
            )
            custom_qrcode_size = st.slider(
                "QR Code Size",
                min_value=32,
                max_value=1024,
                value=QRCODE_SIZE,
                help=f"Size of the QR code watermark in pixels. Default: {QRCODE_SIZE}",
            )

            st.markdown("---")

            # Block Size Setting
            st.subheader("3. Block Size")
            st.caption(
                "Control how the watermark is embedded:\n"
                "- Small (4-8): Better image quality but less robust\n"
                "- Medium (9-12): Balanced quality and robustness\n"
                "- Large (13-16): More robust but may affect quality more"
            )
            custom_block_size = st.slider(
                "Block Size",
                min_value=4,
                max_value=16,
                value=BLOCK_SIZE,
                help=f"Size of image blocks for watermark embedding. Default: {BLOCK_SIZE}",
            )

            st.markdown("---")

            # Alpha Setting
            st.subheader("4. Watermark Strength")
            st.caption(
                "Set how strongly the watermark is embedded:\n"
                "- Light (0.1-0.3): Nearly invisible but less robust\n"
                "- Medium (0.4-0.7): Good balance for most uses\n"
                "- Strong (0.8-1.0): Very robust but may be noticeable"
            )
            custom_alpha = st.slider(
                "Watermark Strength (Alpha)",
                min_value=0.1,
                max_value=1.0,
                value=ALPHA,
                step=0.1,
                help=f"Strength of the watermark embedding. Default: {ALPHA}",
            )

            st.markdown("---")

            # Max Images Setting
            st.subheader("5. Maximum Images")
            st.caption(
                "Set batch processing limit:\n"
                "- Small batch (1-20): Fast processing\n"
                "- Medium batch (21-50): Balanced speed\n"
                "- Large batch (51-100): Slower but handles more images"
            )
            custom_max_images = st.slider(
                "Maximum Images",
                min_value=1,
                max_value=100,
                value=MAX_IMAGES,
                help=f"Maximum number of images to process at once. Default: {MAX_IMAGES}",
            )

            st.session_state.custom_settings.update(
                {
                    "qrcode_size": custom_qrcode_size,
                    "block_size": custom_block_size,
                    "alpha": custom_alpha,
                    "max_images": custom_max_images,
                    "error_tolerance": error_tolerance,
                }
            )
    
    # Process button
    st.write("")  # Add some space
    process_btn = st.button(
        "🚀 Process Images",
        disabled=not (auth_face and watermark_text and uploaded_files),
        key="process_btn",
    )

    if not (auth_face and watermark_text and uploaded_files):
        warning_message = "Please complete all required fields before processing:"
        missing_items = []
        if not auth_face:
            missing_items.append("- Upload an authentication face image")
        if not watermark_text:
            missing_items.append("- Enter watermark text")
        if not uploaded_files:
            missing_items.append("- Upload at least one image to watermark")

        st.warning(warning_message + "\n" + "\n".join(missing_items))

    if process_btn:
        # Validate inputs
        if not auth_face or not has_face(Image.open(auth_face)):
            st.error("Authentication image must contain a clearly visible face.")
        elif not watermark_text:
            st.error("Please enter watermark text.")
        elif not uploaded_files:
            st.error("Please upload at least one image to watermark.")
        elif len(uploaded_files) > st.session_state.custom_settings.get(
            "max_images", MAX_IMAGES
        ):
            st.error(
                f"Maximum {st.session_state.custom_settings.get('max_images', MAX_IMAGES)} images allowed. Please reduce the number of uploads."
            )
        else:
            try:
                # Process the authentication face
                auth_image_array = np.array(auth_img)

                # Get embedding for encryption key generation
                with st.spinner("🔍 Analyzing authentication face..."):
                    embedding = get_face_embedding(auth_image_array)

                if embedding is None:
                    st.error("Could not detect a face in the authentication image.")
                else:
                    # Generate encryption key and helper data using fuzzy extractor
                    with st.spinner(
                        "🔑 Generating secure encryption key from your face..."
                    ):
                        norm_tolerance = (
                            st.session_state.custom_settings.get(
                                "error_tolerance", ERROR_TOLERANCE
                            )
                            / 100.0
                        )
                        encryption_key, helper_data = generate_key_with_helper(
                            embedding, norm_tolerance
                        )

                    # Processing UI
                    st.subheader("🔮 Processing your images...")

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    watermarked = []
                    not_watermarked = []

                    # Create temporary directory for helper files
                    temp_helper_dir = "output/helpers"
                    if not os.path.exists(temp_helper_dir):
                        os.makedirs(temp_helper_dir)

                    # Save helper data to a temporary file
                    helper_filename = (
                        f"{temp_helper_dir}/helper_data_{int(time.time())}.bin"
                    )
                    save_helper_data(helper_data, helper_filename)

                    # Simpan helper data ke session_state
                    with open(helper_filename, "rb") as file:
                        helper_data_bytes = file.read()

                    # Simpan helper file info dalam session state
                    st.session_state.embed_state["helper_filename"] = helper_filename
                    st.session_state.embed_state["helper_data_bytes"] = (
                        helper_data_bytes
                    )

                    # Enkripsi teks watermark dan konversi ke QR code
                    with st.spinner("🔐 Encrypting watermark text..."):
                        encrypted_data = encrypt_watermark(
                            watermark_text, encryption_key
                        )
                        qr_watermark = text_to_qrcode(
                            encrypted_data,
                            (
                                st.session_state.custom_settings.get(
                                    "qrcode_size", QRCODE_SIZE
                                ),
                                st.session_state.custom_settings.get(
                                    "qrcode_size", QRCODE_SIZE
                                ),
                            ),
                        )

                        # Konversi ke bytes
                        qr_bytes = io.BytesIO()
                        qr_watermark.save(qr_bytes, format="PNG")
                        watermark_data = qr_bytes.getvalue()

                    for i, uploaded_file in enumerate(
                        uploaded_files[
                            : st.session_state.custom_settings.get(
                                "max_images", MAX_IMAGES
                            )
                        ]
                    ):
                        status_text.info(
                            f"Processing image {i + 1}/{len(uploaded_files[:st.session_state.custom_settings.get('max_images', MAX_IMAGES)])}: {uploaded_file.name[:30]}{'...' if len(uploaded_file.name) > 30 else ''}"
                        )
                        progress_value = (i + 1) / len(
                            uploaded_files[
                                : st.session_state.custom_settings.get(
                                    "max_images", MAX_IMAGES
                                )
                            ]
                        )
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
                            spinner_text = (
                                "🔍 Found matching face! Embedding watermark..."
                                if auth_required
                                else "💧 Embedding watermark..."
                            )
                            with st.spinner(spinner_text):
                                watermarked_img = embed_watermark(
                                    img, watermark_data, preserve_ratio=True
                                )
                                # Konversi ke bytes untuk disimpan dalam session state
                                img_bytes = io.BytesIO()
                                watermarked_img.save(img_bytes, format="PNG")
                                watermarked.append(
                                    (uploaded_file.name, watermarked_img)
                                )
                                # Simpan sebagai bytes dalam session_state
                                st.session_state.embed_state[
                                    "watermarked_images"
                                ].append((uploaded_file.name, img_bytes.getvalue()))
                        else:
                            spinner_text = (
                                "👤 No matching face found in image. Skipping..."
                                if auth_required
                                else "⏩ Skipping..."
                            )
                            with st.spinner(spinner_text):
                                # Konversi ke bytes untuk disimpan dalam session state
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format="PNG")
                                not_watermarked.append((uploaded_file.name, img))
                                # Simpan sebagai bytes dalam session_state
                                st.session_state.embed_state[
                                    "not_watermarked_images"
                                ].append((uploaded_file.name, img_bytes.getvalue()))

                        time.sleep(0.1)  # Small delay for UI feedback

                    # Buat ZIP dari semua gambar yang di-watermark
                    if watermarked:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                            for name, img in watermarked:
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format="PNG")
                                zip_file.writestr(
                                    f"watermarked_{name}", img_bytes.getvalue()
                                )

                        # Simpan data ZIP dalam session state
                        st.session_state.embed_state["zip_data"] = zip_buffer.getvalue()

                    # Tandai bahwa proses telah selesai
                    st.session_state.embed_state["processing_completed"] = True

                    # Show results
                    status_text.empty()
                    progress_bar.empty()

                    with results_container:
                        st.title("🎉 Processing Complete!")

                        # Download helper data file - make it very prominent
                        with st.container():
                            st.warning(
                                "⚠️ IMPORTANT: Download Your Helper Data File\n\n"
                                "This file is required to extract your watermarks later. Without it, your watermarks cannot be recovered!"
                            )

                            # Baca file helper data
                            helper_data_bytes = None
                            helper_filename_to_use = (
                                st.session_state.helper_filename
                                if "helper_filename" in st.session_state
                                else helper_filename
                            )

                            if os.path.exists(helper_filename_to_use):
                                with open(helper_filename_to_use, "rb") as file:
                                    helper_data_bytes = file.read()

                            if helper_data_bytes:
                                st.download_button(
                                    label="⬇️ DOWNLOAD HELPER DATA FILE",
                                    data=helper_data_bytes,
                                    file_name=os.path.basename(helper_filename_to_use),
                                    mime="application/octet-stream",
                                    help="This file is required to extract watermarks later. Keep it safe!",
                                    key="download_helper",
                                )

                                # Tambahkan informasi tentang file
                                st.caption(
                                    f"File: {os.path.basename(helper_filename_to_use)}"
                                )
                            else:
                                st.error(
                                    "Unable to load helper data file. Please try again."
                                )

                        # Display watermarked images
                        if watermarked:
                            st.subheader(
                                f"✅ {len(watermarked)} images were watermarked"
                            )

                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                                for name, img in watermarked:
                                    img_bytes = io.BytesIO()
                                    img.save(img_bytes, format="PNG")
                                    zip_file.writestr(
                                        f"watermarked_{name}", img_bytes.getvalue()
                                    )

                            st.download_button(
                                label="⬇️ Download All Watermarked Images",
                                data=zip_buffer.getvalue(),
                                file_name="watermarked_images.zip",
                                mime="application/zip",
                                help="Download all watermarked images in ZIP format",
                            )

                            # Create a grid layout for images
                            cols = st.columns(min(3, len(watermarked)))

                            for i, (name, img) in enumerate(watermarked):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    st.image(img, caption=name, width=300)

                                    # Save button for each image
                                    watermarked_img_bytes = io.BytesIO()
                                    img.save(watermarked_img_bytes, format="PNG")
                                    st.download_button(
                                        label="Download",
                                        data=watermarked_img_bytes.getvalue(),
                                        file_name=f"watermarked_{name}",
                                        mime="image/png",
                                        key=f"watermarked_download_btn_{i}",
                                    )

                        # List images that were not watermarked
                        if not_watermarked:
                            st.subheader(
                                f"❎ {len(not_watermarked)} images were not watermarked"
                            )
                            st.caption("No matching face was found in these images")

                            cols = st.columns(min(3, len(not_watermarked)))

                            for i, (name, img) in enumerate(not_watermarked):
                                col_idx = i % len(cols)
                                with cols[col_idx]:
                                    st.image(img, caption=name, width=300)

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                if os.path.exists(helper_filename):
                    os.remove(helper_filename)

    # Add some space at the bottom
    st.write("")
    st.write("")
