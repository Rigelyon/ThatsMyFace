import io

import numpy as np
import streamlit as st
from PIL import Image

from modules.encryption import decrypt_watermark
from modules.face_recognition import get_face_embedding
from modules.fuzzy_extractor import regenerate_key_from_helper
from modules.utils import deserialize_helper_data, has_face
from modules.watermarking import extract_watermark
from modules.qrcode_generator import qrcode_to_text


def display_extract_watermark_page(debug_mode=False):
    # Header
    st.title("🔍 Extract Watermark")
    st.markdown(
        "Verify ownership and reveal hidden watermarks with facial authentication"
    )

    # Menggunakan struktur yang lebih baik untuk penanganan state
    if "extract_state" not in st.session_state:
        st.session_state.extract_state = {
            "extraction_completed": False,
            "original_image": None,
            "watermarked_image": None,
            "extracted_watermark": None,
            "extracted_watermark_bytes": None,
            "extracted_text": None,
            "results": {},
        }

    # Tampilkan hasil jika proses ekstraksi sebelumnya berhasil
    if st.session_state.extract_state["extraction_completed"] and (
        st.session_state.extract_state["extracted_watermark_bytes"]
        or st.session_state.extract_state["extracted_text"]
    ):
        st.success("Watermark extraction successful!")

        # Tombol untuk memulai ekstraksi baru
        if st.button("Extract Another Watermark"):
            # Reset state
            st.session_state.extract_state = {
                "extraction_completed": False,
                "original_image": None,
                "watermarked_image": None,
                "extracted_watermark": None,
                "extracted_watermark_bytes": None,
                "extracted_text": None,
                "results": {},
            }
            st.rerun()

        # Tampilkan hasil ekstraksi
        st.header("Extraction Results")

        col1, col2, col3 = st.columns(3)

        # Tampilkan gambar original
        if st.session_state.extract_state["original_image"]:
            with col1:
                st.subheader("Original Image")
                original_img = Image.open(
                    io.BytesIO(st.session_state.extract_state["original_image"])
                )
                st.image(original_img, use_container_width=True)

        # Tampilkan gambar watermarked
        if st.session_state.extract_state["watermarked_image"]:
            with col2:
                st.subheader("Watermarked Image")
                watermarked_img = Image.open(
                    io.BytesIO(st.session_state.extract_state["watermarked_image"])
                )
                st.image(watermarked_img, use_container_width=True)

        # Tampilkan watermark yang diekstrak
        if st.session_state.extract_state["extracted_watermark_bytes"]:
            with col3:
                st.subheader("Extracted QR Code")
                extracted_img = Image.open(
                    io.BytesIO(
                        st.session_state.extract_state["extracted_watermark_bytes"]
                    )
                )
                st.image(extracted_img, use_container_width=True)

                # Tombol download QR code
                st.download_button(
                    label="⬇️ Download QR Code",
                    data=st.session_state.extract_state["extracted_watermark_bytes"],
                    file_name="extracted_qrcode.png",
                    mime="image/png",
                    key="download_extracted",
                )

        # Tampilkan teks watermark yang diekstrak
        if st.session_state.extract_state["extracted_text"]:
            st.subheader("Extracted Watermark Text")
            st.text_area(
                "Decrypted text:",
                value=st.session_state.extract_state["extracted_text"],
                height=150,
                disabled=True,
            )

        # Keluar dari fungsi untuk tidak menampilkan form
        return

    # Step 1
    st.header("Step 1: Upload Authentication Face")
    st.markdown(
        "How this works: Upload a picture of your face to authenticate and extract the watermark. "
        "This should be the same person who embedded the watermark, but doesn't have to be the exact same photo."
    )

    auth_face = st.file_uploader(
        "Upload your face image for authentication",
        type=["jpg", "jpeg", "png"],
    )

    if auth_face:
        auth_img = Image.open(auth_face)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(auth_img, caption="Authentication Face", use_container_width=True)
        with col2:
            st.success("Face image uploaded successfully!")
            st.info(
                "Authentication Tips:\n"
                "- Make sure your face is clearly visible\n"
                "- Similar lighting conditions to original registration help\n"
                "- If extraction fails, try another photo with better lighting"
            )

    # Step 2
    st.header("Step 2: Upload Helper Data File")
    st.markdown(
        "What's this? This is the helper data file you downloaded when you embedded the watermark. "
        "This file is essential for extracting your watermark."
    )

    helper_file = st.file_uploader("Upload the helper data file (.bin)", type=["bin"])

    if helper_file:
        st.success("Helper data file uploaded successfully!")

    # Step 3
    st.header("Step 3: Upload Images")
    st.markdown(
        "Upload both the original and watermarked images to extract the hidden watermark."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 Original Image")
        st.caption("The unwatermarked version of your image")

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
        st.subheader("💧 Watermarked Image")
        st.caption("The image with the hidden watermark")

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

    # Extract watermark button with improved styling
    st.write("")
    extract_btn_disabled = not (auth_face and helper_file and watermarked_file)

    if extract_btn_disabled:
        warning_message = "Please complete all required fields before extracting:"
        missing_items = []
        if not auth_face:
            missing_items.append("- Upload an authentication face image")
        if not helper_file:
            missing_items.append("- Upload the helper data file")
        if not original_file:
            missing_items.append("- Upload the original image")
        if not watermarked_file:
            missing_items.append("- Upload the watermarked image")

        st.warning(warning_message + "\n" + "\n".join(missing_items))

    extract_btn = st.button(
        "🔍 Extract Watermark", disabled=extract_btn_disabled, key="extract_btn"
    )

    if extract_btn:
        if not auth_face or not has_face(Image.open(auth_face)):
            st.error("Authentication image must contain a clearly visible face.")
        elif not original_file:
            st.error("Please upload the original image.")
        elif not watermarked_file:
            st.error("Please upload the watermarked image.")
        elif not helper_file:
            st.error("Please upload the helper data file.")
        else:
            with st.spinner("✨ Extracting your watermark..."):
                try:
                    # Process the authentication face
                    auth_image_array = np.array(auth_img)

                    # Get embedding for encryption key generation
                    with st.spinner("🔍 Analyzing your face..."):
                        embedding = get_face_embedding(auth_image_array)

                    if embedding is None:
                        st.error(
                            "Face Detection Failed - Could not detect a face in the authentication image. Please try with a clearer image."
                        )
                    else:
                        # Load helper data
                        with st.spinner("🔐 Loading security data..."):
                            helper_data = deserialize_helper_data(
                                helper_file.getvalue()
                            )

                        # Debug: Show helper data information
                        if debug_mode:
                            st.write(
                                "DEBUG: Helper data loaded:",
                                {
                                    "error_tolerance": helper_data.get(
                                        "error_tolerance"
                                    ),
                                    "vector_shape": helper_data.get("vector_shape"),
                                    "vector_mean": helper_data.get("vector_mean"),
                                    "vector_std": helper_data.get("vector_std"),
                                },
                            )
                            st.write(
                                "DEBUG: Current face embedding stats:",
                                {
                                    "mean": float(np.mean(embedding)),
                                    "std": float(np.std(embedding)),
                                    "min": float(np.min(embedding)),
                                    "max": float(np.max(embedding)),
                                    "shape": embedding.shape,
                                },
                            )

                        # Generate encryption key from face embedding using helper data
                        with st.spinner("🔑 Verifying your identity..."):
                            decryption_key = regenerate_key_from_helper(
                                embedding, helper_data
                            )

                        if decryption_key is b"":
                            st.error(
                                "Authentication Failed - Could not verify your identity. The face may be too different from the one used for embedding."
                            )
                            st.info(
                                "Try using a clearer photo in better lighting.\n"
                                "Make sure it's the same person who embedded the watermark.\n"
                                "Try different facial expressions or angles."
                            )
                        else:
                            # Debug: Show key information
                            if debug_mode:
                                st.write(
                                    "DEBUG: Regenerated key (first 8 bytes hex):",
                                    decryption_key[:8].hex(),
                                )

                            if debug_mode:
                                st.write("DEBUG: Attempting to extract watermark...")

                            with st.spinner(
                                "🔍 Analyzing image for hidden watermarks..."
                            ):
                                extracted_qrcode = extract_watermark(
                                    watermarked_img,
                                    original_img,
                                )

                            if extracted_qrcode:
                                if debug_mode:
                                    st.write("DEBUG: QR code extraction successful!")

                                # Simpan hasil gambar ke session state (selalu, jika QR code berhasil diekstrak)
                                img_bytes = io.BytesIO()
                                extracted_qrcode.save(img_bytes, format="PNG")
                                st.session_state.extract_state[
                                    "extracted_watermark_bytes"
                                ] = img_bytes.getvalue()
                                st.session_state.extract_state[
                                    "extracted_watermark"
                                ] = extracted_qrcode
                                st.session_state.extract_state[
                                    "extraction_completed"
                                ] = True

                                original_bytes = io.BytesIO()
                                watermarked_bytes = io.BytesIO()
                                original_img.save(original_bytes, format="PNG")
                                watermarked_img.save(watermarked_bytes, format="PNG")
                                st.session_state.extract_state["original_image"] = (
                                    original_bytes.getvalue()
                                )
                                st.session_state.extract_state["watermarked_image"] = (
                                    watermarked_bytes.getvalue()
                                )

                                # Tampilkan hasil gambar (selalu tampil jika QR code berhasil diekstrak)
                                st.title("🎉 Extraction Result")
                                st.subheader("Extraction Results")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.subheader("📄 Original Image")
                                    st.image(original_img, use_container_width=True)
                                with col2:
                                    st.subheader("💧 Watermarked Image")
                                    st.image(watermarked_img, use_container_width=True)
                                with col3:
                                    st.subheader("✨ Extracted QR Code")
                                    st.image(extracted_qrcode, use_container_width=True)

                                st.download_button(
                                    label="⬇️ Download QR Code",
                                    data=st.session_state.extract_state[
                                        "extracted_watermark_bytes"
                                    ],
                                    file_name="extracted_qrcode.png",
                                    mime="image/png",
                                    key="download_qrcode",
                                )

                                # Dekode QR code & dekripsi teks watermark
                                with st.spinner(
                                    "🔓 Decoding and decrypting watermark..."
                                ):
                                    try:
                                        # Dekode QR code untuk mendapatkan data enkripsi
                                        encrypted_data = qrcode_to_text(
                                            extracted_qrcode
                                        )

                                        if encrypted_data:
                                            # Dekripsi data
                                            decrypted_text = decrypt_watermark(
                                                encrypted_data, decryption_key
                                            )

                                            if decrypted_text:
                                                decrypted_text = decrypted_text.decode(
                                                    "utf-8"
                                                )
                                                st.subheader("Extracted Watermark Text")
                                                st.text_area(
                                                    "Decrypted text:",
                                                    value=decrypted_text,
                                                    height=150,
                                                    disabled=True,
                                                )
                                                st.session_state.extract_state[
                                                    "extracted_text"
                                                ] = decrypted_text
                                            else:
                                                st.error("Decryption Failed")
                                                st.info(
                                                    "Failed to decrypt the watermark text. This usually happens when:\n"
                                                    "- The face doesn't match closely enough with the one used for embedding\n"
                                                    "- The error tolerance was set too low during embedding\n"
                                                    "- The helper data file doesn't match this watermarked image\n"
                                                )
                                        else:
                                            st.error("QR Code Decoding Failed")
                                            st.info(
                                                "Could not decode QR code from extracted watermark. The watermark might be damaged."
                                            )
                                    except Exception as e:
                                        st.error(
                                            f"Decryption Error - Failed to decrypt the watermark: {str(e)}"
                                        )
                            else:
                                st.error("No Watermark Found")
                                st.info(
                                    "Could not extract a watermark from this image. This could be because:\n"
                                    "- The image doesn't contain a watermark\n"
                                    "- The image wasn't watermarked with this application\n"
                                    "- The image was modified after watermarking (resized, cropped, compressed)"
                                )

                except Exception as e:
                    st.error(
                        f"Extraction Error - An error occurred during extraction: {str(e)}"
                    )
                    st.info(
                        "Please try again with different images or contact support if the problem persists."
                    )

    # Add some space at the bottom
    st.write("")
    st.write("")
