import io
import time

import streamlit as st
from PIL import Image
import base64

from modules.watermarking import extract_watermark
from modules.encryption import decrypt_watermark
from modules.qrcode_generator import qrcode_to_text

# Kunci default untuk enkripsi (32 byte), harus sama dengan di embed_test
DEFAULT_KEY = b"0123456789abcdef0123456789abcdef"


def display_watermark_extract_test():
    st.subheader("Watermark Extracting Test")
    st.markdown(
        """
    Watermark extraction test:
    
    1. Upload original image
    2. Upload watermarked image
    3. System will extract watermark, decode QR code, decrypt text, and display results
    """
    )

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
            st.image(watermarked_img, caption="Watermarked Image", use_container_width=True)

    # Extract watermark button - disabled if either image is not uploaded
    if st.button("Extract Watermark", disabled=not (original_img and watermarked_img)):
        with st.spinner("Extracting watermark..."):
            start_time = time.time()

            # Extract watermark (QR code image)
            extracted_qr_img = extract_watermark(watermarked_img, original_img)

            # Selalu tampilkan gambar yang diupload
            st.subheader("Watermark Extraction Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Original Image**")
                st.image(original_img, use_container_width=True)

            with col2:
                st.markdown("**Watermarked Image**")
                st.image(watermarked_img, use_container_width=True)

            with col3:
                st.markdown("**Extracted QR Code**")
                if extracted_qr_img:
                    st.image(
                        extracted_qr_img,
                        caption="Extracted QR Code",
                        use_container_width=True,
                    )
                else:
                    st.error("No QR code extracted")

            # Proses ekstraksi QR code dan dekripsi
            if extracted_qr_img:
                # Decode QR code to get encrypted text
                encrypted_data = qrcode_to_text(extracted_qr_img)

                # Measure QR extraction time
                qr_extraction_time = time.time() - start_time

                # Decrypt the text
                if encrypted_data:
                    decrypted_text = decrypt_watermark(encrypted_data, DEFAULT_KEY)

                    # Measure total processing time
                    processing_time = time.time() - start_time

                    if decrypted_text:
                        decrypted_text = decrypted_text.decode("utf-8")
                        st.success(
                            f"Watermark successfully extracted and decrypted in {processing_time:.3f} seconds!"
                        )

                        # Display extracted text
                        st.subheader("Extracted Watermark Text")
                        st.text_area(
                            "Decrypted Text",
                            value=decrypted_text,
                            height=150,
                            disabled=True,
                        )

                        # Download extracted QR code
                        img_byte_arr = io.BytesIO()
                        extracted_qr_img.save(img_byte_arr, format="PNG")

                        st.download_button(
                            label="Download QR Code Image",
                            data=img_byte_arr.getvalue(),
                            file_name="extracted_qrcode.png",
                            mime="image/png",
                        )

                        # Display statistics
                        st.subheader("Extraction Statistics")
                        st.markdown(
                            f"**QR Code Extraction Time:** {qr_extraction_time:.3f} seconds"
                        )
                        st.markdown(
                            f"**Text Decryption Time:** {processing_time - qr_extraction_time:.3f} seconds"
                        )
                        st.markdown(
                            f"**Total Processing Time:** {processing_time:.3f} seconds"
                        )
                        watermark_width, watermark_height = extracted_qr_img.size
                        original_width, original_height = original_img.size
                        st.markdown(
                            f"**Original Image Size:** {original_width}x{original_height} pixels"
                        )
                        st.markdown(
                            f"**QR Code Size:** {watermark_width}x{watermark_height} pixels"
                        )
                        st.markdown(
                            f"**Text Length:** {len(decrypted_text)} characters"
                        )
                    else:
                        st.error(
                            "Failed to decrypt watermark text. The QR code might be damaged or invalid."
                        )
                else:
                    st.error("Failed to decode QR code from extracted watermark.")
            else:
                st.error("Failed to extract watermark from image.")
                st.markdown(
                    """
                This may be due to:
                - Image does not contain a watermark
                - Watermark damaged due to image modification
                - Watermark embedded with different algorithm
                """
                )
