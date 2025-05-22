import io
import time

import streamlit as st
from PIL import Image
from Crypto.Random import get_random_bytes

from modules.watermarking import embed_watermark
from modules.encryption import encrypt_watermark
from modules.qrcode_generator import text_to_qrcode


# Kunci default untuk enkripsi (32 byte)
DEFAULT_KEY = b"0123456789abcdef0123456789abcdef"


def display_watermark_embed_test():
    st.subheader("Watermark Embedding Test")
    st.markdown(
        """
    This test allows you to test the watermark embedding functionality and see statistics.
    
    1. Upload an image to watermark
    2. Enter text watermark
    3. The system will encrypt the text, convert it to QR code, and embed the QR code as watermark
    4. The watermarked image will be displayed with statistics
    """
    )

    # Image to watermark
    st.markdown("### Image to Watermark")
    image_file = st.file_uploader(
        "Upload image to watermark", type=["jpg", "jpeg", "png"], key="embed_test_image"
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Original Image", use_container_width=True)

    # Text watermark section
    st.markdown("### Text Watermark")
    st.markdown(
        """
    **Text watermark requirements:**
    - Maximum 100 characters
    - The text will be encrypted and converted to QR code
    """
    )

    watermark_text = st.text_area(
        "Enter text for watermark",
        max_chars=100,
        height=100,
        key="watermark_text_input",
    )

    qrcode_size = st.slider(
        "QR Code size (pixels)",
        min_value=100,
        max_value=1000,
        value=300,
        step=100,
        help="Size of QR code in pixels (width x height)",
    )

    # Preview QR code if text is entered
    if watermark_text:
        # Encrypt the text
        encrypted_data = encrypt_watermark(watermark_text, DEFAULT_KEY)

        # Convert to QR code
        qr_image = text_to_qrcode(encrypted_data, (qrcode_size, qrcode_size))

        # Display QR code preview
        st.image(qr_image, caption="QR Code Preview", width=300)

        # Convert QR code to bytes for embedding
        img_byte_arr = io.BytesIO()
        qr_image.save(img_byte_arr, format="PNG")
        watermark_data = img_byte_arr.getvalue()
    else:
        watermark_data = None

    # Embed watermark button
    if st.button("Embed Watermark", disabled=not (image_file and watermark_text)):
        with st.spinner("Embedding watermark..."):
            start_time = time.time()

            # Encrypt the text
            encrypted_data = encrypt_watermark(watermark_text, DEFAULT_KEY)

            # Convert to QR code
            qr_image = text_to_qrcode(encrypted_data, (qrcode_size, qrcode_size))

            # Convert QR code to bytes for embedding
            img_byte_arr = io.BytesIO()
            qr_image.save(img_byte_arr, format="PNG")
            watermark_data = img_byte_arr.getvalue()

            # Embed watermark
            watermarked_img = embed_watermark(image, watermark_data)

            processing_time = time.time() - start_time

            # Display results
            st.success(
                f"Watermark embedded successfully in {processing_time:.3f} seconds!"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("### Watermarked Image")
                st.image(watermarked_img, use_container_width=True)

            # Display QR code that was embedded
            st.markdown("### Embedded QR Code")
            st.image(qr_image, caption="QR Code from Text Watermark", width=300)

            # Download watermarked image
            buffered = io.BytesIO()
            watermarked_img.save(buffered, format="PNG")
            st.download_button(
                label="Download Watermarked Image",
                data=buffered.getvalue(),
                file_name=f"watermarked_{image_file.name}",
                mime="image/png",
            )
