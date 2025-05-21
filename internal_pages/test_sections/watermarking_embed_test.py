import io
import time

import streamlit as st
from PIL import Image

from modules.constants import ALPHA, MAX_WATERMARK_RESOLUTION, MAX_WATERMARK_SIZE
from modules.watermarking import embed_watermark


def display_watermark_embed_test():
    st.subheader("Watermark Embedding Test")
    st.markdown(
        """
    This test allows you to test the watermark embedding functionality and see statistics.
    
    1. Upload an image to watermark
    2. Upload a watermark image (PNG, JPG, JPEG format)
    3. Adjust the watermark strength (Alpha)
    4. The system will embed the watermark and show statistics
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

    # Watermark section
    st.markdown("### Watermark")
    st.markdown(
        """
    **Watermark image requirements:**
    - Format: JPG, JPEG, or PNG
    - File size: Maximum 2MB
    - Resolution: Not more than 1000x1000 pixels
    """
    )

    watermark_data = None
    watermark_file = st.file_uploader(
        "Upload image watermark",
        type=["jpg", "jpeg", "png"],
        key="embed_test_watermark_image",
    )

    if watermark_file:
        # Validate file size (2MB = 2 * 1024 * 1024 bytes)
        if watermark_file.size > 2 * MAX_WATERMARK_SIZE * MAX_WATERMARK_SIZE:
            st.error(f"Watermark file size too large! Max {MAX_WATERMARK_SIZE}x{MAX_WATERMARK_SIZE} pixels.")
        else:
            watermark_image = Image.open(watermark_file)

            # Validate resolution
            if watermark_image.width > MAX_WATERMARK_RESOLUTION or watermark_image.height > MAX_WATERMARK_RESOLUTION:
                st.error(
                    f"Watermark image resolution too large ({watermark_image.width}x{watermark_image.height})! Maximum {MAX_WATERMARK_RESOLUTION}x{MAX_WATERMARK_RESOLUTION} pixels."
                )
            else:
                st.image(
                    watermark_image,
                    caption=f"Watermark Image ({watermark_image.width}x{watermark_image.height})",
                    use_container_width=True,
                    width=200,
                )
                st.success(
                    f"Valid watermark image: {watermark_file.name} ({watermark_image.width}x{watermark_image.height} pixels)"
                )

                img_byte_arr = io.BytesIO()
                watermark_image.save(
                    img_byte_arr,
                    format=watermark_image.format if watermark_image.format else "PNG",
                )
                watermark_data = img_byte_arr.getvalue()

    # Embed watermark button
    if st.button("Embed Watermark", disabled=not (image_file and watermark_data)):
        with st.spinner("Embedding watermark..."):
            start_time = time.time()

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

            # Download watermarked image
            buffered = io.BytesIO()
            watermarked_img.save(buffered, format="PNG")
            st.download_button(
                label="Download Watermarked Image",
                data=buffered.getvalue(),
                file_name=f"watermarked_{image_file.name}",
                mime="image/png",
            )
