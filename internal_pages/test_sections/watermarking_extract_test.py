import io
import time

import streamlit as st
from PIL import Image

from modules.watermarking import extract_watermark


def display_watermark_extract_test():
    st.subheader("Watermark Extracting Test")
    st.markdown(
        """
    Watermark extraction test:
    
    1. Upload original image
    2. Upload watermarked image
    3. System will extract watermark and display results
    """
    )

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

    # Upload watermarked image
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

            # Extract watermark
            extracted_img = extract_watermark(watermarked_img, original_img)

            processing_time = time.time() - start_time

            if extracted_img:
                st.success(
                    f"Watermark successfully extracted in {processing_time:.3f} seconds!"
                )

                st.subheader("Watermark Extraction Results")

                # Display 3 images: original, watermarked, and extracted
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
                        extracted_img,
                        caption="Extracted Watermark",
                        use_container_width=True,
                    )

                # Download extracted watermark image
                img_byte_arr = io.BytesIO()
                extracted_img.save(img_byte_arr, format="PNG")

                st.download_button(
                    label="Download Watermark Image",
                    data=img_byte_arr.getvalue(),
                    file_name="extracted_watermark.png",
                    mime="image/png",
                )

                # Display statistics
                st.subheader("Extraction Statistics")
                st.markdown(f"**Extraction Time:** {processing_time:.3f} seconds")
                watermark_width, watermark_height = extracted_img.size
                original_width, original_height = original_img.size
                st.markdown(
                    f"**Original Image Size:** {original_width}x{original_height} pixels"
                )
                st.markdown(
                    f"**Watermark Size:** {watermark_width}x{watermark_height} pixels"
                )
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
