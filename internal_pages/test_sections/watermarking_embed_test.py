import io
import time

import numpy as np
import streamlit as st
from PIL import Image

from modules.constants import ALPHA
from modules.watermarking import embed_watermark


def display_watermark_embed_test():
    st.subheader("Watermark Embedding Test")
    st.markdown("""
    This test allows you to test the watermark embedding functionality and see statistics.
    
    1. Upload an image to watermark
    2. Upload or enter a watermark (text or image)
    3. Adjust the watermark strength (Alpha)
    4. The system will embed the watermark and show statistics
    """)

    # Image to watermark
    st.markdown("### Image to Watermark")
    image_file = st.file_uploader("Upload image to watermark", type=["jpg", "jpeg", "png"], key="embed_test_image")
    
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Original Image", use_container_width=True)

    # Watermark section
    st.markdown("### Watermark")
    watermark_type = st.radio("Watermark Type", ["Text", "Image"], key="embed_test_watermark_type")
    
    watermark_data = None
    if watermark_type == "Text":
        watermark_text = st.text_area("Enter text watermark", value="Test Watermark", key="embed_test_watermark_text")
        if watermark_text:
            watermark_data = watermark_text.encode('utf-8')
    else:  # Image watermark
        watermark_file = st.file_uploader("Upload image watermark", type=["jpg", "jpeg", "png"], key="embed_test_watermark_image")
        if watermark_file:
            watermark_image = Image.open(watermark_file)
            st.image(watermark_image, caption="Watermark Image", use_container_width=True, width=200)
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            watermark_image.save(img_byte_arr, format=watermark_image.format if watermark_image.format else 'PNG')
            watermark_data = img_byte_arr.getvalue()

    # Watermark strength
    st.markdown("### Watermark Strength")
    alpha_value = st.slider(
        "Alpha (strength)",
        min_value=0.01,
        max_value=0.5,
        value=ALPHA,
        step=0.01,
        key="embed_test_alpha",
        help="Higher values make watermark more robust but may affect image quality"
    )

    # Embed watermark button
    if st.button("Embed Watermark", disabled=not (image_file and watermark_data)):
        with st.spinner("Embedding watermark..."):
            start_time = time.time()
            
            # Embed watermark
            watermarked_img = embed_watermark(image, watermark_data)
            
            processing_time = time.time() - start_time
            
            # Display results
            st.success(f"Watermark embedded successfully in {processing_time:.3f} seconds!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
            
            with col2:
                st.markdown("### Watermarked Image")
                st.image(watermarked_img, use_container_width=True)
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            original_array = np.array(image).astype(np.float32)
            watermarked_array = np.array(watermarked_img).astype(np.float32)
            
            # Ensure same dimensions for comparison
            min_height = min(original_array.shape[0], watermarked_array.shape[0])
            min_width = min(original_array.shape[1], watermarked_array.shape[1])
            
            # Crop both images to the same dimensions
            original_array = original_array[:min_height, :min_width]
            watermarked_array = watermarked_array[:min_height, :min_width]
            
            # Calculate MSE (Mean Squared Error)
            mse = np.mean((original_array - watermarked_array) ** 2)
            if mse == 0:  # Images are identical
                psnr = float('inf')
            else:
                max_pixel = 255.0
                psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            
            # Display statistics
            st.subheader("Watermarking Statistics")
            st.markdown(f"**PSNR:** {psnr:.2f} dB (higher is better)")
            st.markdown(f"**MSE:** {mse:.2f} (lower is better)")
            st.markdown(f"**Watermark Size:** {len(watermark_data)} bytes")
            st.markdown(f"**Alpha Value:** {alpha_value}")
            
            # Download watermarked image
            buffered = io.BytesIO()
            watermarked_img.save(buffered, format="PNG")
            st.download_button(
                label="Download Watermarked Image",
                data=buffered.getvalue(),
                file_name=f"watermarked_{image_file.name}",
                mime="image/png"
            )