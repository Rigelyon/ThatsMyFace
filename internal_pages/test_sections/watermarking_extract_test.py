import io
import time

import streamlit as st
from PIL import Image

from modules.watermarking import extract_watermark, detect_watermark


def display_watermark_extract_test():
    st.subheader("Watermark Extracting Test")
    st.markdown("""
    This test allows you to test the watermark extraction functionality.
    
    1. Upload a watermarked image
    2. The system will attempt to extract the watermark and show statistics
    """)

    # Upload watermarked image
    st.markdown("### Watermarked Image")
    watermarked_file = st.file_uploader("Upload watermarked image", type=["jpg", "jpeg", "png"], key="extract_test_image")
    
    if watermarked_file:
        watermarked_img = Image.open(watermarked_file)
        st.image(watermarked_img, caption="Watermarked Image", use_container_width=True)

    # Extract watermark button
    if st.button("Extract Watermark", disabled=not watermarked_file):
        with st.spinner("Extracting watermark..."):
            start_time = time.time()
            
            # First check if the image likely contains a watermark
            has_watermark = detect_watermark(watermarked_img)
            
            # Extract watermark
            extracted_data = extract_watermark(watermarked_img)
            
            processing_time = time.time() - start_time
            
            # Display results
            if has_watermark:
                st.success(f"Image appears to contain a watermark (confidence: {has_watermark:.2f})")
            else:
                st.warning(f"Image does not appear to contain a watermark (confidence: {1-has_watermark:.2f})")
            
            if extracted_data:
                st.success(f"Watermark extracted successfully in {processing_time:.3f} seconds!")
                
                # Determine if watermark is text or image
                try:
                    # Try to decode as text
                    watermark_text = extracted_data.decode('utf-8')
                    st.subheader("Extracted Text Watermark")
                    st.text_area("Extracted Text", watermark_text, height=150)
                except UnicodeDecodeError:
                    # If not text, try as image
                    try:
                        watermark_img = Image.open(io.BytesIO(extracted_data))
                        st.subheader("Extracted Image Watermark")
                        st.image(watermark_img, caption="Extracted Watermark", use_container_width=True)
                    except Exception:
                        st.error("Could not interpret the extracted watermark as text or image.")
                        st.markdown("**Raw extracted data (hex):**")
                        st.code(extracted_data.hex()[:100] + "..." if len(extracted_data.hex()) > 100 else extracted_data.hex())
                
                # Display statistics
                st.subheader("Extraction Statistics")
                st.markdown(f"**Extraction Time:** {processing_time:.3f} seconds")
                st.markdown(f"**Extracted Data Size:** {len(extracted_data)} bytes")
                
                # Download extracted data
                st.download_button(
                    label="Download Extracted Data",
                    data=extracted_data,
                    file_name="extracted_watermark",
                    mime="application/octet-stream"
                )
            else:
                st.error("Failed to extract watermark from the image.")
                st.markdown("""
                This could be due to:
                - The image doesn't contain a watermark
                - The watermark was damaged by image modifications
                - The watermark was embedded with a different algorithm
                """)