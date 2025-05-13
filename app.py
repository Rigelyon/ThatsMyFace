import streamlit as st
import os
from pages.embed_watermark_page import display_embed_watermark_page
from pages.extract_watermark_page import display_extract_watermark_page

# Set page configuration
st.set_page_config(
    page_title="That's My Face - Image Watermarking with Facial Authentication",
    page_icon="🔐",
    layout="wide",
)

# App title and description
st.title("🔐 That's My Face")
st.subheader("Image Watermarking with Facial Authentication")
st.markdown("""
This application allows you to watermark images using DCT and SVD techniques.
The watermark is encrypted using AES, with the encryption key derived from facial recognition embeddings.
""")

# Create necessary directories
if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("output"):
    os.makedirs("output")

# Sidebar for options
st.sidebar.title("Options")
process_choice = st.sidebar.radio(
    "Choose operation:",
    ["Embed Watermark", "Extract Watermark"]
)

# Debug mode toggle
st.sidebar.markdown("---")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
if debug_mode:
    st.sidebar.info("Debug mode is enabled. Additional technical information will be displayed.")
    os.environ['WATERMARK_DEBUG'] = '1'
else:
    os.environ['WATERMARK_DEBUG'] = '0'

# Main application flow
if process_choice == "Embed Watermark":
    display_embed_watermark_page(debug_mode=debug_mode)

elif process_choice == "Extract Watermark":
    display_extract_watermark_page(debug_mode=debug_mode)

# Footer
st.markdown("---")
st.markdown("**That's My Face** - Secure Image Watermarking with Facial Authentication")