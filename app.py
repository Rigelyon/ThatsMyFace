import streamlit as st
import os
from internal_pages.embed_watermark_page import display_embed_watermark_page
from internal_pages.extract_watermark_page import display_extract_watermark_page
from internal_pages.test_development_page import display_test_development_page

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

if 'operation_options' not in st.session_state:
    st.session_state.operation_options = ["Embed Watermark", "Extract Watermark"]

if 'selected_option' not in st.session_state:
    st.session_state.selected_option = st.session_state.operation_options[0]

process_choice = st.sidebar.radio(
    "Choose operation:",
    st.session_state.operation_options,
    index=st.session_state.operation_options.index(st.session_state.selected_option)
)
st.session_state.selected_option = process_choice

st.sidebar.markdown("---")

def on_toggle_change():
    if st.session_state.debug_mode_toggle and "Test & Development" not in st.session_state.operation_options:
        st.session_state.operation_options.append("Test & Development")
    elif not st.session_state.debug_mode_toggle and "Test & Development" in st.session_state.operation_options:
        if st.session_state.selected_option == "Test & Development":
            st.session_state.selected_option = st.session_state.operation_options[0]
        st.session_state.operation_options.remove("Test & Development")

debug_mode = st.sidebar.toggle("Enable Debug Mode", key="debug_mode_toggle", on_change=on_toggle_change)

if debug_mode:
    st.sidebar.info("Debug mode is enabled. Additional technical information will be displayed.")

# Main application flow
if process_choice == "Embed Watermark":
    display_embed_watermark_page(debug_mode=debug_mode)

elif process_choice == "Extract Watermark":
    display_extract_watermark_page(debug_mode=debug_mode)

elif process_choice == "Test & Development":
    display_test_development_page(debug_mode=debug_mode)

# Footer
st.markdown("---")
st.markdown("**That's My Face** - Secure Image Watermarking with Facial Authentication")