import os

import streamlit as st

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
st.markdown(
    """
This application allows you to watermark images using DCT and SVD techniques.
The watermark is encrypted using AES, with the encryption key derived from facial recognition embeddings.
"""
)

# Create necessary directories
if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("output"):
    os.makedirs("output")

# Sidebar for options
with st.sidebar:
    st.title("🧩 Application Menu")
    st.markdown("---")

    # Initialize state for menu options
    if "operation_options" not in st.session_state:
        st.session_state.operation_options = ["Embed Watermark", "Extract Watermark"]

    if "selected_option" not in st.session_state:
        st.session_state.selected_option = st.session_state.operation_options[0]

    # Dictionary to map options with icons
    menu_icons = {
        "Embed Watermark": "💧 Embed Watermark",
        "Extract Watermark": "🔍 Extract Watermark",
        "Test & Development": "🧪 Test & Development",
    }

    st.subheader("📋 Select Operation:")

    # Using buttons for menu navigation
    col1, col2 = st.columns(2)
    process_choice = st.session_state.selected_option

    for i, option in enumerate(st.session_state.operation_options):
        display_name = menu_icons[option]
        is_selected = option == st.session_state.selected_option

        # Create buttons with different colors for active state
        if st.button(
            display_name,
            key=f"btn_{option}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_option = option
            process_choice = option
            st.rerun()

    st.markdown("---")

    # Debug mode with clearer information
    st.subheader("⚙️ Settings")

    # Inisialisasi debug_mode dalam session_state jika belum ada
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False

    def on_toggle_change():
        # Update status debug mode
        st.session_state.debug_mode = st.session_state.debug_mode_toggle

        # Update daftar operation_options berdasarkan status debug
        test_dev_option = "Test & Development"
        if st.session_state.debug_mode:
            if test_dev_option not in st.session_state.operation_options:
                st.session_state.operation_options.append(test_dev_option)
        else:
            if test_dev_option in st.session_state.operation_options:
                # Jika opsi yang sedang dipilih adalah Test & Development, ubah ke opsi pertama
                if st.session_state.selected_option == test_dev_option:
                    st.session_state.selected_option = (
                        st.session_state.operation_options[0]
                    )
                # Hapus opsi Test & Development
                st.session_state.operation_options.remove(test_dev_option)

    # Set nilai toggle berdasarkan nilai debug_mode yang tersimpan
    debug_mode = st.toggle(
        "Enable Debug Mode",
        key="debug_mode_toggle",
        value=st.session_state.debug_mode,
        on_change=on_toggle_change,
    )

    if debug_mode:
        st.info(
            "Debug mode active. Additional technical information will be displayed."
        )

    # Additional information in sidebar
    st.markdown("---")
    st.markdown("### 📝 About Application")
    st.markdown(
        """
    **That's My Face** is an application for:
    - Embedding watermarks in images
    - Securing with facial authentication
    - Safe watermark extraction
    """
    )

    # Footer sidebar
    st.markdown("---")
    st.caption("© 2025 That's My Face")

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
