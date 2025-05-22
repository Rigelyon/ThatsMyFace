import streamlit as st

from internal_pages.test_sections.face_encrypt_decrypt_test import (
    display_face_encrypt_test,
)
from internal_pages.test_sections.helper_data_comparison_test import (
    display_helper_data_comparison_test,
)
from internal_pages.test_sections.watermarking_embed_test import (
    display_watermark_embed_test,
)
from internal_pages.test_sections.watermarking_extract_test import (
    display_watermark_extract_test,
)


def display_test_development_page(debug_mode=False):
    st.header("Test & Development Page")
    st.warning(
        "This page is only visible in debug mode and is intended for testing and development purposes."
    )

    test_tabs = st.tabs(
        [
            "Face Encrypt and Decrypt",
            "Helper Data Comparison",
            "Watermark Embedding Test",
            "Watermark Extracting Test",
        ]
    )

    with test_tabs[0]:
        display_face_encrypt_test()

    with test_tabs[1]:
        display_helper_data_comparison_test()

    with test_tabs[2]:
        display_watermark_embed_test()

    with test_tabs[3]:
        display_watermark_extract_test()
