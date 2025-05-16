import streamlit as st

from internal_pages.test_sections.face_encrypt_test import display_face_encrypt_test


def display_test_development_page(debug_mode=False):
    st.header("Test & Development Page")
    st.warning("This page is only visible in debug mode and is intended for testing and development purposes.")
    
    # Display the Face Recognition Encryption Test
    display_face_encrypt_test()
