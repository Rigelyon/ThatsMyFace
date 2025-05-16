import time

import numpy as np
import streamlit as st
from PIL import Image

from modules.encryption import encrypt_watermark, decrypt_watermark
from modules.face_recognition import get_face_embedding
from modules.fuzzy_extractor import generate_key_with_helper, regenerate_key_from_helper
from modules.utils import has_face, serialize_embedding, serialize_helper_data


def display_face_encrypt_test(debug_mode=False):
    st.subheader("Face Recognition Encryption Test")
    st.markdown("""
    This test verifies if the system can use a face embedding from one image to decrypt data that was 
    encrypted with another image of the same face.
    
    1. Upload two different photos with the same face
    2. Enter text to encrypt
    3. The system will encrypt the text using the first photo and attempt to decrypt it using the second photo
    4. If successful, this confirms the face recognition system can identify the same person across different photos
    """)
    
    # Create two columns for face uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Face Photo #1 (Encryption)")
        face1_file = st.file_uploader("Upload first face photo", type=["jpg", "jpeg", "png"], key="face1")
        if face1_file:
            face1_img = Image.open(face1_file)
            st.image(face1_img, caption="Face 1", use_container_width=True)
            
            # Check if face is detected
            if not has_face(face1_img):
                st.error("No face detected in image 1. Please upload a clear face photo.")
    
    with col2:
        st.markdown("### Face Photo #2 (Decryption)")
        face2_file = st.file_uploader("Upload second face photo", type=["jpg", "jpeg", "png"], key="face2")
        if face2_file:
            face2_img = Image.open(face2_file)
            st.image(face2_img, caption="Face 2", use_container_width=True)
            
            # Check if face is detected
            if not has_face(face2_img):
                st.error("No face detected in image 2. Please upload a clear face photo.")
    
    # Text input for encryption
    st.markdown("### Text to Encrypt")
    plaintext = st.text_area("Enter text to encrypt:", height=100)
    
    # Error tolerance slider for fuzzy extractor
    error_tolerance = st.slider(
        "Error Tolerance (higher values allow more variation in face images)",
        min_value=1,
        max_value=95,
        value=50,
        help="Higher values make authentication more flexible but potentially less secure"
    )
    
    # Run test button
    if st.button("Run Encryption Test", disabled=not(face1_file and face2_file and plaintext)):
        with st.spinner("Running test..."):
            # Process images
            face1_img = Image.open(face1_file)
            face2_img = Image.open(face2_file)
            
            # Convert to numpy arrays
            face1_array = np.array(face1_img)
            face2_array = np.array(face2_img)
            
            # Get face embeddings
            start_time = time.time()
            embedding1 = get_face_embedding(face1_array)
            embedding2 = get_face_embedding(face2_array)
            embedding_time = time.time() - start_time
            
            if embedding1 is None or embedding2 is None:
                st.error("Failed to extract face embeddings from one or both images")
            else:
                # Add download buttons for both embeddings
                col1, col2 = st.columns(2)
                with col1:
                    embedding1_bytes = serialize_embedding(embedding1)
                    st.download_button(
                        label="⬇️ Download Face 1 Embedding",
                        data=embedding1_bytes,
                        file_name="face1_embedding.npy",
                        mime="application/octet-stream",
                        help="Download the face embedding vector for the first face image"
                    )
                
                with col2:
                    embedding2_bytes = serialize_embedding(embedding2)
                    st.download_button(
                        label="⬇️ Download Face 2 Embedding",
                        data=embedding2_bytes,
                        file_name="face2_embedding.npy",
                        mime="application/octet-stream",
                        help="Download the face embedding vector for the second face image"
                    )
                
                # Generate encryption key and helper data from the first embedding
                # Convert error_tolerance from percentage (1-95) to fraction (0.01-0.95)
                error_tolerance_fraction = error_tolerance / 100.0
                start_time = time.time()
                key1, helper_data = generate_key_with_helper(embedding1, error_tolerance_fraction)
                key_generation_time = time.time() - start_time
                
                # Debug info for embeddings
                if debug_mode:
                    st.write("Face 1 embedding stats:", {
                        'mean': float(np.mean(embedding1)),
                        'std': float(np.std(embedding1)),
                        'min': float(np.min(embedding1)),
                        'max': float(np.max(embedding1)),
                        'shape': embedding1.shape
                    })
                    st.write("Face 2 embedding stats:", {
                        'mean': float(np.mean(embedding2)),
                        'std': float(np.std(embedding2)),
                        'min': float(np.min(embedding2)),
                        'max': float(np.max(embedding2)),
                        'shape': embedding2.shape
                    })
                    st.write("Key 1 (first 8 bytes):", key1[:8].hex())
                    
                    # Show helper data info
                    st.write("Helper data info:", {
                        'error_tolerance': helper_data.get('error_tolerance'),
                        'vector_shape': helper_data.get('vector_shape'),
                        'vector_mean': helper_data.get('vector_mean'),
                        'vector_std': helper_data.get('vector_std')
                    })
                
                # Encrypt with first face
                start_time = time.time()
                encrypted_data = encrypt_watermark(plaintext, key1)
                encryption_time = time.time() - start_time
                
                # Regenerate key from second face using helper data
                start_time = time.time()
                key2 = regenerate_key_from_helper(embedding2, helper_data)
                key_regeneration_time = time.time() - start_time
                
                if debug_mode and key2 is not None:
                    st.write("Regenerated Key 2 (first 8 bytes):", key2[:8].hex())
                
                # Try to decrypt with second face
                start_time = time.time()
                decrypted_data = decrypt_watermark(encrypted_data, key2) if key2 is not None else None
                decryption_time = time.time() - start_time
                
                # Display results
                st.subheader("Test Results")
                
                # Helper data download
                helper_data_bytes = serialize_helper_data(helper_data)
                st.download_button(
                    label="⬇️ Download Helper Data",
                    data=helper_data_bytes,
                    file_name=f"helper_data_test.bin",
                    mime="application/octet-stream",
                    help="This file contains the helper data needed for authentication with fuzzy extractor"
                )
                
                st.info("The helper data file is required for decryption with any face image, even if it's the same person. It contains error correction data but not the secret key itself.")
                
                st.info("You can download the face embeddings for research or development purposes. These are numerical vector representations of the face features that can be used with compatible machine learning models.")
                
                # Check if decryption was successful
                if decrypted_data is not None:
                    try:
                        decrypted_text = decrypted_data.decode('utf-8')
                        
                        # Check if original text matches decrypted text
                        if decrypted_text == plaintext:
                            st.success("✅ TEST PASSED: Successfully decrypted the text with a different image of the same face!")
                            st.markdown(f"**Original text:** {plaintext}")
                            st.markdown(f"**Decrypted text:** {decrypted_text}")
                        else:
                            st.error("❌ TEST FAILED: Decryption succeeded but the decrypted text doesn't match the original!")
                            st.markdown(f"**Original text:** {plaintext}")
                            st.markdown(f"**Decrypted text:** {decrypted_text}")
                    except UnicodeDecodeError:
                        st.error("❌ TEST FAILED: Data was decrypted but couldn't be converted to text")
                else:
                    if key2 is None:
                        st.error("❌ TEST FAILED: Could not regenerate the key from the second face")
                        st.info("This usually means the face images are too different, or the error tolerance is too low. Try increasing the error tolerance or using more similar images.")
                    else:
                        st.error("❌ TEST FAILED: Could not decrypt the text with the second face")
                
                # Performance metrics
                if debug_mode:
                    st.subheader("Performance Metrics")
                    st.info(f"Time to extract face embeddings: {embedding_time:.4f} seconds")
                    st.info(f"Time to generate key with helper data: {key_generation_time:.4f} seconds")
                    st.info(f"Time to encrypt: {encryption_time:.4f} seconds")
                    st.info(f"Time to regenerate key from helper data: {key_regeneration_time:.4f} seconds")
                    st.info(f"Time to decrypt: {decryption_time:.4f} seconds")
                    st.info(f"Encrypted data size: {len(encrypted_data)} bytes")
                    st.info(f"Helper data size: {len(serialize_helper_data(helper_data))} bytes")
                    
                # Show key similarity if debug mode is enabled
                if debug_mode:
                    # Calculate cosine similarity between embeddings
                    dot_product = np.dot(embedding1, embedding2)
                    norm1 = np.linalg.norm(embedding1)
                    norm2 = np.linalg.norm(embedding2)
                    similarity = dot_product / (norm1 * norm2)
                    
                    st.info(f"Face embedding similarity: {similarity:.4f} (higher is better, above 0.7 typically indicates same person)")
                    
                    # Compare keys
                    key_match_count = sum(a == b for a, b in zip(key1, key2))
                    key_match_percentage = (key_match_count / len(key1)) * 100
                    st.info(f"Key similarity: {key_match_percentage:.2f}% ({key_match_count}/{len(key1)} bytes match)")
