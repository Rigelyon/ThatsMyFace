import streamlit as st
import numpy as np
from PIL import Image
import time

from modules.utils import has_face
from modules.face_recognition import get_face_embedding
from modules.encryption import generate_key_from_embedding, encrypt_watermark, decrypt_watermark


def display_test_development_page(debug_mode=False):
    st.header("Test & Development Page")
    st.warning("This page is only visible in debug mode and is intended for testing and development purposes.")
    
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
            st.image(face1_img, caption="Face 1", use_column_width=True)
            
            # Check if face is detected
            if not has_face(face1_img):
                st.error("No face detected in image 1. Please upload a clear face photo.")
    
    with col2:
        st.markdown("### Face Photo #2 (Decryption)")
        face2_file = st.file_uploader("Upload second face photo", type=["jpg", "jpeg", "png"], key="face2")
        if face2_file:
            face2_img = Image.open(face2_file)
            st.image(face2_img, caption="Face 2", use_column_width=True)
            
            # Check if face is detected
            if not has_face(face2_img):
                st.error("No face detected in image 2. Please upload a clear face photo.")
    
    # Text input for encryption
    st.markdown("### Text to Encrypt")
    plaintext = st.text_area("Enter text to encrypt:", height=100)
    
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
                # Generate encryption keys from embeddings
                key1 = generate_key_from_embedding(embedding1)
                key2 = generate_key_from_embedding(embedding2)
                
                # Debug info
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
                    st.write("Key 2 (first 8 bytes):", key2[:8].hex())
                
                # Encrypt with first face
                start_time = time.time()
                encrypted_data = encrypt_watermark(plaintext, key1)
                encryption_time = time.time() - start_time
                
                # Try to decrypt with second face
                start_time = time.time()
                decrypted_data = decrypt_watermark(encrypted_data, key2)
                decryption_time = time.time() - start_time
                
                # Display results
                st.subheader("Test Results")
                
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
                    st.error("❌ TEST FAILED: Could not decrypt the text with the second face")
                
                # Performance metrics
                if debug_mode:
                    st.subheader("Performance Metrics")
                    st.info(f"Time to extract face embeddings: {embedding_time:.4f} seconds")
                    st.info(f"Time to encrypt: {encryption_time:.4f} seconds")
                    st.info(f"Time to decrypt: {decryption_time:.4f} seconds")
                    st.info(f"Encrypted data size: {len(encrypted_data)} bytes")
                    
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
