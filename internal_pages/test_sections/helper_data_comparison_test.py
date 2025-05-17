import time

import numpy as np
import streamlit as st
from PIL import Image

from modules.constants import MIN_SIMILARITY_THRESHOLD, SIMILARITY_THRESHOLD, MAX_SIMILARITY_THRESHOLD
from modules.face_recognition import get_face_embedding, calculate_similarity
from modules.fuzzy_extractor import generate_key_with_helper, regenerate_key_from_helper
from modules.utils import has_face, serialize_helper_data


def display_helper_data_comparison_test():
    st.subheader("Helper Data Similarity Test")
    st.markdown("""
    This test allows you to compare helper data generated from two different faces.
    
    1. Upload two face photos (can be the same person or different people)
    2. Adjust the error tolerance to control matching sensitivity
    3. The system will generate and compare helper data from both faces
    4. Security and reliability conclusions will be displayed based on the results
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Face Photo #1")
        face1_file = st.file_uploader("Upload first face photo", type=["jpg", "jpeg", "png"], key="helper_face1")
        if face1_file:
            face1_img = Image.open(face1_file)
            st.image(face1_img, caption="Face 1", use_container_width=True)

            # Check if face is detected
            if not has_face(face1_img):
                st.error("No face detected in image 1. Please upload a clear face photo.")

    with col2:
        st.markdown("### Face Photo #2")
        face2_file = st.file_uploader("Upload second face photo", type=["jpg", "jpeg", "png"], key="helper_face2")
        if face2_file:
            face2_img = Image.open(face2_file)
            st.image(face2_img, caption="Face 2", use_container_width=True)

            # Check if face is detected
            if not has_face(face2_img):
                st.error("No face detected in image 2. Please upload a clear face photo.")

    # Error tolerance slider
    st.markdown("""
    ### Error Tolerance
    Adjust this slider to control the balance between security (lower values) and flexibility (higher values).
    - **Low values** (1-30): Stricter matching - Requires very similar face images but more secure.
    - **Medium values** (31-70): Balanced - Good for most use cases.
    - **High values** (71-95): Flexible matching - Works with more varied face images but potentially less secure.
    """)

    error_tolerance = st.slider(
        "Error Tolerance",
        min_value=1,
        max_value=95,
        value=60,
        key="helper_tolerance",
        help="Higher values allow more variation in face images but may reduce security"
    )

    # Run test button
    if st.button("Run Comparison Test", disabled=not (face1_file and face2_file)):
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
                # Calculate face similarity
                similarity = calculate_similarity(embedding1, embedding2)

                # Show face similarity assessment with color coding
                st.subheader("Face Similarity Assessment")
                if similarity < MIN_SIMILARITY_THRESHOLD:
                    st.error(f"⚠️ Face Similarity: {similarity:.4f} - These appear to be different people")
                elif similarity < SIMILARITY_THRESHOLD:
                    st.warning(f"⚠️ Face Similarity: {similarity:.4f} - These faces have low similarity")
                elif similarity < MAX_SIMILARITY_THRESHOLD:
                    st.success(f"✅ Face Similarity: {similarity:.4f} - Good similarity between faces")
                else:
                    st.success(f"✅ Face Similarity: {similarity:.4f} - Excellent match between faces")

                # Normalize the error_tolerance from user slider (1-95) to (0.01-0.95)
                user_tolerance = error_tolerance / 100.0

                # Generate helper data for both faces
                start_time = time.time()
                key1, helper_data1 = generate_key_with_helper(embedding1, user_tolerance)
                key2, helper_data2 = generate_key_with_helper(embedding2, user_tolerance)
                key_generation_time = time.time() - start_time

                # Compare the keys
                key_match_count = sum(a == b for a, b in zip(key1, key2))
                key_match_percentage = (key_match_count / len(key1)) * 100

                # Compare helper data structure
                helper_data_differences = {}
                common_keys = set(helper_data1.keys()).intersection(set(helper_data2.keys()))

                for key in common_keys:
                    if isinstance(helper_data1[key], (int, float)) and isinstance(helper_data2[key], (int, float)):
                        value1 = helper_data1[key]
                        value2 = helper_data2[key]
                        if key == "error_tolerance":
                            helper_data_differences[key] = "Same" if value1 == value2 else f"{value1} vs {value2}"
                        elif key in ["vector_mean", "vector_std"]:
                            diff = abs(value1 - value2)
                            relative_diff = diff / max(abs(value1), abs(value2), 1e-10) * 100
                            helper_data_differences[key] = f"{relative_diff:.2f}% difference"
                    elif key == "vector_shape":
                        helper_data_differences[key] = "Same" if helper_data1[key] == helper_data2[key] else "Different"

                # Cross-regeneration test (try to regenerate key1 using embedding2 with helper_data1)
                can_cross_regenerate1 = regenerate_key_from_helper(embedding2, helper_data1) == key1
                can_cross_regenerate2 = regenerate_key_from_helper(embedding1, helper_data2) == key2

                # Display results
                st.subheader("Comparison Results")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Helper Data Face #1")
                    st.info(f"Size: {len(serialize_helper_data(helper_data1))} bytes")
                    st.json({
                        'error_tolerance': helper_data1.get('error_tolerance'),
                        'vector_shape': helper_data1.get('vector_shape'),
                        'vector_mean': helper_data1.get('vector_mean'),
                        'vector_std': helper_data1.get('vector_std')
                    })

                    # Download helper data 1
                    helper_data1_bytes = serialize_helper_data(helper_data1)
                    st.download_button(
                        label="⬇️ Download Helper Data #1",
                        data=helper_data1_bytes,
                        file_name="helper_data_face1.bin",
                        mime="application/octet-stream"
                    )

                with col2:
                    st.markdown("### Helper Data Face #2")
                    st.info(f"Size: {len(serialize_helper_data(helper_data2))} bytes")
                    st.json({
                        'error_tolerance': helper_data2.get('error_tolerance'),
                        'vector_shape': helper_data2.get('vector_shape'),
                        'vector_mean': helper_data2.get('vector_mean'),
                        'vector_std': helper_data2.get('vector_std')
                    })

                    # Download helper data 2
                    helper_data2_bytes = serialize_helper_data(helper_data2)
                    st.download_button(
                        label="⬇️ Download Helper Data #2",
                        data=helper_data2_bytes,
                        file_name="helper_data_face2.bin",
                        mime="application/octet-stream"
                    )

                # Comparison results
                st.subheader("Comparison Analysis")

                # Face similarity
                st.markdown(f"**Face Embedding Similarity:** {similarity:.4f}")

                # Key similarity
                st.markdown(
                    f"**Key Similarity:** {key_match_percentage:.2f}% ({key_match_count}/{len(key1)} bytes match)")

                # Helper data differences
                st.markdown("**Helper Data Differences:**")
                for key, value in helper_data_differences.items():
                    st.markdown(f"- {key}: {value}")

                # Cross-regeneration results
                st.markdown("**Cross-Regeneration Test:**")
                if can_cross_regenerate1:
                    st.success("✅ Key #1 can be regenerated using face #2")
                else:
                    st.error("❌ Key #1 cannot be regenerated using face #2")

                if can_cross_regenerate2:
                    st.success("✅ Key #2 can be regenerated using face #1")
                else:
                    st.error("❌ Key #2 cannot be regenerated using face #1")

                # Security analysis
                st.subheader("Security Analysis")

                if similarity > SIMILARITY_THRESHOLD and key_match_percentage > 90:
                    st.warning(
                        "⚠️ **High Security Risk**: Helper data generated from both faces are very similar, which can compromise security. This indicates that both faces can be 'backup keys' for each other.")
                elif similarity > MIN_SIMILARITY_THRESHOLD and key_match_percentage > 70:
                    st.warning(
                        "⚠️ **Medium Security Risk**: There is significant similarity between helper data from both faces. For better security, make sure to use more different faces.")
                else:
                    st.success(
                        "✅ **Good Security**: Helper data from both faces are sufficiently different, indicating a good level of security. This is the expected result when comparing different faces.")

                # Performance metrics
                st.subheader("Performance Metrics")
                st.info(f"Face embedding extraction time: {embedding_time:.4f} seconds")
                st.info(f"Key generation with helper data time: {key_generation_time:.4f} seconds")
