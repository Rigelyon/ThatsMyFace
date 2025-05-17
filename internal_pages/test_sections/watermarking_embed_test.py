import io
import time
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from modules.constants import ALPHA
from modules.watermarking import embed_watermark, detect_watermark


def calculate_histogram(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate histograms for each channel of the image
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        Tuple of histograms for each channel (R, G, B)
    """
    # Check if image is grayscale or color
    if len(image_array.shape) == 2 or image_array.shape[2] == 1:
        # Grayscale image
        hist = cv2.calcHist([image_array], [0], None, [256], [0, 256])
        return hist, None, None
    else:
        # Color image - calculate histogram for each channel
        hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256])
        return hist_r, hist_g, hist_b


def plot_histogram_comparison(original_array: np.ndarray, watermarked_array: np.ndarray) -> plt.Figure:
    """
    Create a plot comparing histograms of original and watermarked images
    
    Args:
        original_array: Numpy array of the original image
        watermarked_array: Numpy array of the watermarked image
        
    Returns:
        Matplotlib figure with the histogram comparison
    """
    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Check if image is grayscale or color
    is_grayscale = len(original_array.shape) == 2 or original_array.shape[2] == 1
    
    if is_grayscale:
        # Handle grayscale image
        hist_orig = cv2.calcHist([original_array], [0], None, [256], [0, 256])
        hist_wm = cv2.calcHist([watermarked_array], [0], None, [256], [0, 256])
        
        axs[0].plot(hist_orig, color='black', alpha=0.7, label='Original')
        axs[0].plot(hist_wm, color='blue', alpha=0.7, label='Watermarked')
        axs[0].set_title('Grayscale Histogram')
        axs[0].legend()
        
        # Hide the other two axes
        axs[1].set_visible(False)
        axs[2].set_visible(False)
    else:
        # Handle color image - calculate histogram for each channel
        channels = ['Red', 'Green', 'Blue']
        colors = ['red', 'green', 'blue']
        
        for i, col in enumerate(colors):
            hist_orig = cv2.calcHist([original_array], [i], None, [256], [0, 256])
            hist_wm = cv2.calcHist([watermarked_array], [i], None, [256], [0, 256])
            
            axs[i].plot(hist_orig, color=col, alpha=0.7, label='Original')
            axs[i].plot(hist_wm, color='black', linestyle='--', alpha=0.7, label='Watermarked')
            axs[i].set_title(f'{channels[i]} Channel Histogram')
            axs[i].legend()
    
    # Set common labels
    for ax in axs:
        if ax.get_visible():
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig


def calculate_frequency_domain_stats(image_array: np.ndarray) -> dict:
    """
    Calculate statistics in the frequency domain using DCT
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        Dictionary with frequency domain statistics
    """
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    
    # Apply DCT
    dct = cv2.dct(np.float32(gray))
    
    # Calculate statistics
    stats = {
        'mean': float(np.mean(dct)),
        'std': float(np.std(dct)),
        'min': float(np.min(dct)),
        'max': float(np.max(dct)),
        'energy': float(np.sum(dct**2)),
    }
    
    return stats


def calculate_image_diff(original_array: np.ndarray, watermarked_array: np.ndarray) -> np.ndarray:
    """
    Calculate and visualize the difference between original and watermarked images
    
    Args:
        original_array: Numpy array of the original image
        watermarked_array: Numpy array of the watermarked image
        
    Returns:
        Difference image array (enhanced for visualization)
    """
    if len(original_array.shape) == 3:
        # For color images
        diff = cv2.absdiff(original_array, watermarked_array)
        # Enhance the difference for better visualization
        diff_enhanced = cv2.convertScaleAbs(diff, alpha=5.0)  # Scale up differences
    else:
        # For grayscale images
        diff = cv2.absdiff(original_array, watermarked_array)
        diff_enhanced = cv2.convertScaleAbs(diff, alpha=5.0)
    
    return diff_enhanced


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
        watermark_text = st.text_area("Enter text watermark", value="Omke gams omke gams!", key="embed_test_watermark_text")
        if watermark_text:
            watermark_data = watermark_text.encode('utf-8')
    else:
        watermark_file = st.file_uploader("Upload image watermark", type=["jpg", "jpeg", "png"], key="embed_test_watermark_image")
        if watermark_file:
            watermark_image = Image.open(watermark_file)
            st.image(watermark_image, caption="Watermark Image", use_container_width=True, width=200)

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
            
            # Prepare image arrays for analysis
            original_array = np.array(image).astype(np.float32)
            watermarked_array = np.array(watermarked_img).astype(np.float32)
            
            # Ensure same dimensions for comparison
            min_height = min(original_array.shape[0], watermarked_array.shape[0])
            min_width = min(original_array.shape[1], watermarked_array.shape[1])
            
            # Crop both images to the same dimensions
            original_array = original_array[:min_height, :min_width]
            watermarked_array = watermarked_array[:min_height, :min_width]
            
            # Create tabs for different statistics
            stat_tabs = st.tabs(["Basic Metrics", "Visual Differences", "Histogram Analysis", "Frequency Domain", "Watermark Verification"])
            
            with stat_tabs[0]:
                st.subheader("Basic Image Quality Metrics")
                
                # Calculate MSE (Mean Squared Error)
                mse = np.mean((original_array - watermarked_array) ** 2)
                if mse == 0:  # Images are identical
                    psnr = float('inf')
                else:
                    max_pixel = 255.0
                    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
                
                # Calculate SSIM (Structural Similarity Index)
                if len(original_array.shape) == 3:
                    # Convert to grayscale for SSIM
                    gray_original = cv2.cvtColor(original_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    gray_watermarked = cv2.cvtColor(watermarked_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    ssim_value = ssim(gray_original, gray_watermarked)
                else:
                    ssim_value = ssim(original_array.astype(np.uint8), watermarked_array.astype(np.uint8))
                
                # Display metrics as a table
                metrics_data = {
                    "Metric": ["PSNR (dB)", "MSE", "SSIM"],
                    "Value": [f"{psnr:.2f}", f"{mse:.4f}", f"{ssim_value:.4f}"],
                    "Interpretation": [
                        "Higher is better. Typically good values are 30+ dB",
                        "Lower is better. Values close to 0 indicate high similarity",
                        "Higher is better (0-1). Values close to 1 indicate high structural similarity"
                    ]
                }
                
                st.table(metrics_data)
                
                # Additional basic stats
                st.subheader("Image and Watermark Information")
                info_data = {
                    "Property": [
                        "Image Dimensions", 
                        "Watermark Size", 
                        "Watermark to Image Ratio", 
                        "Alpha (Strength)",
                        "Processing Time"
                    ],
                    "Value": [
                        f"{min_width} x {min_height} pixels",
                        f"{len(watermark_data)} bytes",
                        f"{len(watermark_data) / (min_width * min_height * 3):.6f}",
                        f"{alpha_value}",
                        f"{processing_time:.3f} seconds"
                    ]
                }
                st.table(info_data)
            
            with stat_tabs[1]:
                st.subheader("Visual Differences Analysis")
                
                # Calculate and display difference image
                diff_img = calculate_image_diff(original_array.astype(np.uint8), watermarked_array.astype(np.uint8))
                
                # Calculate percentage of affected pixels (thresholded)
                threshold = 10  # Threshold to consider a pixel "changed"
                affected_pixels = np.sum(diff_img > threshold) / diff_img.size * 100
                
                st.markdown(f"**Percentage of Visibly Affected Pixels:** {affected_pixels:.2f}%")
                st.markdown("**Enhanced Difference Image** (brighter areas indicate larger differences)")
                st.image(diff_img, use_container_width=True)
                
                # Create a heatmap of differences
                st.markdown("**Difference Heatmap by Channel**")
                if len(original_array.shape) == 3:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    channels = ['Red', 'Green', 'Blue']
                    
                    for i in range(3):
                        diff_channel = cv2.absdiff(original_array[:,:,i].astype(np.uint8), 
                                                watermarked_array[:,:,i].astype(np.uint8))
                        im = axs[i].imshow(diff_channel, cmap='hot')
                        axs[i].set_title(f'{channels[i]} Channel Differences')
                        axs[i].axis('off')
                        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    # For grayscale images
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(diff_img, cmap='hot')
                    ax.set_title('Grayscale Differences')
                    ax.axis('off')
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    st.pyplot(fig)
            
            with stat_tabs[2]:
                st.subheader("Histogram Analysis")
                
                # Plot histograms
                hist_fig = plot_histogram_comparison(
                    original_array.astype(np.uint8), 
                    watermarked_array.astype(np.uint8)
                )
                st.pyplot(hist_fig)
                
                # Calculate histogram correlation
                if len(original_array.shape) == 3:
                    # For color images - calculate for each channel
                    correlations = []
                    for i in range(3):
                        hist1 = cv2.calcHist([original_array.astype(np.uint8)], [i], None, [256], [0, 256]).flatten()
                        hist2 = cv2.calcHist([watermarked_array.astype(np.uint8)], [i], None, [256], [0, 256]).flatten()
                        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                        correlations.append(correlation)
                    
                    channels = ['Red', 'Green', 'Blue']
                    for i, corr in enumerate(correlations):
                        st.markdown(f"**{channels[i]} Channel Histogram Correlation:** {corr:.4f}")
                    
                    avg_correlation = sum(correlations) / len(correlations)
                    st.markdown(f"**Average Histogram Correlation:** {avg_correlation:.4f}")
                else:
                    # For grayscale images
                    hist1 = cv2.calcHist([original_array.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
                    hist2 = cv2.calcHist([watermarked_array.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    st.markdown(f"**Grayscale Histogram Correlation:** {correlation:.4f}")
            
            with stat_tabs[3]:
                st.subheader("Frequency Domain Analysis")
                
                # Calculate frequency domain statistics
                original_freq_stats = calculate_frequency_domain_stats(original_array.astype(np.uint8))
                watermarked_freq_stats = calculate_frequency_domain_stats(watermarked_array.astype(np.uint8))
                
                # Create a comparison table
                freq_comparison = {
                    "Metric": list(original_freq_stats.keys()),
                    "Original Image": [f"{original_freq_stats[k]:.2f}" for k in original_freq_stats],
                    "Watermarked Image": [f"{watermarked_freq_stats[k]:.2f}" for k in watermarked_freq_stats],
                    "% Change": [
                        f"{(watermarked_freq_stats[k] - original_freq_stats[k]) / original_freq_stats[k] * 100:.2f}%" 
                        for k in original_freq_stats
                    ]
                }
                
                st.table(freq_comparison)
                
                # Display DCT coefficient visualization
                st.markdown("**DCT Coefficient Visualization**")
                
                # Get grayscale images for DCT
                if len(original_array.shape) == 3:
                    gray_original = cv2.cvtColor(original_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    gray_watermarked = cv2.cvtColor(watermarked_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray_original = original_array.astype(np.uint8)
                    gray_watermarked = watermarked_array.astype(np.uint8)
                
                # Apply DCT
                dct_original = cv2.dct(np.float32(gray_original))
                dct_watermarked = cv2.dct(np.float32(gray_watermarked))
                
                # Visualize log-scaled DCT coefficients
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                
                # Apply log transformation for better visualization
                dct_log_original = np.log(np.abs(dct_original) + 1)
                dct_log_watermarked = np.log(np.abs(dct_watermarked) + 1)
                
                im1 = axs[0].imshow(dct_log_original, cmap='viridis')
                axs[0].set_title('Original Image DCT')
                axs[0].axis('off')
                
                im2 = axs[1].imshow(dct_log_watermarked, cmap='viridis')
                axs[1].set_title('Watermarked Image DCT')
                axs[1].axis('off')
                
                plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
                plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with stat_tabs[4]:
                st.subheader("Watermark Verification")
                
                # Verify if the watermark is detectable
                watermark_detection_result = detect_watermark(watermarked_img)
                
                if watermark_detection_result:
                    st.success("✅ Watermark detected successfully in the image!")
                    watermark_probability = "High"
                else:
                    st.warning("⚠️ Watermark not clearly detected in the image.")
                    watermark_probability = "Low"
                
                # Display watermark quality assessment
                st.markdown("### Watermark Quality Assessment")
                
                # Create a scoring system based on the metrics
                
                # PSNR score: 0-10
                psnr_score = min(10, max(0, psnr / 5))
                
                # SSIM score: 0-10
                ssim_score = min(10, max(0, ssim_value * 10))
                
                # Visibility score (based on affected pixels): 0-10
                # Lower affected pixels is better for invisibility
                visibility_score = min(10, max(0, 10 - (affected_pixels / 2)))
                
                # Watermark detection score: 0-10
                detection_score = 10 if watermark_detection_result else 0
                
                # Total score
                quality_scores = {
                    "Aspect": ["Imperceptibility (PSNR)", "Structural Preservation (SSIM)", 
                              "Visual Imperceptibility", "Detectability", "Overall Score"],
                    "Score": [
                        f"{psnr_score:.1f}/10",
                        f"{ssim_score:.1f}/10",
                        f"{visibility_score:.1f}/10",
                        f"{detection_score}/10",
                        f"{(psnr_score + ssim_score + visibility_score + detection_score) / 4:.1f}/10"
                    ],
                    "Interpretation": [
                        "Higher is better" if psnr_score > 7 else "Needs improvement",
                        "Higher is better" if ssim_score > 7 else "Needs improvement",
                        "Higher is better" if visibility_score > 7 else "Watermark may be visible",
                        "Successful detection" if detection_score == 10 else "Detection failed",
                        "Excellent" if (psnr_score + ssim_score + visibility_score + detection_score) / 4 > 8 else 
                        "Good" if (psnr_score + ssim_score + visibility_score + detection_score) / 4 > 6 else 
                        "Needs improvement"
                    ]
                }
                
                st.table(quality_scores)
                
                # Overall conclusion
                overall_score = (psnr_score + ssim_score + visibility_score + detection_score) / 4
                
                st.subheader("Watermark Embedding Result")
                
                if overall_score > 8:
                    st.success("""
                    **Excellent watermarking result!**
                    
                    The watermark has been embedded with high quality, maintaining image fidelity while ensuring detectability.
                    """)
                elif overall_score > 6:
                    st.info("""
                    **Good watermarking result.**
                    
                    The watermark has been embedded successfully with acceptable quality.
                    Consider adjusting Alpha value for optimal balance between visibility and robustness.
                    """)
                else:
                    st.warning("""
                    **Watermarking needs improvement.**
                    
                    The current watermarking parameters may not be optimal. Consider:
                    - Adjusting Alpha value (lower for less visibility, higher for better detection)
                    - Using a smaller watermark size
                    - Trying a different image with more texture to hide the watermark
                    """)
            
            # Download watermarked image
            buffered = io.BytesIO()
            watermarked_img.save(buffered, format="PNG")
            st.download_button(
                label="Download Watermarked Image",
                data=buffered.getvalue(),
                file_name=f"watermarked_{image_file.name}",
                mime="image/png"
            )