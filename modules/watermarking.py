import numpy as np
from PIL import Image
import io
from scipy.fftpack import dct, idct

from modules.constants import BLOCK_SIZE, ALPHA


def rgb_to_ycbcr(img):
    """Konversi gambar RGB ke YCbCr"""
    # Pastikan gambar dalam format RGB (3 channel)
    if isinstance(img, Image.Image):
        img = img.convert("RGB")

    img_array = np.array(img, dtype=np.float32) / 255.0

    # Periksa dan tangani gambar RGBA
    if img_array.shape[-1] == 4:
        # Ambil hanya 3 channel pertama (RGB) dan abaikan alpha
        img_array = img_array[:, :, :3]

    # Matriks transformasi RGB ke YCbCr
    transform = np.array(
        [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]]
    )

    # Konversi setiap pixel
    ycbcr = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            ycbcr[i, j, :] = np.dot(transform, img_array[i, j, :])

    # Offset komponen Cb dan Cr
    ycbcr[:, :, 1:] += 0.5

    return ycbcr


def ycbcr_to_rgb(img):
    """Konversi gambar YCbCr ke RGB"""
    img_array = img.copy()

    # Offset komponen Cb dan Cr
    img_array[:, :, 1:] -= 0.5

    # Matriks transformasi YCbCr ke RGB
    transform = np.array([[1.0, 0.0, 1.403], [1.0, -0.344, -0.714], [1.0, 1.773, 0.0]])

    # Konversi setiap pixel
    rgb = np.zeros_like(img_array)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            rgb[i, j, :] = np.dot(transform, img_array[i, j, :])

    # Clip nilai ke range [0, 1]
    rgb = np.clip(rgb, 0, 1)

    # Konversi kembali ke nilai 0-255
    return (rgb * 255).astype(np.uint8)


def apply_dct_to_block(block):
    """Terapkan DCT 2D pada blok"""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def apply_idct_to_block(block):
    """Terapkan Inverse DCT 2D pada blok"""
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def resize_watermark(watermark, target_height, target_width):
    """Ubah ukuran watermark agar sesuai dengan jumlah blok dalam gambar"""
    watermark_img = (
        Image.open(io.BytesIO(watermark)) if isinstance(watermark, bytes) else watermark
    )

    # Konversi ke grayscale
    watermark_img = watermark_img.convert("L")

    # Skalakan sesuai dengan target
    watermark_resized = watermark_img.resize(
        (target_width, target_height), Image.LANCZOS
    )

    return watermark_resized


def embed_watermark(image, watermark_data):
    """
    Sisipkan watermark ke dalam gambar

    Parameters:
    image (PIL.Image.Image): Gambar asli dalam format RGB
    watermark_data (bytes atau PIL.Image.Image): Data watermark

    Returns:
    PIL.Image.Image: Gambar hasil watermarking
    """
    # Pastikan gambar dalam format RGB
    image = image.convert("RGB")

    # Konversi watermark_data ke Image jika berbentuk bytes
    if isinstance(watermark_data, bytes):
        watermark_img = Image.open(io.BytesIO(watermark_data))
    else:
        watermark_img = watermark_data

    # Konversi gambar asli ke array numpy
    img_array = np.array(image)

    # Konversi gambar asli ke YCbCr
    ycbcr = rgb_to_ycbcr(image)

    # Ambil channel Y (luminance)
    Y = ycbcr[:, :, 0]

    # Hitung jumlah blok dalam gambar
    height, width = Y.shape
    num_blocks_h = height // BLOCK_SIZE
    num_blocks_w = width // BLOCK_SIZE

    # Ukuran watermark harus sesuai dengan jumlah blok dalam gambar
    watermark_resized = resize_watermark(watermark_img, num_blocks_h, num_blocks_w)
    watermark_array = np.array(watermark_resized) / 255.0  # Normalisasi ke [0, 1]

    # Iterasi melalui setiap blok 8x8
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Ekstrak blok 8x8 dari channel Y
            block = Y[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]

            # Terapkan DCT ke blok
            dct_block = apply_dct_to_block(block)

            # Terapkan SVD ke hasil DCT
            U, S, Vt = np.linalg.svd(dct_block, full_matrices=True)

            # Modifikasi nilai singular dengan nilai watermark
            S[0] += ALPHA * watermark_array[i, j]

            # Rekonstruksi blok dengan inverse SVD
            modified_dct_block = np.dot(U, np.dot(np.diag(S), Vt))

            # Terapkan inverse DCT
            modified_block = apply_idct_to_block(modified_dct_block)

            # Ganti blok asli dengan blok yang telah dimodifikasi
            Y[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ] = modified_block

    # Ganti channel Y asli dengan Y yang telah dimodifikasi
    ycbcr[:, :, 0] = Y

    # Konversi kembali ke RGB
    rgb_array = ycbcr_to_rgb(ycbcr)

    # Buat gambar PIL dari array
    watermarked_img = Image.fromarray(rgb_array)

    return watermarked_img


def extract_watermark(watermarked_image, original_image):
    """
    Ekstrak watermark dari gambar yang sudah disisipkan watermark

    Parameters:
    watermarked_image (PIL.Image.Image): Gambar yang disisipi watermark
    original_image (PIL.Image.Image): Gambar asli sebelum disisipi watermark

    Returns:
    PIL.Image.Image: Watermark hasil ekstraksi
    """
    # Pastikan kedua gambar dalam format RGB
    watermarked_image = watermarked_image.convert("RGB")
    original_image = original_image.convert("RGB")

    # Konversi gambar ke YCbCr
    ycbcr_watermarked = rgb_to_ycbcr(watermarked_image)
    ycbcr_original = rgb_to_ycbcr(original_image)

    # Ambil channel Y (luminance)
    Y_watermarked = ycbcr_watermarked[:, :, 0]
    Y_original = ycbcr_original[:, :, 0]

    # Hitung jumlah blok dalam gambar
    height, width = Y_watermarked.shape
    num_blocks_h = height // BLOCK_SIZE
    num_blocks_w = width // BLOCK_SIZE

    # Buat array untuk menyimpan hasil ekstraksi watermark
    extracted_watermark = np.zeros((num_blocks_h, num_blocks_w))

    # Iterasi melalui setiap blok 8x8
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Ekstrak blok 8x8 dari kedua gambar
            block_watermarked = Y_watermarked[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]
            block_original = Y_original[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ]

            # Terapkan DCT ke blok
            dct_block_watermarked = apply_dct_to_block(block_watermarked)
            dct_block_original = apply_dct_to_block(block_original)

            # Terapkan SVD ke hasil DCT
            _, S_watermarked, _ = np.linalg.svd(
                dct_block_watermarked, full_matrices=True
            )
            _, S_original, _ = np.linalg.svd(dct_block_original, full_matrices=True)

            # Ekstrak watermark dengan menghitung selisih singular value yang dimodifikasi
            extracted_watermark[i, j] = (S_watermarked[0] - S_original[0]) / ALPHA

    # Normalisasi hasil ke range [0, 255]
    extracted_watermark = np.clip(extracted_watermark, 0, 1)
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)

    # Buat gambar PIL dari array
    extracted_img = Image.fromarray(extracted_watermark)

    return extracted_img
