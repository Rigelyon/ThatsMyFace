import numpy as np
from PIL import Image
import io
from scipy.fftpack import dct, idct
import streamlit as st

from modules.constants import BLOCK_SIZE, ALPHA


def get_watermark_settings():
    """
    Mengambil pengaturan watermark dari session state atau menggunakan nilai default
    """
    if "custom_settings" in st.session_state:
        settings = st.session_state.custom_settings
        return {
            "block_size": settings.get("block_size", BLOCK_SIZE),
            "alpha": settings.get("alpha", ALPHA),
        }
    return {"block_size": BLOCK_SIZE, "alpha": ALPHA}


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


def resize_watermark(watermark, target_height, target_width, preserve_ratio=False):
    """Ubah ukuran watermark agar sesuai dengan jumlah blok dalam gambar

    Parameters:
    watermark (bytes atau PIL.Image): Data watermark
    target_height (int): Tinggi target
    target_width (int): Lebar target
    preserve_ratio (bool): Jika True, pertahankan aspect ratio dan tambahkan padding putih

    Returns:
    PIL.Image: Watermark yang telah diubah ukurannya
    """
    watermark_img = (
        Image.open(io.BytesIO(watermark)) if isinstance(watermark, bytes) else watermark
    )

    # Konversi ke grayscale
    watermark_img = watermark_img.convert("L")

    if preserve_ratio:
        # Hitung ukuran baru dengan mempertahankan aspect ratio
        original_width, original_height = watermark_img.size
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        # Ubah ukuran watermark dengan mempertahankan aspect ratio
        resized_watermark = watermark_img.resize((new_width, new_height), Image.LANCZOS)

        # Buat gambar putih dengan ukuran target
        final_watermark = Image.new("L", (target_width, target_height), 255)

        # Hitung posisi untuk meletakkan watermark di tengah
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2

        # Tempelkan watermark yang diubah ukurannya ke gambar putih
        final_watermark.paste(resized_watermark, (paste_x, paste_y))

        return final_watermark
    else:
        # Metode lama: skalakan langsung ke ukuran target
        watermark_resized = watermark_img.resize(
            (target_width, target_height), Image.LANCZOS
        )

        return watermark_resized


def embed_watermark(image, watermark_data, preserve_ratio=False, custom_settings=None):
    """
    Sisipkan watermark ke dalam gambar

    Parameters:
    image (PIL.Image.Image): Gambar asli dalam format RGB
    watermark_data (bytes atau PIL.Image.Image): Data watermark
    preserve_ratio (bool): Jika True, watermark akan mempertahankan aspect ratio
    custom_settings (dict): Pengaturan kustom untuk override nilai default

    Returns:
    PIL.Image.Image: Gambar hasil watermarking
    """
    # Dapatkan pengaturan yang akan digunakan
    settings = custom_settings if custom_settings else get_watermark_settings()
    block_size = settings.get("block_size", BLOCK_SIZE)
    alpha = settings.get("alpha", ALPHA)

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
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size

    # Ukuran watermark harus sesuai dengan jumlah blok dalam gambar
    watermark_resized = resize_watermark(
        watermark_img, num_blocks_h, num_blocks_w, preserve_ratio
    )
    watermark_array = np.array(watermark_resized) / 255.0  # Normalisasi ke [0, 1]

    # Iterasi melalui setiap blok
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Ekstrak blok dari channel Y
            block = Y[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]

            # Terapkan DCT ke blok
            dct_block = apply_dct_to_block(block)

            # Terapkan SVD ke hasil DCT
            U, S, Vt = np.linalg.svd(dct_block, full_matrices=True)

            # Modifikasi nilai singular dengan nilai watermark
            S[0] += alpha * watermark_array[i, j]

            # Rekonstruksi blok dengan inverse SVD
            modified_dct_block = np.dot(U, np.dot(np.diag(S), Vt))

            # Terapkan inverse DCT
            modified_block = apply_idct_to_block(modified_dct_block)

            # Ganti blok asli dengan blok yang telah dimodifikasi
            Y[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ] = modified_block

    # Ganti channel Y asli dengan Y yang telah dimodifikasi
    ycbcr[:, :, 0] = Y

    # Konversi kembali ke RGB
    rgb_array = ycbcr_to_rgb(ycbcr)

    # Buat gambar PIL dari array
    watermarked_img = Image.fromarray(rgb_array)

    return watermarked_img


def extract_watermark(watermarked_image, original_image, custom_settings=None):
    """
    Ekstrak watermark dari gambar yang sudah disisipkan watermark

    Parameters:
    watermarked_image (PIL.Image.Image): Gambar yang disisipi watermark
    original_image (PIL.Image.Image): Gambar asli sebelum disisipi watermark
    custom_settings (dict): Pengaturan kustom untuk override nilai default

    Returns:
    PIL.Image.Image: Watermark hasil ekstraksi
    """
    # Dapatkan pengaturan yang akan digunakan
    settings = custom_settings if custom_settings else get_watermark_settings()
    block_size = settings.get("block_size", BLOCK_SIZE)
    alpha = settings.get("alpha", ALPHA)

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
    num_blocks_h = height // block_size
    num_blocks_w = width // block_size

    # Buat array untuk menyimpan hasil ekstraksi watermark
    extracted_watermark = np.zeros((num_blocks_h, num_blocks_w))

    # Iterasi melalui setiap blok
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Ekstrak blok dari kedua gambar
            block_watermarked = Y_watermarked[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
            ]
            block_original = Y_original[
                i * block_size : (i + 1) * block_size,
                j * block_size : (j + 1) * block_size,
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
            extracted_watermark[i, j] = (S_watermarked[0] - S_original[0]) / alpha

    # Normalisasi hasil ke range [0, 255]
    extracted_watermark = np.clip(extracted_watermark, 0, 1)
    extracted_watermark = (extracted_watermark * 255).astype(np.uint8)

    # Buat gambar PIL dari array
    extracted_img = Image.fromarray(extracted_watermark)

    return extracted_img
