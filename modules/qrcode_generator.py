import qrcode
from PIL import Image
import io
from typing import Union, Tuple
import base64

from pyzbar.pyzbar import decode


def text_to_qrcode(
    text: Union[str, bytes], size: Tuple[int, int] = (300, 300)
) -> Image.Image:
    """
    Mengubah teks atau data terenkripsi menjadi QR code dengan ukuran tertentu

    Args:
        text: Teks atau data terenkripsi
        size: Ukuran QR code (width, height)

    Returns:
        PIL Image berisi QR code
    """
    # Konversi bytes menjadi string base64 jika input berupa bytes
    if isinstance(text, bytes):
        text = base64.b64encode(text).decode("utf-8")

    # Buat QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # Error correction level tinggi
        box_size=10,
        border=4,
    )

    qr.add_data(text)
    qr.make(fit=True)

    # Buat gambar QR code
    img = qr.make_image(fill_color="black", back_color="white")

    # Resize ke ukuran yang diinginkan
    img = img.resize(size, Image.LANCZOS)

    return img


def qrcode_to_text(qr_image: Image.Image) -> str:
    """
    Mengekstrak data dari QR code

    Args:
        qr_image: Gambar QR code

    Returns:
        Teks yang terkandung dalam QR code
    """
    try:
        # Decode QR code
        decoded_data = decode(qr_image)

        if decoded_data:
            # Ambil data dari QR code pertama yang ditemukan
            text = decoded_data[0].data.decode("utf-8")

            # Coba decode base64 jika memungkinkan
            try:
                decoded_bytes = base64.b64decode(text)
                return decoded_bytes
            except:
                # Jika bukan base64, kembalikan teks asli
                return text
        else:
            return None
    except Exception as e:
        print(f"Error decoding QR code: {str(e)}")
        return None
