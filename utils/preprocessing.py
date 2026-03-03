import cv2
import numpy as np
from PIL import Image

# ── Konstanta preprocessing (identik dengan Notebook 02)
IMG_SIZE         = (224, 224)
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID  = (8, 8)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Menjalankan pipeline preprocessing PCD lengkap pada satu gambar
    yang diupload melalui antarmuka Streamlit.

    Pipeline identik dengan Notebook 02:
      1. Resize ke 224x224
      2. Konversi RGB → LAB
      3. CLAHE pada channel L (luminance)
      4. Konversi LAB → RGB
      5. Normalisasi piksel ke [0, 1]

    Args:
        img (PIL.Image): Gambar input dari Streamlit file uploader.

    Returns:
        np.ndarray: Array gambar hasil preprocessing,
                    shape (224, 224, 3), dtype float32, nilai [0, 1].
    """
    # Konversi PIL Image ke numpy array BGR untuk OpenCV
    img_rgb = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Tahap 1: Resize
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # ── Tahap 2: Konversi BGR → LAB
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # ── Tahap 3: CLAHE pada channel L
    l, a, b   = cv2.split(img_lab)
    clahe     = cv2.createCLAHE(
        clipLimit    = CLAHE_CLIP_LIMIT,
        tileGridSize = CLAHE_TILE_GRID
    )
    l_clahe   = clahe.apply(l)
    img_clahe = cv2.merge((l_clahe, a, b))

    # ── Tahap 4: Konversi LAB → BGR → RGB
    img_bgr_result = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
    img_rgb_result = cv2.cvtColor(img_bgr_result, cv2.COLOR_BGR2RGB)

    # ── Tahap 5: Normalisasi piksel ke [0, 1]
    img_normalized = img_rgb_result.astype(np.float32) / 255.0

    return img_normalized


def preprocess_for_efficientnet(img: Image.Image) -> np.ndarray:
    """
    Preprocessing khusus untuk EfficientNetB0.
    Pipeline PCD sama namun TANPA normalisasi /255 karena
    EfficientNetB0 memiliki preprocessing internal sendiri
    yang mengharapkan input dalam rentang [0, 255].

    Args:
        img (PIL.Image): Gambar input dari Streamlit file uploader.

    Returns:
        np.ndarray: Array gambar hasil preprocessing,
                    shape (224, 224, 3), dtype float32, nilai [0, 255].
    """
    img_rgb = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b   = cv2.split(img_lab)
    clahe     = cv2.createCLAHE(
        clipLimit    = CLAHE_CLIP_LIMIT,
        tileGridSize = CLAHE_TILE_GRID
    )
    l_clahe   = clahe.apply(l)
    img_clahe = cv2.merge((l_clahe, a, b))

    img_bgr_result = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2BGR)
    img_rgb_result = cv2.cvtColor(img_bgr_result, cv2.COLOR_BGR2RGB)

    # Kembalikan dalam float32 tanpa normalisasi
    return img_rgb_result.astype(np.float32)


def get_display_image(img: Image.Image) -> np.ndarray:
    """
    Menyiapkan gambar untuk ditampilkan di Streamlit setelah
    preprocessing (hanya resize, tanpa normalisasi).

    Args:
        img (PIL.Image): Gambar input asli.

    Returns:
        np.ndarray: Array gambar RGB uint8 ukuran 224x224.
    """
    img_rgb    = np.array(img.convert('RGB'))
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img_resized