import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf

from utils.preprocessing import (
    preprocess_image,
    preprocess_for_efficientnet,
    get_display_image
)

# ═══════════════════════════════════════════════════════════
#  KONFIGURASI HALAMAN
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "Plant Disease Classifier",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ═══════════════════════════════════════════════════════════
#  KONSTANTA
# ═══════════════════════════════════════════════════════════
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato_healthy'
]

# Informasi detail setiap kelas penyakit
CLASS_INFO = {
    'Pepper__bell___Bacterial_spot': {
        'display_name' : 'Pepper Bell — Bacterial Spot',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Paprika (Pepper Bell)',
        'penyakit'     : 'Bacterial Spot (Bercak Bakteri)',
        'penyebab'     : 'Bakteri Xanthomonas campestris pv. vesicatoria',
        'gejala'       : (
            'Bercak kecil berwarna coklat kehitaman pada daun, '
            'dikelilingi lingkaran kuning. Bercak dapat menyatu '
            'membentuk area nekrotik yang lebih besar.'
        ),
        'dampak'       : 'Penurunan hasil panen hingga 20–30% jika tidak ditangani.',
        'penanganan'   : (
            'Aplikasi bakterisida berbahan tembaga, '
            'rotasi tanaman, dan penggunaan benih bebas penyakit.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Pepper__bell___healthy': {
        'display_name' : 'Pepper Bell — Healthy',
        'status'       : '🟢 Healthy',
        'tanaman'      : 'Paprika (Pepper Bell)',
        'penyakit'     : 'Tidak ada penyakit',
        'penyebab'     : '-',
        'gejala'       : 'Daun berwarna hijau segar, tidak ada bercak atau perubahan warna.',
        'dampak'       : 'Tidak ada dampak negatif.',
        'penanganan'   : 'Pertahankan kondisi dengan perawatan rutin dan pemupukan yang tepat.',
        'warna_status' : '#00CC44'
    },
    'Potato___Early_blight': {
        'display_name' : 'Potato — Early Blight',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Kentang (Potato)',
        'penyakit'     : 'Early Blight (Hawar Daun Awal)',
        'penyebab'     : 'Jamur Alternaria solani',
        'gejala'       : (
            'Bercak coklat tua berbentuk konsentris (seperti cincin) '
            'pada daun tua. Bercak dikelilingi area kuning.'
        ),
        'dampak'       : 'Penurunan hasil panen kentang hingga 30% pada infeksi berat.',
        'penanganan'   : (
            'Aplikasi fungisida seperti chlorothalonil atau mancozeb, '
            'pemangkasan daun terinfeksi, dan rotasi tanaman.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Potato___Late_blight': {
        'display_name' : 'Potato — Late Blight',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Kentang (Potato)',
        'penyakit'     : 'Late Blight (Hawar Daun Akhir)',
        'penyebab'     : 'Oomycete Phytophthora infestans',
        'gejala'       : (
            'Bercak berwarna hijau keabu-abuan hingga coklat gelap, '
            'tepi tidak beraturan, sering disertai pertumbuhan jamur '
            'putih pada bagian bawah daun saat lembab.'
        ),
        'dampak'       : (
            'Sangat destruktif, dapat menghancurkan seluruh tanaman '
            'dalam waktu singkat. Penyebab Kelaparan Besar Irlandia 1840s.'
        ),
        'penanganan'   : (
            'Fungisida sistemik (metalaxyl), varietas tahan penyakit, '
            'dan pengendalian kelembaban lingkungan.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Tomato_Early_blight': {
        'display_name' : 'Tomato — Early Blight',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Tomat (Tomato)',
        'penyakit'     : 'Early Blight (Hawar Daun Awal)',
        'penyebab'     : 'Jamur Alternaria solani',
        'gejala'       : (
            'Bercak coklat dengan pola cincin konsentris pada daun bagian bawah. '
            'Daun menguning di sekitar bercak dan akhirnya gugur.'
        ),
        'dampak'       : 'Kehilangan daun besar-besaran menyebabkan penurunan fotosintesis.',
        'penanganan'   : (
            'Fungisida tembaga atau klorotalonil, '
            'hindari penyiraman dari atas, mulsa untuk mencegah percikan tanah.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Tomato_Late_blight': {
        'display_name' : 'Tomato — Late Blight',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Tomat (Tomato)',
        'penyakit'     : 'Late Blight (Hawar Daun Akhir)',
        'penyebab'     : 'Oomycete Phytophthora infestans',
        'gejala'       : (
            'Lesi berwarna hijau gelap hingga coklat berminyak pada daun, '
            'batang, dan buah. Pertumbuhan jamur putih pada kondisi lembab.'
        ),
        'dampak'       : 'Dapat menghancurkan seluruh kebun tomat dalam 7–10 hari.',
        'penanganan'   : (
            'Fungisida preventif dan kuratif, '
            'pemangkasan bagian terinfeksi, peningkatan sirkulasi udara.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Tomato_Septoria_leaf_spot': {
        'display_name' : 'Tomato — Septoria Leaf Spot',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Tomat (Tomato)',
        'penyakit'     : 'Septoria Leaf Spot (Bercak Daun Septoria)',
        'penyebab'     : 'Jamur Septoria lycopersici',
        'gejala'       : (
            'Bercak kecil melingkar berwarna putih keabu-abuan dengan '
            'tepi coklat gelap. Titik hitam kecil (pycnidia) terlihat '
            'di tengah bercak.'
        ),
        'dampak'       : 'Defoliasi parah menyebabkan buah terbakar sinar matahari.',
        'penanganan'   : (
            'Fungisida tembaga, klorotalonil, atau mankozeb. '
            'Hindari kelembaban berlebih dan penyiraman dari atas.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        'display_name' : 'Tomato — Spider Mites',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Tomat (Tomato)',
        'penyakit'     : 'Spider Mites — Two-spotted Spider Mite',
        'penyebab'     : 'Tungau Tetranychus urticae',
        'gejala'       : (
            'Stippling (titik-titik kuning kecil) pada permukaan daun, '
            'daun menguning dan mengering. Jaring halus terlihat '
            'pada serangan berat.'
        ),
        'dampak'       : 'Penurunan fotosintesis dan kualitas buah secara signifikan.',
        'penanganan'   : (
            'Akarisida (abamectin, bifenazate), '
            'pelepasan predator alami (Phytoseiidae), '
            'dan peningkatan kelembaban udara.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Tomato__Target_Spot': {
        'display_name' : 'Tomato — Target Spot',
        'status'       : '🔴 Diseased',
        'tanaman'      : 'Tomat (Tomato)',
        'penyakit'     : 'Target Spot (Bercak Target)',
        'penyebab'     : 'Jamur Corynespora cassiicola',
        'gejala'       : (
            'Bercak coklat besar dengan pola cincin konsentris '
            'menyerupai target/sasaran tembak. '
            'Dapat menyerang daun, batang, dan buah.'
        ),
        'dampak'       : 'Defoliasi dan penurunan kualitas serta kuantitas buah.',
        'penanganan'   : (
            'Fungisida azoxystrobin atau difenokonazol, '
            'rotasi tanaman, dan sanitasi lahan.'
        ),
        'warna_status' : '#FF4B4B'
    },
    'Tomato_healthy': {
        'display_name' : 'Tomato — Healthy',
        'status'       : '🟢 Healthy',
        'tanaman'      : 'Tomat (Tomato)',
        'penyakit'     : 'Tidak ada penyakit',
        'penyebab'     : '-',
        'gejala'       : 'Daun hijau segar, permukaan bersih tanpa bercak atau perubahan warna.',
        'dampak'       : 'Tidak ada dampak negatif.',
        'penanganan'   : 'Pertahankan kondisi optimal dengan pemupukan dan irigasi yang tepat.',
        'warna_status' : '#00CC44'
    },
}

# ═══════════════════════════════════════════════════════════
#  LOAD MODEL (CACHED)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_mobilenet() -> tf.keras.Model:
    """
    Memuat model MobileNetV2 dari file .h5 yang tersimpan
    di dalam repository dan menyimpannya di cache Streamlit
    agar tidak perlu di-load ulang setiap kali ada interaksi.

    Returns:
        tf.keras.Model: Model MobileNetV2 siap inferensi.
    """
    model = tf.keras.models.load_model('models/mobilenetv2_best.h5')
    return model


@st.cache_resource
def load_efficientnet() -> tf.keras.Model:
    """
    Memuat model EfficientNetB0 dari file .keras yang tersimpan
    di dalam repository. Arsitektur dibangun ulang menggunakan
    custom EfficientNetPreprocessing layer sebagai pengganti
    Lambda layer, lalu bobot di-load dari file .keras.

    Returns:
        tf.keras.Model: Model EfficientNetB0 siap inferensi.
    """
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, models

    class EfficientNetPreprocessing(tf.keras.layers.Layer):
        def call(self, inputs):
            from tensorflow.keras.applications.efficientnet import preprocess_input
            return preprocess_input(inputs)

        def get_config(self):
            return super().get_config()

    input_shape = (224, 224, 3)
    base_model  = EfficientNetB0(
        input_shape = input_shape,
        include_top = False,
        weights     = None
    )

    inputs  = tf.keras.Input(shape=input_shape)
    x       = EfficientNetPreprocessing()(inputs)
    x       = base_model(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model   = models.Model(inputs, outputs)
    model.load_weights('models/efficientnetb0_best.keras')
    return model

# ═══════════════════════════════════════════════════════════
#  FUNGSI HELPER
# ═══════════════════════════════════════════════════════════
def predict(model: tf.keras.Model, img: Image.Image, model_name: str) -> tuple:
    """
    Menjalankan inferensi model pada gambar yang diupload.

    Args:
        model      (tf.keras.Model): Model yang digunakan.
        img        (PIL.Image)     : Gambar input.
        model_name (str)           : Nama model untuk menentukan preprocessing.

    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    if model_name == 'EfficientNetB0':
        img_array = preprocess_for_efficientnet(img)
    else:
        img_array = preprocess_image(img)

    img_array  = np.expand_dims(img_array, axis=0)
    pred_probs = model.predict(img_array, verbose=0)[0]
    pred_idx   = np.argmax(pred_probs)

    return CLASS_NAMES[pred_idx], float(pred_probs[pred_idx]), pred_probs


def plot_confidence_bar(probs: np.ndarray, predicted_class: str) -> plt.Figure:
    """
    Membuat bar chart horizontal confidence score untuk semua kelas,
    dengan bar kelas terprediksi disorot dengan warna berbeda.

    Args:
        probs           (np.ndarray): Array probabilitas untuk semua kelas.
        predicted_class (str)       : Nama kelas hasil prediksi.

    Returns:
        plt.Figure: Figure matplotlib siap ditampilkan di Streamlit.
    """
    short_names = [
        c.replace('Pepper__bell___', 'Pepper_')
         .replace('Potato___', 'Potato_')
         .replace('Tomato_', 'Tom_')
         .replace('Two_spotted_spider_mite', 'SpiderMite')
         .replace('__', '_')
        for c in CLASS_NAMES
    ]

    colors = [
        '#FF4B4B' if c == predicted_class else '#AAAAAA'
        for c in CLASS_NAMES
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(short_names, probs * 100, color=colors, edgecolor='white')

    for bar, val in zip(bars, probs * 100):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}%',
            va='center', fontsize=8
        )

    ax.set_xlabel('Confidence (%)')
    ax.set_title('Confidence Score per Kelas', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.grid(axis='x', alpha=0.3)

    legend = [
        mpatches.Patch(color='#FF4B4B', label='Predicted Class'),
        mpatches.Patch(color='#AAAAAA', label='Other Classes')
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/'
        'Sunflower_from_Silesia2.jpg/320px-Sunflower_from_Silesia2.jpg',
        use_column_width=True
    )
    st.title("🌿 Plant Disease Classifier")
    st.markdown("---")

    st.subheader("⚙️ Konfigurasi Model")
    model_choice = st.selectbox(
        label   = "Pilih Model",
        options = ["MobileNetV2", "EfficientNetB0"],
        help    = (
            "MobileNetV2: Lebih cepat, akurasi 94.09%\n"
            "EfficientNetB0: Lebih berat, akurasi 92.89%"
        )
    )

    st.markdown("---")
    st.subheader("📊 Informasi Model")

    if model_choice == "MobileNetV2":
        st.metric("Accuracy (Test Set)", "94.09%")
        st.metric("F1-Score (Macro)", "94.07%")
        st.metric("Inference Time", "50.11 ms")
    else:
        st.metric("Accuracy (Test Set)", "92.89%")
        st.metric("F1-Score (Macro)", "92.98%")
        st.metric("Inference Time", "97.09 ms")

    st.markdown("---")
    st.subheader("🌱 Tanaman yang Didukung")
    st.markdown("""
    - 🌶️ **Pepper Bell** (Paprika)
    - 🥔 **Potato** (Kentang)
    - 🍅 **Tomato** (Tomat)
    """)

    st.markdown("---")
    st.caption(
        "Dibuat menggunakan CNN Transfer Learning\n"
        "Dataset: PlantVillage (10 kelas)\n"
        "Preprocessing: CLAHE + Normalisasi"
    )


# ═══════════════════════════════════════════════════════════
#  HALAMAN UTAMA
# ═══════════════════════════════════════════════════════════
st.title("🌿 Plant Disease Classification System")
st.markdown(
    "Sistem klasifikasi penyakit tanaman berbasis **Deep Learning** "
    "menggunakan arsitektur **CNN Transfer Learning** dengan preprocessing "
    "**Pengolahan Citra Digital (PCD)**."
)
st.markdown("---")

# ── Upload gambar
st.subheader("📤 Upload Gambar Daun")
uploaded_file = st.file_uploader(
    label       = "Pilih gambar daun (JPG / PNG)",
    type        = ["jpg", "jpeg", "png"],
    help        = "Upload foto daun tanaman untuk diklasifikasikan."
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # ── Layout dua kolom: gambar & hasil prediksi
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(img, caption="Gambar yang diupload", use_column_width=True)

        # Tampilkan gambar setelah preprocessing
        with st.expander("🔬 Lihat Gambar Setelah Preprocessing PCD"):
            img_preprocessed = get_display_image(img)
            st.image(
                img_preprocessed,
                caption = "Setelah Resize + CLAHE",
                use_column_width = True
            )
            st.caption(
                "Pipeline PCD: Resize 224×224 → "
                "RGB→LAB → CLAHE → LAB→RGB → Normalisasi"
            )

    with col2:
        st.subheader("🔍 Hasil Prediksi")

        # Load model sesuai pilihan
        with st.spinner(f"Memuat model {model_choice}..."):
            if model_choice == "MobileNetV2":
                model = load_mobilenet()
            else:
                model = load_efficientnet()

        # Jalankan prediksi
        with st.spinner("Menganalisis gambar..."):
            pred_class, confidence, all_probs = predict(model, img, model_choice)

        info = CLASS_INFO[pred_class]

        # ── Kotak hasil prediksi utama
        st.markdown(
            f"""
            <div style="
                background-color: {info['warna_status']}22;
                border-left: 5px solid {info['warna_status']};
                padding: 16px;
                border-radius: 8px;
                margin-bottom: 16px;
            ">
                <h3 style="margin:0; color: {info['warna_status']};">
                    {info['status']}
                </h3>
                <h4 style="margin:4px 0;">{info['display_name']}</h4>
                <p style="margin:0; font-size:24px; font-weight:bold;">
                    Confidence: {confidence * 100:.2f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ── Progress bar confidence
        st.progress(float(confidence))

        # ── Model yang digunakan
        st.caption(f"Model: {model_choice}")

    st.markdown("---")

    # ── Confidence bar chart
    st.subheader("📊 Distribusi Confidence Score")
    fig = plot_confidence_bar(all_probs, pred_class)
    st.pyplot(fig)

    st.markdown("---")

    # ── Informasi detail penyakit
    st.subheader("📋 Informasi Detail")

    detail_col1, detail_col2 = st.columns(2, gap="large")

    with detail_col1:
        st.markdown("#### 🌱 Informasi Tanaman & Penyakit")
        st.table(pd.DataFrame({
            'Kategori'  : ['Tanaman', 'Penyakit', 'Penyebab'],
            'Keterangan': [
                info['tanaman'],
                info['penyakit'],
                info['penyebab']
            ]
        }))

    with detail_col2:
        st.markdown("#### ⚠️ Gejala & Dampak")
        st.info(f"**Gejala:**\n\n{info['gejala']}")
        st.warning(f"**Dampak:**\n\n{info['dampak']}")

    st.markdown("#### 💊 Rekomendasi Penanganan")
    st.success(f"**Penanganan:**\n\n{info['penanganan']}")

    st.markdown("---")

    # ── Top-3 prediksi
    st.subheader("🏆 Top-3 Prediksi")
    top3_idx    = np.argsort(all_probs)[::-1][:3]
    top3_cols   = st.columns(3)

    for i, (col, idx) in enumerate(zip(top3_cols, top3_idx)):
        cls_name   = CLASS_NAMES[idx]
        cls_info   = CLASS_INFO[cls_name]
        prob       = all_probs[idx] * 100
        medal      = ["🥇", "🥈", "🥉"][i]

        with col:
            st.metric(
                label = f"{medal} #{i+1}",
                value = f"{prob:.2f}%",
                delta = cls_info['display_name']
            )

else:
    # ── Tampilan default saat belum ada gambar
    st.info(
        "👆 Upload gambar daun tanaman untuk memulai klasifikasi.\n\n"
        "Tanaman yang didukung: **Pepper Bell**, **Potato**, **Tomato**"
    )

    st.markdown("---")
    st.subheader("📖 Tentang Sistem Ini")

    about_col1, about_col2, about_col3 = st.columns(3)

    with about_col1:
        st.markdown("""
        #### 🧠 Model
        - **MobileNetV2** — Akurasi 94.09%
        - **EfficientNetB0** — Akurasi 92.89%
        - Transfer Learning dari ImageNet
        - Fine-tuning 2 fase
        """)

    with about_col2:
        st.markdown("""
        #### 🔬 Preprocessing PCD
        - Resize 224×224 piksel
        - Color Space RGB → LAB
        - CLAHE Enhancement
        - Normalisasi Piksel
        """)

    with about_col3:
        st.markdown("""
        #### 🌿 Kelas yang Didukung
        - 3 spesies tanaman
        - 10 kelas total
        - 7 jenis penyakit
        - 3 kelas sehat
        """)