# 🌿 Plant Disease Classification System

Sistem klasifikasi penyakit tanaman berbasis **Deep Learning** menggunakan pendekatan **CNN Transfer Learning** dengan pipeline *Preprocessing Pengolahan Citra Digital (PCD)* untuk meningkatkan kualitas citra sebelum inferensi model.

---

## 📊 Model Performance

| Model              | Accuracy | F1-Score | Inference Time |
| ------------------ | -------- | -------- | -------------- |
| **MobileNetV2**    | 94.09%   | 94.07%   | 50.11 ms       |
| **EfficientNetB0** | 92.89%   | 92.98%   | 97.09 ms       |

📌 *MobileNetV2 menunjukkan performa terbaik dari sisi akurasi dan kecepatan inferensi.*

---

## 🌱 Supported Classes

### 🫑 Pepper Bell

* Bacterial Spot
* Healthy

### 🥔 Potato

* Early Blight
* Late Blight

### 🍅 Tomato

* Early Blight
* Late Blight
* Septoria Leaf Spot
* Spider Mites
* Target Spot
* Healthy

---

## 🔬 Preprocessing Pipeline (PCD)

Tahapan pengolahan citra sebelum masuk ke model:

1. **Resize** → 224×224 piksel
2. **Color Space Conversion** → RGB → LAB
3. **CLAHE Enhancement** pada channel Luminance (L)
4. **Konversi kembali ke RGB**
5. **Normalisasi piksel** ke rentang [0, 1]

Pipeline ini dirancang untuk meningkatkan kontras daun dan memperjelas pola penyakit.

---

## 🛠️ Tech Stack

* Python 3.12
* TensorFlow / Keras
* OpenCV
* Streamlit

---

## 📁 Project Structure

```
plantdisease-classifier/
│
├── app.py
├── models/
│   ├── mobilenetv2_best.h5
│   └── efficientnetb0_best.keras
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

---

## 🚀 Deployment Guide (Streamlit Cloud)

### 1️⃣ Siapkan Struktur Folder

Pastikan struktur repository seperti berikut:

```
plantdisease-classifier/
├── app.py
├── models/
│   ├── mobilenetv2_best.h5
│   └── efficientnetb0_best.keras
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

---

### 2️⃣ Push ke GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/username/plantdisease-classifier.git
git push -u origin main
```

---

### 3️⃣ Deploy di Streamlit Cloud

1. Buka [https://streamlit.io](https://streamlit.io)
2. Sign in menggunakan GitHub
3. Klik **New App**
4. Pilih repository `plantdisease-classifier`
5. Isi **Main file path** dengan: `app.py`
6. Klik **Deploy**