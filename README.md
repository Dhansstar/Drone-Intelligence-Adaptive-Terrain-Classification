---
title: DetectionApp
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
- tensorflow
- xception
- computer-vision
pinned: false
short_description: Autonomous Terrain Classification with Xception & Safety Logic


---

# 🚁 Drone Intelligence: Adaptive Terrain Classification

**Autonomous Geospasial Mapping with Dual-Model Strategy (CNN & Xception)**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## 📌 1. Project Overview & Problem Statement
Di era industri 4.0, tantangan utama klasifikasi gambar otomatis oleh drone adalah reliabilitas. Model sering mengalami **"AI Hallucination"** pada data *Out-of-Distribution* (OOD).

* **Problem:** Tekstur loreng harimau di tengah hutan sering disalahartikan sebagai *Glacier* atau *Mountain* karena kemiripan pola fraktal. Kesalahan identifikasi berakibat fatal pada navigasi lapangan.
* **Objective:** Membangun sistem klasifikasi 6 medan (*Buildings, Forest, Glacier, Mountain, Sea, Street*) dengan mekanisme **Confidence Thresholding** untuk menjamin keamanan operasional.

---

## 🔬 2. Exploratory Data Analysis (EDA) Insight
Berdasarkan analisis distribusi warna dan tepi (*edge detection*):
* **High Similarity:** Kelas *Glacier* dan *Mountain* memiliki kemiripan struktur tepi yang sangat tinggi.
* **Man-Made vs Natural:** *Buildings* dan *Street* dibedakan melalui pola garis lurus dan sudut tajam.
* **Color Distribution:** *Forest* mendominasi kanal *Green*, sementara *Sea* memuncak di kanal *Blue*.

---

## 🛠️ 3. Feature Engineering & Augmentation
Strategi augmentasi untuk menghadapi guncangan drone dan variasi cahaya:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Inisiasi Augmentasi Data
datagen = ImageDataGenerator(
    rescale=1./255,           # Normalisasi nilai piksel (0-1)
    rotation_range=15,        # Simulasi kondisi drone yang miring (tilt)
    width_shift_range=0.1,    
    height_shift_range=0.1,   # Simulasi objek yang tidak selalu di tengah frame
    zoom_range=0.1,           # Simulasi perubahan jarak/ketinggian drone
    horizontal_flip=True,     
    brightness_range=[0.9, 1.1],
    validation_split=0.2      # Split 20% untuk validasi
)

train_gen = datagen.flow_from_directory(
    'dataset/train', 
    target_size=(150, 150), 
    batch_size=32,
    class_mode='categorical', 
    subset='training'
)
```
---

## 🧠 4. Model Architecture: The Improvement (Xception)
Menggunakan Transfer Learning dengan arsitektur Xception untuk memisahkan pencarian pola spasial dan kanal warna secara independen (Depthwise Separable Convolutions).

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Membangun Arsitektur Model Improvement
model_imp = Sequential([
    base_model,                # Xception Pre-trained Weights
    GlobalAveragePooling2D(),  # Mereduksi parameter & menjaga info spasial
    BatchNormalization(),    
    Dense(256, activation='relu'),
    Dropout(0.4),              # Mencegah overfitting
    Dense(6, activation='softmax') 
])

# Kompilasi Model
model_imp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks untuk stabilitas training
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]
```
---

## 📊 5. Performance Evaluation
Model mencapai akurasi 91%.


| Category | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| 🌲 **Forest** | 0.99 | 0.99 | **0.99** | 🏆 **Best** |
| 🌊 **Sea** | 0.93 | 0.96 | 0.94 | ✅ Solid |
| 🏢 **Buildings** | 0.89 | 0.91 | 0.90 | ✅ Solid |
| 🧊 **Glacier** | 0.85 | 0.86 | 0.86 | ⚠️ Challenging |
| 🏔️ **Mountain** | 0.88 | 0.83 | 0.85 | ⚠️ Challenging |
| 🛣️ **Street** | 0.91 | 0.88 | 0.89 | ✅ Solid |

> **Note:** Kelas *Glacier* dan *Mountain* memiliki tantangan ekstra karena kemiripan tekstur visual, sehingga implementasi *Confidence Thresholding* sangat krusial di sini.

---

### 🛡️ 6. Model Inference with Safety Logic
Mekanisme pertahanan agar model tidak "memaksa" menebak jika tingkat kepercayaan rendah.

```python
def predict_drone_vision(img_path, threshold=0.69):
    # Preprocessing Image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    predictions = model.predict(img_array, verbose=0)
    score = np.max(predictions)
    class_idx = np.argmax(predictions)

    # LOGIKA SAFETY: Confidence Thresholding
    if score < threshold:
        return "WARNING: UNKNOWN OBJECT DETECTED", score
    else:
        return class_names[class_idx], score
```

---

## 🚀 7. Deployment Detail (Hugging Face Spaces)
Sistem dijalankan dalam container Docker untuk menjamin portabilitas.

**SDK: Docker (Python 3.11-slim)**

**Storage: Git LFS (Large File Storage) untuk model .h5**

```python
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src/ ./src/
EXPOSE 8501
ENTRYPOINT ["python3", "-m", "streamlit", "run", "src/streamlit_app.py"]
```

---

## 📁 8. Project Structure
Berikut adalah susunan direktori repositori ini:

```text
.
├── deployment/              # Folder Konfigurasi Deployment
│   ├── src/                 # Source Code Aplikasi
│   │   ├── streamlit_app.py # Logika Utama Dashboard Streamlit
│   │   └── drone_model.h5   # Saved Xception Model Weights
│   ├── Dockerfile           # Setup Container (Python 3.11-slim)
│   ├── requirements.txt     # Library Dependencies (TF, Streamlit, etc.)
│   └── .gitattributes       # Konfigurasi Git LFS untuk File Besar
├── analysis.ipynb           # Notebook Analisis Utama
├── analysis_inf.ipynb       # Notebook Simulasi Inferensi
├── url                      # Referensi URL Deployment
└── README.md
```

---

## 🏁 9. Conclusion
Kombinasi Transfer Learning dan Safety Logic menciptakan visi drone yang tidak hanya akurat, tapi juga reliabel terhadap gangguan dunia nyata.

**Author: Risyadhana Syaifuddin**