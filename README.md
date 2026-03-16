# 🔍 BERT Fake News Detector

> Sistem Deteksi Berita Palsu pada Platform Berita Online Menggunakan Metode **Bidirectional Encoder Representations from Transformers (BERT)**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![IndoBERT](https://img.shields.io/badge/IndoBERT-Fine--tuned-purple?style=flat-square)

---

## 📌 Deskripsi

Penelitian ini mengembangkan model deteksi berita palsu berbasis **IndoBERT** yang di-*fine-tune* untuk mengklasifikasikan judul berita Bahasa Indonesia ke dalam dua kelas:

- 🔴 **FAKE** — Berita palsu / hoaks (sumber: TurnBackHoax.id)
- 🟢 **REAL** — Berita valid (sumber: Antara News)

Model berhasil mencapai **akurasi 87.50%** dengan **Macro F1-Score 87.43%** pada data uji.

---

## 📊 Hasil Evaluasi Model

| Metrik | FAKE | REAL | Macro Avg |
|--------|------|------|-----------|
| Precision | 0.9412 | 0.8261 | 0.8836 |
| Recall | 0.8000 | 0.9500 | 0.8750 |
| F1-Score | 0.8649 | 0.8837 | 0.8743 |
| **Accuracy** | — | — | **0.8750** |

| Metrik Tambahan | Nilai |
|-----------------|-------|
| Sensitivity (TPR) | 0.9500 |
| Specificity (TNR) | 0.8000 |

---

## 🗂️ Struktur Project

```
bert-deteksi/
│
├── app.py                          # Flask application & routing
├── requirements.txt                # Dependencies
├── README.md
│
├── flask_components/               # Output dari Jupyter Notebook
│   ├── models/
│   │   ├── bert_model/             # ⚠️ tidak di-upload (terlalu besar)
│   │   └── bert_tokenizer/         # ⚠️ tidak di-upload (terlalu besar)
│   ├── data/
│   │   ├── dataset_info.json
│   │   ├── splitting_info.json
│   │   └── word_analysis.json
│   ├── results/
│   │   ├── final_summary.json
│   │   ├── evaluation_results.json
│   │   ├── training_results.json
│   │   └── sample_test_results.csv
│   ├── visualizations/
│   │   ├── confusion_matrix.html
│   │   ├── metrics_per_class.html
│   │   ├── label_distribution.html
│   │   ├── source_distribution.html
│   │   ├── top_words_fake.html
│   │   ├── top_words_real.html
│   │   └── title_length_boxplot.html
│   └── flask_config.json           # Konfigurasi path (auto-generated)
│
├── templates/
│   ├── index.html                  # Halaman publik / beranda
│   ├── layouts/
│   │   └── main_layout.html        # Layout dashboard admin
│   ├── admin/
│   │   └── login.html              # Halaman login
│   ├── dashboard/
│   │   └── index.html              # Dashboard admin
│   ├── analisis/
│   │   └── index.html              # Halaman hasil analisis
│   └── tentang/
│       └── index.html              # Halaman tentang penelitian
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

---

## ⚙️ Konfigurasi Model

| Parameter | Nilai |
|-----------|-------|
| Base Model | `indolem/indobert-base-uncased` |
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 4 |
| Max Length | 128 token |
| Optimizer | AdamW |
| Framework | PyTorch + HuggingFace Trainer |

---

## 📦 Dataset

| Keterangan | Nilai |
|------------|-------|
| Total Sampel | 196 berita |
| Data Latih | 156 sampel (80%) |
| Data Uji | 40 sampel (20%) |
| Balancing | 50:50 (FAKE:REAL) |
| Sumber FAKE | TurnBackHoax.id |
| Sumber REAL | Antara News |

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/bert-fakenews-detector.git
cd bert-fakenews-detector
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Model BERT

Jalankan seluruh cell di `train2.ipynb` terlebih dahulu untuk men-generate folder `flask_components/` beserta model, tokenizer, dan semua file JSON/HTML hasil analisis.

> ⚠️ Pastikan folder `flask_components/models/bert_model/` dan `flask_components/models/bert_tokenizer/` sudah tersedia sebelum menjalankan Flask.

### 5. Jalankan Aplikasi

```bash
python app.py
```

Akses di browser: **http://localhost:5000**

---

## 🔑 Akses Admin

| | |
|---|---|
| URL | http://localhost:5000/login |
| Username | `admin` |
| Password | `admin123` |

---

## 🌐 Endpoint API

### `POST /predict`

Endpoint untuk prediksi real-time via AJAX.

**Request:**
```json
{
  "text": "SALAH Video Warga Israel Mengungsi di Gurun Pasir"
}
```

**Response:**
```json
{
  "label": "FAKE",
  "confidence": 0.9234,
  "prob_fake": 0.9234,
  "prob_real": 0.0766,
  "text": "SALAH Video Warga Israel Mengungsi di Gurun Pasir"
}
```

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|-------|-----------|
| Model | IndoBERT (indobenchmark/indobert-base-p1) |
| Deep Learning | PyTorch + HuggingFace Transformers |
| Backend | Flask (Python) |
| Frontend | HTML5, CSS3, JavaScript |
| Visualisasi | Plotly (interactive charts) |
| Icons | Phosphor Icons |
| Font | Plus Jakarta Sans |

---

## 📁 File Penting

| File | Keterangan |
|------|------------|
| `app.py` | Entry point Flask, routing, model inference |
| `train2.ipynb` | Notebook training & evaluasi model BERT |
| `flask_components/flask_config.json` | Peta path semua file output notebook |
| `flask_components/results/final_summary.json` | Ringkasan performa model |
| `flask_components/results/evaluation_results.json` | Detail hasil evaluasi |

---

## ⚠️ Catatan

- Folder `flask_components/models/` **tidak di-upload** ke GitHub karena ukuran file model BERT mencapai ratusan MB (melebihi batas GitHub 100MB). Jalankan notebook untuk men-generate ulang.
- Semua nilai metrik di tampilan web dibaca **secara otomatis** dari file JSON output notebook — tidak ada nilai yang di-hardcode di `app.py`.

---

## 👤 Author

Penelitian Skripsi — Deteksi Berita Palsu Menggunakan BERT  
Bahasa Indonesia · Natural Language Processing · Text Classification