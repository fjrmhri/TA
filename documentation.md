# Dokumentasi Optimasi Notebook Hoax Detection + NER

## 1. Ringkasan Dataset
Sumber data yang digunakan (dibaca dari folder `scrape_output/`):
- `data_hoaks_turnbackhoaks.csv`: 12,353 baris (label `hoax=1`)
- `data_nonhoaks_cnn.csv`: 807 baris (label `hoax=0`)
- `data_nonhoaks_detik.csv`: 18,711 baris (label `hoax=0`)
- `data_nonhoaks_kompas.csv`: 5,136 baris (label `hoax=0`)

Total awal gabungan: 37,007 baris.

Distribusi label total awal:
- `0`: 24,654
- `1`: 12,353

Kondisi kolom utama (`url`, `judul`, `tanggal`, `isi_berita`, `Narasi`, `Clean Narasi`, `hoax`, `summary`, `source`):
- Tersedia di seluruh file
- Missing rate 0% pada kolom utama

Strategi deduplikasi minimal yang dipakai di notebook:
1. `drop_duplicates(url_norm)`
2. `drop_duplicates(judul_norm, tanggal_norm)`
3. Setelah text terbentuk, `drop_duplicates(text, label)`

Hasil dedup berdasarkan implementasi notebook (dengan text strategy saat ini):
- Setelah dedup URL: 37,007
- Setelah dedup `judul+tanggal`: 36,965
- Setelah dedup `text+label`: 36,920

Distribusi label final setelah dedup:
- `0`: 24,599
- `1`: 12,321

## 2. Preprocessing
### Pembentukan Text
Representasi teks default:
- Prioritas utama: `judul + " [SEP] " + summary`
- Dipakai jika `summary` memiliki minimal 8 kata (`summary_min_words=8`)

Fallback jika syarat di atas tidak terpenuhi:
- Ambil dari urutan kolom: `Clean Narasi` -> `Narasi` -> `isi_berita` -> `judul` -> `summary`
- Pecah dengan `sent_tokenize`
- Ambil maksimal 3 kalimat awal (`fallback_num_sentences=3`)
- Batasi panjang maksimal 96 kata (`fallback_max_words=96`)

Alasan pemilihan:
- `summary` tersedia 100% pada semua sumber
- Kombinasi `judul + summary` lebih ringkas dan stabil untuk menekan risiko truncation/OOM dibanding memakai artikel penuh

### Split Data
Split dilakukan stratified dengan rasio 70/15/15:
- Train: 25,844
- Validasi: 5,538
- Uji: 5,538

Distribusi label sebelum augmentasi pada train:
- `0`: 17,219
- `1`: 8,625

Catatan penting:
- Augmentasi hanya dilakukan pada `df_latih`
- `df_validasi` dan `df_uji` tidak diaugmentasi

## 3. Augmentasi Minoritas
Metode utama: **NER-based Entity Replacement**
- Jalankan NER pada teks minoritas train (`label` minoritas)
- Kumpulkan pool entitas per tipe: `PER`, `ORG`, `LOC`
- Buat contoh baru dengan mengganti entitas pada teks sumber menggunakan entitas lain bertipe sama

Aturan penggantian:
- Maksimal 1 penggantian entitas per sampel (`augment_max_per_sample=1`)
- Pengganti tidak boleh identik (case-insensitive) dengan entitas asli
- Seed tetap (`augment_seed=42`) agar reproducible

Target augmentasi:
- `augment_target_ratio=1.0` (nyaris seimbang)
- Berdasarkan split saat ini, target minoritas train = 17,219
- Gap awal yang diisi augmentasi = 8,594 sampel

Fallback jika entitas tidak ditemukan/replace gagal:
- EDA ringan (`fallback_eda_ringan`) dengan probabilitas kecil (`augment_fallback_eda_prob=0.05`)
- Operasi: delete/swap sederhana pada token

Notebook menampilkan ringkasan augmentasi:
- total augmented
- augmented via NER replacement
- augmented via fallback EDA

## 4. Model dan Training
Model klasifikasi:
- `indolem/indobert-base-uncased`

Tokenizer:
- tokenizer yang sesuai model di atas

Pengaturan input:
- `max_length=192`
- Dynamic padding dengan `DataCollatorWithPadding`
- `pad_to_multiple_of=8` saat CUDA aktif

Penanganan imbalance:
- Default: `class_weight` (weighted cross-entropy)
- Opsi alternatif via config: `focal` (`focal_gamma=2.0`)

Alasan default `class_weight`:
- Lebih stabil untuk imbalance moderat (~2:1) dan lebih sederhana untuk baseline yang konsisten
- Focal tetap disediakan sebagai opsi eksperimen

Metrik evaluasi (`compute_metrics`):
- Accuracy
- Precision/Recall/F1 Macro
- Precision/Recall/F1 Weighted
- Confusion matrix scalar (`cm_tn`, `cm_fp`, `cm_fn`, `cm_tp`)

Pemilihan best model:
- `metric_for_best_model = "f1_macro"`
- `load_best_model_at_end = True`

Strategi anti-OOM (Colab T4 15GB):
- `fp16=True` saat CUDA
- `gradient_checkpointing=True`
- `auto_find_batch_size=True`
- teks input diringkas (judul+summary/fallback kalimat awal)
- NER inferensi default di CPU (`device=-1`) agar tidak menahan dua model besar bersamaan di GPU
- Saat NER augmentasi dijalankan di GPU dan terjadi OOM, pipeline otomatis fallback ke CPU

## 5. Cara Menjalankan
Urutan cell notebook:
1. Instal dependensi
2. Mount Google Drive
3. Import library
4. Set konfigurasi
5. Load data + preprocessing + split + augmentasi train-only
6. Tokenisasi dan data collator
7. Training + evaluasi validasi/uji + simpan model
8. Inference kalimat + NER
9. Ekspor model ke file ZIP

Requirement utama:
- `transformers`, `datasets`, `accelerate`, `sentencepiece`, `scikit-learn`, `nltk`, `torch`, `pandas`, `numpy`

Artefak output:
- Folder model: `indobert_hoax_ner_model/`
- File ZIP: `indobert_hoax_ner_model_final.zip`
