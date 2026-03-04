# Dokumentasi Optimasi Notebook Hoax Detection + NER

## 1) Ringkasan Dataset
Sumber dataset yang digunakan:
- `data_hoaks_turnbackhoaks.csv`: 12.353 baris, label `hoax=1`
- `data_nonhoaks_cnn.csv`: 807 baris, label `hoax=0`
- `data_nonhoaks_detik.csv`: 18.711 baris, label `hoax=0`
- `data_nonhoaks_kompas.csv`: 5.136 baris, label `hoax=0`

Total awal gabungan: **37.007 baris**.

Kolom yang tersedia pada keempat file:
- `url`, `judul`, `tanggal`, `isi_berita`, `Narasi`, `Clean Narasi`, `hoax`, `summary`, `source`

Tipe data dan missing rate (hasil audit):
- Kolom teks bertipe `object`
- Kolom `hoax` bertipe `int64`
- Missing rate pada kolom utama: **0%** di semua file

Distribusi label awal:
- Total: `0 = 24.654`, `1 = 12.353`
- Per file:
  - TurnBackHoax: `{1: 12353}`
  - CNN: `{0: 807}`
  - Detik: `{0: 18711}`
  - Kompas: `{0: 5136}`

Statistik panjang teks kandidat (karakter, ringkas):
- TurnBackHoax:
  - `Clean Narasi` mean 354,5 | p90 516
  - `Narasi` mean 463,8 | p90 690
  - `isi_berita` mean 3966,5 | p90 5002
  - `judul` mean 68,5 | p90 98
  - `summary` mean 463,8 | p90 690
- CNN:
  - `Clean Narasi` mean 5814,2 | p90 8542,4
  - `Narasi` mean 8146,5 | p90 12365,6
  - `isi_berita` mean 8146,5 | p90 12365,6
  - `judul` mean 59,8 | p90 69
  - `summary` mean 388,2 | p90 546
- Detik:
  - `Clean Narasi` mean 2732,6 | p90 4390
  - `Narasi` mean 3840,6 | p90 6319
  - `isi_berita` mean 3840,6 | p90 6319
  - `judul` mean 62,5 | p90 76
  - `summary` mean 249,3 | p90 380
- Kompas:
  - `Clean Narasi` mean 3860,8 | p90 5268,5
  - `Narasi` mean 5405,4 | p90 7232
  - `isi_berita` mean 5405,4 | p90 7232
  - `judul` mean 73,2 | p90 91
  - `summary` mean 276,2 | p90 371

Strategi dedup minimal yang diterapkan:
1. `drop_duplicates(url_norm)`
2. `drop_duplicates(judul_norm, tanggal_norm)`
3. setelah text terbentuk: `drop_duplicates(text, label)`

Hasil dedup:
- Duplikasi URL: `0`
- Duplikasi `judul+tanggal`: `42`
- Duplikasi `text+label`: `45`
- Total final setelah dedup: **36.920**
- Distribusi label final: `0 = 24.599`, `1 = 12.321`

## 2) Preprocessing
Pembentukan text utama:
- Prioritas: `judul + " [SEP] " + summary`
- Dipakai jika `summary` memiliki minimal 8 kata (`summary_min_words=8`)
- `summary` dibatasi ke 96 kata (`summary_max_words=96`) untuk menekan outlier panjang

Fallback jika syarat prioritas tidak terpenuhi:
- Sumber fallback berurutan: `Clean Narasi -> Narasi -> isi_berita -> judul -> summary`
- Ambil kalimat awal dengan `sent_tokenize` (`fallback_num_sentences=3`)
- Batasi hingga 96 kata (`fallback_max_words=96`)

Alasan keputusan representasi text:
- `summary` tersedia pada seluruh file dan relatif ringkas dibanding `isi_berita/Narasi`
- Strategi `judul+summary` stabil untuk klasifikasi dan menekan risiko truncation/OOM
- Penggunaan aktual strategi: `judul_summary = 36.740` baris (99,51%), fallback = `180` baris

Pembersihan minimal:
- Normalisasi whitespace (`\s+ -> satu spasi`)
- Trim string pada pembentukan text

Split dataset:
- Stratified split 70/15/15 (`random_state=42`)
- Komposisi:
  - Train: 25.844
  - Validasi: 5.538
  - Uji: 5.538
- Distribusi train sebelum augmentasi:
  - `0: 17.219`
  - `1: 8.625`

Kontrol train-only:
- Augmentasi dijalankan **hanya** pada `df_latih`
- Notebook menambahkan assert agar ukuran dan label `df_validasi`/`df_uji` tidak berubah

## 3) Augmentasi Minoritas
Metode utama: **NER-based Entity Replacement** (selaras arsitektur NER notebook)

Alur:
1. Ambil sampel kelas minoritas dari set train
2. Jalankan NER (`cahya/bert-base-indonesian-NER`) pada teks minoritas
3. Bangun pool entitas per tipe: `PER`, `ORG`, `LOC`
4. Bentuk teks augmented dengan mengganti entitas menggunakan entitas lain bertipe sama

Aturan penggantian:
- Token entitas dibersihkan dulu (`strip`, normalisasi spasi, buang noise tepi)
- Replacement pakai regex word-boundary (bukan `str.replace` mentah)
- Maksimal 1 penggantian per sampel (`augment_max_per_sample=1`)
- Gunakan `seed=42` (`augment_seed`) untuk reproducibility
- Cegah duplikasi sejak awal (dibandingkan dengan data train asli dan antar hasil augment)

Batas augmentasi:
- Target rasio minoritas mendekati mayoritas (`augment_target_ratio=1.0`)
- Putaran augment dibatasi (`augment_max_rounds=6`) agar dataset tidak meledak

Fallback saat NER tidak efektif:
- EDA ringan (`augment_fallback_eda_prob=0.05`): swap/delete kecil

Catatan penting:
- Augmentasi tidak diterapkan pada validasi/uji
- Notebook mencetak jumlah augmented aktual (`augmented_total`, `augmented_ner`, `augmented_eda`)

## 4) Model dan Training
Model klasifikasi:
- `indolem/indobert-base-uncased`

Tokenizer:
- `AutoTokenizer` sesuai model di atas

Konfigurasi input:
- `max_length=192`
- Dynamic padding dengan `DataCollatorWithPadding`
- `pad_to_multiple_of=8` saat CUDA

Penanganan imbalance:
- Default: `class_weight` (weighted cross-entropy)
- Opsi alternatif: `focal` (`focal_gamma=2.0`)
- Alasan default `class_weight`: baseline lebih stabil dan sederhana untuk ketidakseimbangan kelas

Metrik evaluasi (`compute_metrics`):
- Accuracy
- Precision/Recall/F1 Macro
- Precision/Recall/F1 Weighted
- Confusion matrix scalar: `cm_tn`, `cm_fp`, `cm_fn`, `cm_tp`

Pemilihan best model:
- `metric_for_best_model="f1_macro"`
- `load_best_model_at_end=True`

Early stopping:
- `EarlyStoppingCallback`
- `early_stopping_patience=2`
- `early_stopping_threshold=0.0`

Strategi anti-OOM (Colab T4 15GB):
- `fp16=True` saat CUDA
- `gradient_checkpointing=True`
- `auto_find_batch_size=True`
- Input diringkas (`judul+summary` + fallback terkontrol)
- NER inferensi default di CPU (`device=-1`)
- Opsi NER GPU tetap ada, dengan mode aman offload classifier (`ner_safe_offload_classifier=True`)

## 5) Cara Menjalankan
Urutan cell notebook:
1. Install dependency
2. Mount Google Drive
3. Import library
4. Set konfigurasi
5. Load data + audit dataset + preprocessing + split + augmentasi train-only
6. Tokenisasi + data collator
7. Training + evaluasi validasi/uji + simpan model
8. Inference integrasi klasifikasi + NER
9. Kompres model + unduh ZIP

Requirements:
- `transformers`, `datasets`, `accelerate`, `sentencepiece`, `scikit-learn`, `nltk`, `torch`, `pandas`, `numpy`

Artefak output:
- Folder model: `indobert_hoax_ner_model/`
- File zip: `indobert_hoax_ner_model_final.zip`

## 6) Catatan Validasi
Checklist yang diverifikasi pada implementasi:
- Notebook JSON valid dan sel Python utama (3,4,5,6,7,8,9) lolos parse sintaks
- `metric_for_best_model` memakai `f1_macro`
- `auto_find_batch_size`, `fp16` (conditional CUDA), dan `gradient_checkpointing` aktif
- Validasi train-only untuk augmentasi ditambah assert eksplisit di notebook

Acceptance runbook di Colab T4:
- Jalankan cell 1 sampai 9 berurutan
- Jika batch awal terlalu besar, `auto_find_batch_size` otomatis menurunkan batch size
- Verifikasi metrik validasi/uji tampil lengkap dan model/zip tersimpan
