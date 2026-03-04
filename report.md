# Report Implementasi dan Analisis Berita_Hoax

## 1) Ringkasan
Implementasi mencakup:
- forensik notebook Colab yang sudah dijalankan (kode + output),
- audit folder model `indobert_hoax_ner_model_final` termasuk checkpoint,
- audit ringkas 4 file CSV dataset,
- pembuatan backend `FastAPI` siap Hugging Face Spaces (Docker, port 7860),
- pembuatan frontend static siap Vercel,
- patch minimal dokumentasi lama pada angka yang mismatch.

Seluruh angka performa/akurasi di report ini berasal dari bukti output notebook atau `trainer_state.json`.

---

## 2) Forensik Notebook (Bukti Output)

Sumber utama:
- `Deteksi_Hoax_NER_Optimized.ipynb` output cell 5, 7, 8.

### 2.1 Audit dataset mentah (cell 5 output)
- Total baris gabungan: `37007`
- Distribusi label total: `{0: 24654, 1: 12353}`
- Per sumber:
  - `cnn`: 807 baris, `{0: 807}`
  - `detik`: 18711 baris, `{0: 18711}`
  - `kompas`: 5136 baris, `{0: 5136}`
  - `turnbackhoax`: 12353 baris, `{1: 12353}`
- Statistik panjang teks dicetak lengkap untuk `Clean Narasi`, `Narasi`, `isi_berita`, `judul`, `summary`.

### 2.2 Ringkasan prapemrosesan (cell 5 output)
- Jumlah baris awal: `37007`
- Dedup URL dihapus: `0`
- Dedup `judul+tanggal` dihapus: `42`
- Dedup `text+label` dihapus: `45`
- Total final setelah dedup: `36920`
- Distribusi label final: `{0: 24599, 1: 12321}`
- Strategi representasi text: `{'judul_summary': 36764, 'fallback': 201}`

### 2.3 Split dataset dan augmentasi (cell 5 output)
- `Latih: 34438 | Validasi: 5538 | Uji: 5538`  
  Catatan: angka latih di output ini sudah termasuk augmentasi train-only.
- Train sebelum augmentasi: `{0: 17219, 1: 8625}`
- Train sesudah augmentasi: `{1: 17219, 0: 17219}`
- Augmentasi:
  - total: `8594`
  - via NER replacement: `8479`
  - via fallback EDA: `115`

### 2.4 Metrik validasi dan uji (cell 7 output)
Validasi:
- `eval_loss: 7.998592082003597e-06`
- `eval_accuracy: 1.0`
- `eval_precision_macro: 1.0`
- `eval_recall_macro: 1.0`
- `eval_f1_macro: 1.0`
- `eval_precision_weighted: 1.0`
- `eval_recall_weighted: 1.0`
- `eval_f1_weighted: 1.0`
- confusion matrix: `TN=3690, FP=0, FN=0, TP=1848`

Uji:
- `test_loss: 7.969898433657363e-06`
- `test_accuracy: 1.0`
- `test_precision_macro: 1.0`
- `test_recall_macro: 1.0`
- `test_f1_macro: 1.0`
- `test_precision_weighted: 1.0`
- `test_recall_weighted: 1.0`
- `test_f1_weighted: 1.0`
- confusion matrix: `TN=3690, FP=0, FN=0, TP=1848`

### 2.5 Log training tambahan dari trainer state
Sumber:
- `indobert_hoax_ner_model_final/checkpoint-3231/trainer_state.json`

Fakta:
- `global_step=3231`, `epoch=3.0`, `num_train_epochs=3`
- `best_model_checkpoint=indobert_hoax_ner_model/checkpoint-1077`
- `best_metric=1.0`
- Eval history penting:
  - step 1077 (epoch 1.0): eval_loss `8.017749678401742e-06`, eval_accuracy `1.0`, eval_f1_macro `1.0`
  - step 2154 (epoch 2.0): eval_loss `4.6360419219126925e-06`, eval_accuracy `1.0`, eval_f1_macro `1.0`
  - step 3231 (epoch 3.0): eval_loss `2.9902155347372172e-06`, eval_accuracy `1.0`, eval_f1_macro `1.0`

### 2.6 Warning/error yang muncul di output notebook
Sumber:
- `Deteksi_Hoax_NER_Optimized.ipynb` output cell 5/7/8.

Temuan:
1. Unauthenticated HF request warning  
   Konteks: pemanggilan model/tokenizer dari Hub.
2. `Exception in thread Thread-auto_conversion ... JSONDecodeError`  
   Konteks: safetensors auto conversion background thread saat load model awal training.
3. `early stopping required metric_for_best_model, but did not find eval_f1_macro so early stopping is disabled`  
   Konteks: callback early stopping tidak aktif.
4. `You seem to be using the pipelines sequentially on GPU`  
   Konteks: penggunaan pipeline NER augmentasi secara sequential.
5. Load report UNEXPECTED/MISSING keys saat load model NER/IndoBERT base  
   Konteks: perbedaan arsitektur/task head.

### 2.7 Bukti kemampuan arsitektur integrasi (cell 8 output)
Arsitektur melakukan:
- klasifikasi per kalimat,
- ekstraksi entitas NER pada kalimat.

Bukti output:
- Device: `Device klasifikasi: cuda | Device NER pipeline: -1`
- Contoh kalimat menghasilkan label (`Fakta`) + confidence (`100.0%`) + daftar entitas (`PER`, `NOR`, `MON`, `PRD`).

---

## 3) Analisis Folder Model `indobert_hoax_ner_model_final`

### 3.1 Inventaris file utama + ukuran
Root:
- `config.json` (833 B)
- `model.safetensors` (442,262,496 B)
- `tokenizer.json` (736,506 B)
- `tokenizer_config.json` (350 B)
- `training_args.bin` (5,201 B)

Checkpoint `checkpoint-1077`:
- `config.json` (833 B)
- `model.safetensors` (442,262,496 B)
- `trainer_state.json` (1,938 B)
- `training_args.bin` (5,201 B)
- `optimizer.pt` (884,649,355 B)
- `scheduler.pt` (1,465 B)
- `scaler.pt` (1,383 B)
- `rng_state.pth` (14,645 B)
- `tokenizer.json` (736,506 B)
- `tokenizer_config.json` (350 B)

Checkpoint `checkpoint-3231`:
- `config.json` (833 B)
- `model.safetensors` (442,262,496 B)
- `trainer_state.json` (3,775 B)
- `training_args.bin` (5,201 B)
- `optimizer.pt` (884,649,355 B)
- `scheduler.pt` (1,465 B)
- `scaler.pt` (1,383 B)
- `rng_state.pth` (14,645 B)
- `tokenizer.json` (736,506 B)
- `tokenizer_config.json` (350 B)

### 3.2 Verifikasi label mapping
Sumber:
- `indobert_hoax_ner_model_final/config.json`
- kode notebook cell 8 (`prediksi_kelas == 1 -> "Hoaks", else "Fakta"`).

Temuan:
- `config.json` tidak menyimpan `id2label/label2id`.
- Mapping operasional inferensi dipatok dari logika notebook:
  - `0 -> Fakta`
  - `1 -> Hoaks`

### 3.3 Best checkpoint dan implikasi deployment
Sumber:
- `indobert_hoax_ner_model_final/checkpoint-3231/trainer_state.json`

Temuan:
- Best checkpoint tercatat: `checkpoint-1077`.
- Namun artefak root (`model.safetensors` + tokenizer + config) tersedia dan valid untuk inferensi standar.

Implikasi deployment:
- Default backend memakai model root dari Hub `fjrmhri/Deteksi_Hoax_TA` (`use_safetensors=True`).
- Jika gagal, fallback ke root lokal `indobert_hoax_ner_model_final/`.
- Metadata checkpoint tetap didokumentasikan untuk transparansi reproduksibilitas.

---

## 4) Audit Dataset CSV Nyata

Sumber:
- `data_hoaks_turnbackhoaks.csv`
- `data_nonhoaks_cnn.csv`
- `data_nonhoaks_detik.csv`
- `data_nonhoaks_kompas.csv`

### 4.1 Validasi kolom wajib
Kolom wajib tersedia di semua file:
- `url`, `judul`, `tanggal`, `Narasi`, `Clean Narasi`, `summary`, `hoax`

### 4.2 Distribusi label (hitung ulang file asli)
- `turnbackhoax`: 12,353 baris (`hoax=1`)
- `cnn`: 807 baris (`hoax=0`)
- `detik`: 18,711 baris (`hoax=0`)
- `kompas`: 5,136 baris (`hoax=0`)
- Total: `{1: 12353, 0: 24654}`

Hasil ini konsisten dengan audit mentah pada output notebook cell 5.

### 4.3 Strategi representasi text
Berdasarkan output notebook cell 5:
- `judul_summary` dominan (`36764`)
- fallback (`201`)

Ini konsisten dengan implementasi notebook yang memprioritaskan `judul + [SEP] + summary`.

---

## 5) Kesimpulan Kemampuan, Performa, dan Akurasi (berbasis bukti)

1. Kemampuan arsitektur:
   - Klasifikasi biner Fakta/Hoaks pada level kalimat.
   - NER opsional untuk entitas dalam teks.
2. Performa tercatat:
   - Validasi dan uji mencapai metrik `1.0` untuk accuracy, precision/recall/F1 (macro+weighted) pada output notebook.
3. Keputusan deployment:
   - Backend tetap mengikuti mapping label notebook (0=Fakta, 1=Hoaks), karena config model tidak menyimpan label mapping.

---

## 6) Risiko Generalisasi (Wajib) dan Rekomendasi Konkret

Metrik 1.0 sangat tinggi dan berisiko menandakan overfit, leakage laten, atau bias distribusi data.

### Risiko utama
1. Potensi kemiripan lintas split walaupun dedup sudah dilakukan (`url`, `judul+tanggal`, `text+label`).
2. Bias sumber:
   - semua hoaks dari TurnBackHoax,
   - semua non-hoaks dari CNN/Detik/Kompas.
   Model bisa belajar gaya sumber, bukan semata semantik hoaks.
3. Domain/style shift:
   - data baru dari sumber lain atau waktu berbeda bisa menurunkan performa nyata.

### Rencana evaluasi lanjutan (konkret)
1. Holdout berbasis sumber:
   - latih pada subset sumber, uji pada sumber yang tidak terlihat saat latih.
2. Holdout berbasis waktu:
   - split train/test menurut rentang tanggal (train masa lalu, test masa lebih baru).
3. K-fold cross-validation:
   - lakukan stratified k-fold dan laporkan mean + std metrik.
4. Out-of-source benchmark:
   - kumpulkan sampel hoaks/non-hoaks dari portal lain, evaluasi tanpa fine-tune ulang.
5. Uji adversarial:
   - paraphrase, typo/noise injection, penggantian entitas, dan perubahan urutan klausa.
6. Error analysis manual:
   - audit false positive/false negative per kategori topik.
7. Kalibrasi confidence:
   - evaluasi reliability diagram / ECE untuk memastikan confidence benar-benar terkalibrasi.

---

## 7) Ringkasan Deliverable Implementasi

### 7.1 Backend (`backend/`)
File:
- `backend/app.py`
- `backend/Dockerfile`
- `backend/requirements.txt`
- `backend/README.md`

Fitur:
- `GET /health -> {"status":"ok"}`
- `POST /analyze` sesuai schema request/response deterministik.
- Aturan warna:
  - confidence < threshold => `orange`
  - label `Hoaks` => `red`
  - selainnya => `green`
- Split paragraf via blank line.
- Split kalimat regex-based tanpa NLTK runtime download.
- Load model dari Hub (`use_safetensors=True`, dukung `HF_TOKEN`) + fallback lokal.
- Inferensi batched dengan `torch.inference_mode()`.
- NER default CPU dan opsional (`include_ner`).
- CORS configurable (`FRONTEND_ORIGIN` atau `*`).
- Guard input > 50k chars => HTTP 413.

### 7.2 Frontend (`frontend/`)
File:
- `frontend/index.html`
- `frontend/app.js`
- `frontend/styles.css`
- `frontend/README.md`

Fitur:
- textarea + tombol `Periksa`
- loader + error box
- legend warna
- highlight per paragraf/kalimat (`span.hl`)
- panel confidence berurutan sesuai output utama
- panel NER + toggle `Tampilkan NER`
- tombol `Reset` dan `Copy hasil`
- ringkasan count hoaks/fakta/low-confidence
- backend URL configurable (konstanta `BACKEND_URL` + input override di UI)

---

## 8) Catatan Sinkronisasi Dokumentasi

`documentation.md` dipatch minimal pada bagian mismatch:
- angka pemakaian strategi text (`judul_summary` dan `fallback`)
- klarifikasi ukuran split sebelum vs sesudah augmentasi train-only.

---

## 9) Referensi Bukti

1. Notebook:
   - `Deteksi_Hoax_NER_Optimized.ipynb` (output cell 5, 7, 8)
2. Model artifacts:
   - `indobert_hoax_ner_model_final/config.json`
   - `indobert_hoax_ner_model_final/checkpoint-1077/trainer_state.json`
   - `indobert_hoax_ner_model_final/checkpoint-3231/trainer_state.json`
3. Dataset:
   - `data_hoaks_turnbackhoaks.csv`
   - `data_nonhoaks_cnn.csv`
   - `data_nonhoaks_detik.csv`
   - `data_nonhoaks_kompas.csv`
