# PRA PROPOSAL TUGAS AKHIR/SKRIPSI

## JUDUL USULAN TUGAS AKHIR/SKRIPSI
**Deteksi Hoaks Berita Bahasa Indonesia Berbasis Fine-Tuning IndoBERT pada Level Kalimat dengan Integrasi Named Entity Recognition (NER) serta Deployment FastAPI dan Web**

---

## LATAR BELAKANG
Penyebaran hoaks pada konteks berita dan media sosial dapat memengaruhi pengambilan keputusan publik, menurunkan kepercayaan pada institusi, dan memicu disinformasi lanjutan. Kebutuhan utama pada penelitian ini adalah membangun sistem deteksi hoaks berbahasa Indonesia yang tidak hanya memberikan label “Hoaks/Fakta”, tetapi juga menampilkan hasil analisis pada level kalimat agar mudah ditinjau ulang oleh pengguna.

Repositori ini mengimplementasikan pipeline end-to-end yang mencakup pengumpulan data (scraping), pembentukan dataset terstruktur, pelatihan model klasifikasi berbasis Transformer (IndoBERT) untuk klasifikasi hoaks/fakta, integrasi modul NER untuk membantu interpretasi (menandai entitas penting seperti orang, organisasi, lokasi, tanggal, dan lainnya), serta deployment dalam bentuk API (FastAPI + Docker) dan antarmuka web (frontend static). Sistem produksi difokuskan pada pemrosesan teks multi-paragraf: paragraf dipecah, lalu setiap kalimat diklasifikasikan dan dipetakan kembali ke paragraf untuk ditampilkan sebagai highlight.

Pada tahap awal, data berasal dari beberapa portal berita untuk kelas non-hoaks dan TurnBackHoax untuk kelas hoaks. Tantangan utama pada skema seperti ini adalah risiko bias “label = sumber” (model belajar gaya sumber, bukan semantik hoaks). Karena itu, pipeline data dibangun ulang menjadi unit teks yang lebih mirip input produksi (klaim/kalimat/lead), dilakukan pembersihan boilerplate, dilakukan split berbasis grup, serta audit kebocoran (leakage audit) untuk memantau indikator-indikator kebocoran antar kelas.

Tujuan khusus penelitian ini adalah:
1) merancang arsitektur deteksi hoaks level kalimat yang stabil terhadap pergeseran domain,
2) menyediakan keluaran interpretatif melalui integrasi NER,
3) menyediakan implementasi siap pakai melalui backend API dan frontend web yang menampilkan highlight, confidence, dan informasi entitas.

---

## TINJAUAN PUSTAKA
Deteksi hoaks dapat dipandang sebagai masalah klasifikasi teks. Metode klasik seperti TF‑IDF + SVM/LogReg membutuhkan rekayasa fitur dan dapat sensitif terhadap pergeseran gaya bahasa. Metode deep learning (CNN/RNN/LSTM) dapat mempelajari fitur otomatis, tetapi sering memerlukan data besar dan masih kurang efisien dalam menangkap dependensi konteks panjang.

Transformer (mis. BERT) menjadi pendekatan dominan pada NLP modern karena memodelkan konteks dua arah. Fine‑tuning model pra-latih bahasa Indonesia seperti IndoBERT memungkinkan adaptasi cepat ke domain berita/cek fakta, dengan biaya pelatihan yang lebih efisien dibanding melatih dari nol.

Pada implementasi ini, deteksi dilakukan pada level kalimat (sentence-level) agar hasil dapat dipetakan kembali ke paragraf dan ditampilkan pada teks input. Untuk meningkatkan keterjelasan hasil, ditambahkan NER sebagai modul sekunder yang mengekstrak entitas penting pada kalimat. Informasi entitas dipakai untuk:
- membantu pengguna memahami “siapa/apa/di mana/kapan” yang sedang dibahas,
- membantu audit keluaran model,
- meningkatkan UX melalui underline entitas dan tooltip skor/jenis.

Roadmap pengembangan sistem yang relevan:
- baseline klasifikasi,
- penguatan robust (cleaning, group split, leakage audit),
- interpretabilitas (NER + visualisasi),
- deployment (API + web),
- evaluasi lanjutan (out-of-source, temporal holdout, error analysis).

---

## METODE
Tahapan penelitian dan implementasi pada repositori ini dirancang sebagai berikut:

1) **Pengumpulan data (scraping)**
   - Mengambil data dari beberapa sumber berita (non-hoaks) dan situs cek fakta (hoaks).
   - Output disimpan pada folder `scrape_output/` sebagai CSV per sumber beserta artefak resume (cache/progress/failed URLs). Bagian scraping hanya dijelaskan seperlunya karena fokus utama penelitian ini adalah arsitektur pemodelan dan deployment.

2) **Prapemrosesan dan pembersihan**
   - Normalisasi whitespace dan pembersihan boilerplate (mis. penanda sumber, “bagikan url”, tautan sosial, dan pola headline tertentu).
   - Pembentukan unit teks yang lebih menyerupai input produksi (kalimat klaim/lead) untuk mengurangi ketergantungan model pada gaya sumber.

3) **Pembentukan dataset terproses dan split**
   - Membangun dataset `data/processed/{train,val,test}.csv` dengan kolom inti: `text, label, source, url_hash, title_hash, unit_type`.
   - Split menggunakan stratified group split (group oleh `url_hash`) untuk mengurangi kemiripan lintas split.
   - Leakage audit berbasis marker disimpan ke `data/processed/leakage_audit.json` dan ringkasan split disimpan ke `data/processed/summary.json`.

4) **Pelatihan model klasifikasi (IndoBERT)**
   - Fine‑tuning `indolem/indobert-base-uncased` untuk klasifikasi biner Fakta/Hoaks.
   - Kontrol OOM: dynamic padding, fp16 (GPU), gradient checkpointing, auto batch size.
   - Evaluasi dengan metrik accuracy dan F1 (macro/weighted) serta confusion matrix.

5) **Inferensi produksi dan integrasi NER**
   - Inferensi dilakukan dengan split paragraf (blank line) → split kalimat (regex) → prediksi batch.
   - Skema hasil per kalimat mencakup: `label`, `prob_hoax`, `prob_fakta`, `confidence`, `color`.
   - Aturan warna: `orange` untuk confidence rendah, `red` untuk Hoaks, `green` untuk Fakta.
   - NER opsional untuk mengekstrak entitas, menampilkan underline di teks, dan tooltip skor/jenis pada UI.

6) **Deployment**
   - Backend: FastAPI + Docker (Hugging Face Spaces) dengan endpoint `/health` dan `/analyze`.
   - Frontend: static site (Vercel) yang memanggil backend dan menampilkan ringkasan, highlight per paragraf/kalimat, panel confidence, dan panel NER (collapsible).

Diagram alur ringkas:

```mermaid
flowchart TD
  A[Scraping] --> B[Scrape Output CSV + Cache/Progress]
  B --> C[Cleaning & Unit Text Builder]
  C --> D[Processed Dataset train/val/test + Leakage Audit]
  D --> E[Fine-tune IndoBERT Classifier]
  E --> F[Model Artifact + Calibration (opsional)]
  F --> G[Backend FastAPI /analyze]
  G --> H[Frontend: Highlight + Confidence + NER Tooltip]
```

---

## DAFTAR PUSTAKA
1. Repository model klasifikasi: fjrmhri/Deteksi_Hoax_TA (Hugging Face Hub).
2. Model NER Bahasa Indonesia: cahya/bert-base-indonesian-NER (Hugging Face Hub).
3. Library: Hugging Face Transformers (dokumentasi resmi).
4. Framework: FastAPI (dokumentasi resmi).
