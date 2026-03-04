# Backend (FastAPI + Hugging Face Spaces Docker)

Backend ini menyediakan analisis hoaks per kalimat dengan preprocessing konsisten:
- split paragraf (`blank line`)
- split kalimat (regex ringan)
- klasifikasi per kalimat + ringkasan per paragraf
- NER opsional (default CPU, `cahya/bert-base-indonesian-NER`)

## Endpoint

### `GET /health`
Response contoh:
```json
{
  "status": "ok",
  "model_source": "hub",
  "model_id": "fjrmhri/Deteksi_Hoax_TA",
  "num_labels": 2,
  "id2label": {"0": "Fakta", "1": "Hoaks"},
  "label2id": {"Fakta": 0, "Hoaks": 1},
  "startup_sanity": {
    "checked": true,
    "status": "ok",
    "message": "startup sanity: kedua kelas muncul pada sampel processed test."
  }
}
```

### `POST /analyze`
Request:
```json
{
  "text": "string",
  "include_ner": true
}
```

Response utama:
- `paragraphs[].sentences[]` berisi:
  - `label`
  - `prob_hoax`
  - `prob_fakta`
  - `confidence`
  - `color`

Warna:
- `orange` jika `confidence < 0.65` (server-side default)
- `red` jika `label = Hoaks`
- `green` selain itu

## Model loading

Urutan load model:
1. Hub: `MODEL_ID` (default `fjrmhri/Deteksi_Hoax_TA`)
2. Fallback lokal: `indobert_hoax_ner_model_final/`

Saat startup backend akan log:
- model source (`hub/local`)
- `num_labels`
- `id2label` / `label2id`

## Startup sanity test

Saat startup backend mencoba baca:
- `data/processed/test.csv` (override via `PROCESSED_TEST_PATH`)

Backend menguji 2 sampel kecil untuk memastikan prediksi tidak hanya satu kelas.
Jika kolaps satu kelas, backend tetap hidup tapi mengembalikan `startup_sanity.status = warning`.

## Environment variables

- `HF_TOKEN` (opsional)
- `MODEL_ID` (default `fjrmhri/Deteksi_Hoax_TA`)
- `NER_MODEL_ID` (default `cahya/bert-base-indonesian-NER`)
- `LOCAL_MODEL_PATH` (opsional path fallback lokal)
- `PROCESSED_TEST_PATH` (opsional path sanity test)
- `FRONTEND_ORIGIN` (opsional CORS)
- `ORANGE_THRESHOLD` (default `0.65`)
- `MAX_LENGTH` (default `256`)
- `MAX_INPUT_CHARS` (default `50000`)
- `BATCH_SIZE` (default `16`)

## Run local

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker (HF Spaces)

```bash
cd backend
docker build -t hoax-backend .
docker run --rm -p 7860:7860 hoax-backend
```

`Dockerfile` sudah expose `7860` dan menjalankan:
`uvicorn app:app --host 0.0.0.0 --port 7860`
