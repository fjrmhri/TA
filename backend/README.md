# Backend (Hugging Face Spaces Docker + FastAPI)

Backend ini menyediakan analisis deteksi hoaks per kalimat + NER opsional untuk teks multi paragraf.

## Endpoint

- `GET /health`  
  Response: `{"status":"ok"}`

- `POST /analyze`  
  Request JSON:
  ```json
  {
    "text": "string",
    "include_ner": true
  }
  ```
  Opsi tambahan (opsional):
  ```json
  {
    "confidence_orange_threshold": 0.7
  }
  ```

  Response JSON deterministik:
  ```json
  {
    "model": {
      "source": "hub",
      "model_id": "fjrmhri/Deteksi_Hoax_TA",
      "max_length": 256
    },
    "summary": {
      "num_paragraphs": 1,
      "num_sentences": 3,
      "hoax_sentences": 1,
      "fakta_sentences": 2,
      "low_conf_sentences": 0
    },
    "paragraphs": [
      {
        "paragraph_index": 0,
        "sentences": [
          {
            "sentence_index": 0,
            "text": "...",
            "label": "Hoaks",
            "prob_hoax": 0.97,
            "prob_fakta": 0.03,
            "confidence": 0.97,
            "color": "red"
          }
        ],
        "paragraph_summary": {
          "hoax_sentences": 1,
          "fakta_sentences": 0,
          "avg_confidence": 0.97,
          "max_hoax_prob": 0.97
        }
      }
    ],
    "ner": {
      "enabled": true,
      "model_id": "cahya/bert-base-indonesian-NER",
      "entities": [
        {
          "text": "Anies Baswedan",
          "entity_group": "PER",
          "score": 0.99
        }
      ]
    }
  }
  ```

## Aturan Warna

- `orange` jika `confidence < confidence_orange_threshold`
- selain itu `red` jika label `Hoaks`
- selain itu `green`

Jika client tidak mengirim `confidence_orange_threshold`, backend memakai default server-side dari `ORANGE_THRESHOLD` (default `0.65`).

## Environment Variables

- `HF_TOKEN` (opsional): token Hugging Face untuk rate limit/auth.
- `MODEL_ID` (default: `fjrmhri/Deteksi_Hoax_TA`)
- `NER_MODEL_ID` (default: `cahya/bert-base-indonesian-NER`)
- `FRONTEND_ORIGIN` (opsional): jika diisi, CORS dibatasi ke origin ini.
- `ORANGE_THRESHOLD` (default: `0.65`)
- `MAX_LENGTH` (default: `256`)
- `MAX_INPUT_CHARS` (default: `50000`)
- `BATCH_SIZE` (default: `16`)
- `LOCAL_MODEL_PATH` (opsional): override path fallback lokal.

## Model Loading

1. Coba load dari Hugging Face Hub (`MODEL_ID`) dengan `use_safetensors=True`.
2. Jika gagal, fallback ke lokal:
   - default path: `../indobert_hoax_ner_model_final/`
   - source response akan menjadi `"local"`.

Mapping label dipatok eksplisit:
- `0 -> Fakta`
- `1 -> Hoaks`

## Jalankan Lokal

```bash
cd backend
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker (HF Spaces)

```bash
cd backend
docker build -t hoax-backend .
docker run --rm -p 7860:7860 hoax-backend
```

Untuk Hugging Face Spaces Docker:
- gunakan `backend/Dockerfile`
- expose port `7860`
- set secret `HF_TOKEN` bila diperlukan.
