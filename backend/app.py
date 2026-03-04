import logging
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("hoax-backend")

MODEL_ID = os.getenv("MODEL_ID", "fjrmhri/Deteksi_Hoax_TA")
NER_MODEL_ID = os.getenv("NER_MODEL_ID", "cahya/bert-base-indonesian-NER")
HF_TOKEN = os.getenv("HF_TOKEN")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
ORANGE_THRESHOLD = float(os.getenv("ORANGE_THRESHOLD", "0.65"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "50000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))

ROOT_DIR = Path(__file__).resolve().parents[1]
LOCAL_MODEL_PATH = Path(
    os.getenv("LOCAL_MODEL_PATH", str(ROOT_DIR / "indobert_hoax_ner_model_final"))
)
PROCESSED_TEST_PATH = Path(
    os.getenv("PROCESSED_TEST_PATH", str(ROOT_DIR / "data" / "processed" / "test.csv"))
)

PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n){2,}")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?:[\"”’)\]]+)?)|[^.!?]+$")
HOAX_LABEL_TOKENS = ("hoaks", "hoax", "fake", "false", "disinfo", "misinfo")
FAKTA_LABEL_TOKENS = ("fakta", "fact", "true", "valid", "nonhoax", "non-hoax")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Teks input multi paragraf.")
    include_ner: bool = Field(True, description="Jalankan NER jika true.")


app = FastAPI(title="Hoax Sentence Analyzer API", version="2.0.0")

if FRONTEND_ORIGIN:
    allowed_origins = [FRONTEND_ORIGIN]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSIFIER_TOKENIZER = None
CLASSIFIER_MODEL = None
NER_PIPELINE = None
MODEL_SOURCE = "unknown"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ID2LABEL: Dict[int, str] = {0: "Fakta", 1: "Hoaks"}
LABEL2ID: Dict[str, int] = {"Fakta": 0, "Hoaks": 1}
NUM_LABELS = 2
FAKTA_CLASS_ID = 0
HOAX_CLASS_ID = 1
STARTUP_SANITY: Dict[str, object] = {
    "checked": False,
    "status": "not_run",
    "message": "startup sanity belum dijalankan",
}


def _float(value: float) -> float:
    return round(float(value), 6)


def _hf_auth_kwargs() -> Dict:
    kwargs: Dict = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return kwargs


def _normalize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "", str(name).strip().lower())


def _resolve_label_maps(model_config) -> None:
    global ID2LABEL, LABEL2ID, NUM_LABELS, FAKTA_CLASS_ID, HOAX_CLASS_ID

    raw_id2label = getattr(model_config, "id2label", None)
    if isinstance(raw_id2label, dict) and raw_id2label:
        parsed = {}
        for key, value in raw_id2label.items():
            try:
                parsed[int(key)] = str(value)
            except Exception:
                continue
        if parsed:
            ID2LABEL = dict(sorted(parsed.items(), key=lambda item: item[0]))
        else:
            ID2LABEL = {0: "Fakta", 1: "Hoaks"}
    else:
        ID2LABEL = {0: "Fakta", 1: "Hoaks"}

    LABEL2ID = {name: idx for idx, name in ID2LABEL.items()}
    NUM_LABELS = len(ID2LABEL)

    hoax_candidates = []
    fakta_candidates = []
    for idx, label_name in ID2LABEL.items():
        normalized = _normalize_label(label_name)
        if any(token in normalized for token in HOAX_LABEL_TOKENS):
            hoax_candidates.append(idx)
        if any(token in normalized for token in FAKTA_LABEL_TOKENS):
            fakta_candidates.append(idx)

    HOAX_CLASS_ID = hoax_candidates[0] if hoax_candidates else (1 if NUM_LABELS > 1 else 0)
    if fakta_candidates:
        FAKTA_CLASS_ID = fakta_candidates[0]
    else:
        FAKTA_CLASS_ID = 0 if HOAX_CLASS_ID != 0 else (1 if NUM_LABELS > 1 else 0)


def _predict_batch(sentences: List[str]) -> List[Dict]:
    if not sentences:
        return []

    rows: List[Dict] = []
    with torch.inference_mode():
        for start_idx in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[start_idx : start_idx + BATCH_SIZE]
            encoded = CLASSIFIER_TOKENIZER(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = CLASSIFIER_MODEL(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu()
            pred_ids = probs.argmax(dim=-1).tolist()

            for text, pred_id, prob_tensor in zip(batch, pred_ids, probs):
                values = prob_tensor.tolist()
                prob_hoax = values[HOAX_CLASS_ID] if HOAX_CLASS_ID < len(values) else 0.0
                prob_fakta = values[FAKTA_CLASS_ID] if FAKTA_CLASS_ID < len(values) else 0.0
                label_name = ID2LABEL.get(int(pred_id), str(pred_id))
                is_hoax = _normalize_label(label_name) in {"hoaks", "hoax", "fake", "false"}
                label = "Hoaks" if is_hoax else "Fakta"
                confidence = max(prob_hoax, prob_fakta)
                color = "orange" if confidence < ORANGE_THRESHOLD else ("red" if label == "Hoaks" else "green")
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "pred_id": int(pred_id),
                        "prob_hoax": _float(prob_hoax),
                        "prob_fakta": _float(prob_fakta),
                        "confidence": _float(confidence),
                        "color": color,
                    }
                )
    return rows


def _load_classifier() -> None:
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, MODEL_SOURCE

    auth_kwargs = _hf_auth_kwargs()
    try:
        LOGGER.info("Loading classifier from Hub: %s", MODEL_ID)
        CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, **auth_kwargs)
        CLASSIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            **auth_kwargs,
        )
        MODEL_SOURCE = "hub"
    except Exception as hub_exc:
        LOGGER.warning("Hub load failed: %s", hub_exc)
        if not LOCAL_MODEL_PATH.exists():
            raise RuntimeError(
                f"Model Hub gagal dan fallback lokal tidak ditemukan: {LOCAL_MODEL_PATH}"
            ) from hub_exc
        LOGGER.info("Loading classifier from local fallback: %s", LOCAL_MODEL_PATH)
        CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(
            str(LOCAL_MODEL_PATH),
            local_files_only=True,
        )
        CLASSIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            str(LOCAL_MODEL_PATH),
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        MODEL_SOURCE = "local"

    CLASSIFIER_MODEL.to(DEVICE)
    CLASSIFIER_MODEL.eval()
    _resolve_label_maps(CLASSIFIER_MODEL.config)
    LOGGER.info(
        "Classifier ready | source=%s | device=%s | num_labels=%s | id2label=%s | label2id=%s",
        MODEL_SOURCE,
        DEVICE,
        NUM_LABELS,
        ID2LABEL,
        LABEL2ID,
    )


def _get_ner_pipeline():
    global NER_PIPELINE
    if NER_PIPELINE is None:
        kwargs = _hf_auth_kwargs()
        NER_PIPELINE = pipeline(
            "ner",
            model=NER_MODEL_ID,
            aggregation_strategy="simple",
            device=-1,
            **kwargs,
        )
    return NER_PIPELINE


def _split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()]
    if paragraphs:
        return paragraphs
    stripped = text.strip()
    return [stripped] if stripped else []


def _split_sentences(paragraph: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", paragraph).strip()
    if not normalized:
        return []
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(normalized)]
    sentences = [sent for sent in sentences if sent]
    return sentences if sentences else [normalized]


def _extract_entities(text: str) -> List[Dict]:
    try:
        ner = _get_ner_pipeline()
        raw_entities = ner(text)
    except Exception as exc:
        raise RuntimeError(f"Gagal menjalankan NER: {exc}") from exc

    entities: List[Dict] = []
    seen = set()
    for entity in raw_entities:
        ent_text = str(entity.get("word", "")).strip()
        ent_group = str(entity.get("entity_group", "")).strip()
        score = float(entity.get("score", 0.0))
        key = (ent_text.lower(), ent_group)
        if not ent_text or not ent_group or key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "text": ent_text,
                "entity_group": ent_group,
                "score": _float(score),
            }
        )
    return entities


def _run_startup_sanity() -> None:
    global STARTUP_SANITY
    STARTUP_SANITY = {
        "checked": True,
        "status": "ok",
        "message": "startup sanity ok",
    }

    if not PROCESSED_TEST_PATH.exists():
        STARTUP_SANITY = {
            "checked": True,
            "status": "warning",
            "message": f"processed test tidak ditemukan: {PROCESSED_TEST_PATH}",
        }
        LOGGER.warning(STARTUP_SANITY["message"])
        return

    try:
        df_test = pd.read_csv(PROCESSED_TEST_PATH)
        if not {"text", "label"}.issubset(df_test.columns):
            STARTUP_SANITY = {
                "checked": True,
                "status": "warning",
                "message": f"kolom text/label tidak lengkap pada {PROCESSED_TEST_PATH}",
            }
            LOGGER.warning(STARTUP_SANITY["message"])
            return

        samples = []
        for label in [0, 1]:
            subset = df_test[df_test["label"] == label]
            if not subset.empty:
                samples.append(subset.iloc[0]["text"])
        if len(samples) < 2:
            samples = df_test["text"].astype(str).head(2).tolist()

        preds = _predict_batch([str(s) for s in samples if str(s).strip()])
        pred_labels = {row["label"] for row in preds}
        if len(pred_labels) < 2:
            STARTUP_SANITY = {
                "checked": True,
                "status": "warning",
                "message": "startup sanity: prediksi sampel hanya satu kelas (potensi collapse).",
                "pred_labels": sorted(pred_labels),
            }
            LOGGER.warning(STARTUP_SANITY["message"])
            return

        STARTUP_SANITY = {
            "checked": True,
            "status": "ok",
            "message": "startup sanity: kedua kelas muncul pada sampel processed test.",
            "pred_labels": sorted(pred_labels),
        }
        LOGGER.info(STARTUP_SANITY["message"])
    except Exception as exc:
        STARTUP_SANITY = {
            "checked": True,
            "status": "warning",
            "message": f"startup sanity gagal dijalankan: {exc}",
        }
        LOGGER.warning(STARTUP_SANITY["message"])


@app.on_event("startup")
def startup_event() -> None:
    _load_classifier()
    _run_startup_sanity()


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "model_source": MODEL_SOURCE,
        "model_id": MODEL_ID,
        "num_labels": NUM_LABELS,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "startup_sanity": STARTUP_SANITY,
    }


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> Dict[str, object]:
    if CLASSIFIER_MODEL is None or CLASSIFIER_TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model classifier belum siap.")

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Field 'text' tidak boleh kosong.")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Input terlalu panjang ({len(text)} chars). Maksimum {MAX_INPUT_CHARS} chars.",
        )

    paragraphs_raw = _split_paragraphs(text)
    paragraph_responses = []
    total_sentences = 0
    total_hoax = 0
    total_fakta = 0
    total_low_conf = 0

    for paragraph_idx, paragraph_text in enumerate(paragraphs_raw):
        sentences = _split_sentences(paragraph_text)
        classified = _predict_batch(sentences)

        sentence_items = []
        paragraph_hoax = 0
        paragraph_fakta = 0
        paragraph_low = 0
        conf_values: List[float] = []
        hoax_probs: List[float] = []

        for sentence_idx, row in enumerate(classified):
            if row["label"] == "Hoaks":
                paragraph_hoax += 1
            else:
                paragraph_fakta += 1
            if row["confidence"] < ORANGE_THRESHOLD:
                paragraph_low += 1

            conf_values.append(row["confidence"])
            hoax_probs.append(row["prob_hoax"])

            sentence_items.append(
                {
                    "sentence_index": sentence_idx,
                    "text": row["text"],
                    "label": row["label"],
                    "prob_hoax": row["prob_hoax"],
                    "prob_fakta": row["prob_fakta"],
                    "confidence": row["confidence"],
                    "color": row["color"],
                }
            )

        paragraph_summary = {
            "hoax_sentences": paragraph_hoax,
            "fakta_sentences": paragraph_fakta,
            "avg_confidence": _float(sum(conf_values) / len(conf_values)) if conf_values else 0.0,
            "max_hoax_prob": _float(max(hoax_probs)) if hoax_probs else 0.0,
        }
        paragraph_responses.append(
            {
                "paragraph_index": paragraph_idx,
                "sentences": sentence_items,
                "paragraph_summary": paragraph_summary,
            }
        )

        total_sentences += len(sentence_items)
        total_hoax += paragraph_hoax
        total_fakta += paragraph_fakta
        total_low_conf += paragraph_low

    entities: List[Dict] = []
    if payload.include_ner:
        try:
            entities = _extract_entities(text)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "model": {
            "source": MODEL_SOURCE,
            "model_id": MODEL_ID,
            "max_length": MAX_LENGTH,
            "num_labels": NUM_LABELS,
        },
        "summary": {
            "num_paragraphs": len(paragraph_responses),
            "num_sentences": total_sentences,
            "hoax_sentences": total_hoax,
            "fakta_sentences": total_fakta,
            "low_conf_sentences": total_low_conf,
        },
        "paragraphs": paragraph_responses,
        "ner": {
            "enabled": payload.include_ner,
            "model_id": NER_MODEL_ID,
            "entities": entities,
        },
    }
