import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

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

LOCAL_MODEL_PATH = Path(
    os.getenv(
        "LOCAL_MODEL_PATH",
        str((Path(__file__).resolve().parents[1] / "indobert_hoax_ner_model_final")),
    )
)

PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n){2,}")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?:[\"”’)\]]+)?)|[^.!?]+$")

# Mapping dipatok eksplisit mengikuti notebook:
# kelas 0 -> Fakta, kelas 1 -> Hoaks
LABEL_MAP = {0: "Fakta", 1: "Hoaks"}

CLASSIFIER_TOKENIZER = None
CLASSIFIER_MODEL = None
NER_PIPELINE = None
MODEL_SOURCE = "hub"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Teks input multi paragraf.")
    include_ner: bool = Field(True, description="Jalankan NER jika true.")
    confidence_orange_threshold: Optional[float] = Field(
        None, description="Threshold untuk warna oranye."
    )


app = FastAPI(title="Hoax Sentence Analyzer API", version="1.0.0")

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


def _float(value: float) -> float:
    return round(float(value), 6)


def _hf_auth_kwargs() -> Dict:
    kwargs: Dict = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN
    return kwargs


def _load_classifier() -> None:
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, MODEL_SOURCE

    auth_kwargs = _hf_auth_kwargs()
    try:
        LOGGER.info("Loading classifier from Hub: %s", MODEL_ID)
        CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, **auth_kwargs)
        CLASSIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            use_safetensors=True,
            **auth_kwargs,
        )
        MODEL_SOURCE = "hub"
    except Exception as hub_exc:
        LOGGER.warning("Hub load failed: %s", hub_exc)
        if not LOCAL_MODEL_PATH.exists():
            raise RuntimeError(
                f"Model Hub gagal dan fallback lokal tidak ditemukan: {LOCAL_MODEL_PATH}"
            ) from hub_exc

        LOGGER.info("Fallback loading local model from: %s", LOCAL_MODEL_PATH)
        CLASSIFIER_TOKENIZER = AutoTokenizer.from_pretrained(
            str(LOCAL_MODEL_PATH),
            local_files_only=True,
        )
        CLASSIFIER_MODEL = AutoModelForSequenceClassification.from_pretrained(
            str(LOCAL_MODEL_PATH),
            local_files_only=True,
            use_safetensors=True,
        )
        MODEL_SOURCE = "local"

    CLASSIFIER_MODEL.to(DEVICE)
    CLASSIFIER_MODEL.eval()
    LOGGER.info("Classifier ready on device=%s source=%s", DEVICE, MODEL_SOURCE)


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

    sentences = [m.group(0).strip() for m in SENTENCE_RE.finditer(normalized)]
    sentences = [s for s in sentences if s]
    if sentences:
        return sentences
    return [normalized]


def _pick_color(label: str, confidence: float, orange_threshold: float) -> str:
    if confidence < orange_threshold:
        return "orange"
    if label == "Hoaks":
        return "red"
    return "green"


def _classify_sentences(sentences: List[str], orange_threshold: float) -> List[Dict]:
    if not sentences:
        return []

    results: List[Dict] = []
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
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

            logits = CLASSIFIER_MODEL(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

            for text, p in zip(batch, probs):
                prob_fakta = float(p[0])
                prob_hoax = float(p[1])
                pred_id = 1 if prob_hoax >= prob_fakta else 0
                label = LABEL_MAP[pred_id]
                confidence = max(prob_hoax, prob_fakta)
                color = _pick_color(label, confidence, orange_threshold)

                results.append(
                    {
                        "text": text,
                        "label": label,
                        "prob_hoax": _float(prob_hoax),
                        "prob_fakta": _float(prob_fakta),
                        "confidence": _float(confidence),
                        "color": color,
                    }
                )
    return results


def _extract_entities(text: str) -> List[Dict]:
    try:
        ner = _get_ner_pipeline()
        raw_entities = ner(text)
    except Exception as exc:
        raise RuntimeError(f"Gagal menjalankan NER: {exc}") from exc

    entities: List[Dict] = []
    seen = set()
    for ent in raw_entities:
        ent_text = str(ent.get("word", "")).strip()
        entity_group = str(ent.get("entity_group", "")).strip()
        score = float(ent.get("score", 0.0))

        key = (ent_text.lower(), entity_group)
        if not ent_text or not entity_group or key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "text": ent_text,
                "entity_group": entity_group,
                "score": _float(score),
            }
        )
    return entities


@app.on_event("startup")
def startup_event() -> None:
    _load_classifier()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest) -> Dict:
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

    threshold = (
        ORANGE_THRESHOLD
        if payload.confidence_orange_threshold is None
        else payload.confidence_orange_threshold
    )
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="confidence_orange_threshold harus berada pada rentang 0 sampai 1.",
        )

    paragraphs_raw = _split_paragraphs(text)
    paragraph_responses = []

    total_sentences = 0
    total_hoax = 0
    total_fakta = 0
    total_low_conf = 0

    for p_idx, paragraph_text in enumerate(paragraphs_raw):
        sentences = _split_sentences(paragraph_text)
        classified = _classify_sentences(sentences, threshold)

        sentence_items = []
        paragraph_hoax = 0
        paragraph_fakta = 0
        paragraph_low = 0
        conf_values: List[float] = []
        hoax_probs: List[float] = []

        for s_idx, item in enumerate(classified):
            if item["label"] == "Hoaks":
                paragraph_hoax += 1
            else:
                paragraph_fakta += 1
            if item["confidence"] < threshold:
                paragraph_low += 1

            conf_values.append(item["confidence"])
            hoax_probs.append(item["prob_hoax"])

            sentence_items.append(
                {
                    "sentence_index": s_idx,
                    "text": item["text"],
                    "label": item["label"],
                    "prob_hoax": item["prob_hoax"],
                    "prob_fakta": item["prob_fakta"],
                    "confidence": item["confidence"],
                    "color": item["color"],
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
                "paragraph_index": p_idx,
                "sentences": sentence_items,
                "paragraph_summary": paragraph_summary,
            }
        )

        total_sentences += len(sentence_items)
        total_hoax += paragraph_hoax
        total_fakta += paragraph_fakta
        total_low_conf += paragraph_low

    entities = []
    if payload.include_ner:
        try:
            entities = _extract_entities(text)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    response = {
        "model": {
            "source": MODEL_SOURCE,
            "model_id": MODEL_ID,
            "max_length": MAX_LENGTH,
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
    return response
