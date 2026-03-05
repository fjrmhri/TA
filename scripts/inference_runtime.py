#!/usr/bin/env python
"""Reusable model loading + inference runtime for local scripts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PARAGRAPH_SPLIT_RE = re.compile(r"(?:\r?\n){2,}")
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+(?:[\"”’)\]]+)?)|[^.!?]+$")

HOAX_LABEL_TOKENS = ("hoaks", "hoax", "fake", "false", "disinfo", "misinfo")
FAKTA_LABEL_TOKENS = ("fakta", "fact", "true", "valid", "nonhoax", "non-hoax")

INFERENCE_CLEAN_PATTERNS = [
    (re.compile(r"(?i)\buncategorized\b"), " "),
    (re.compile(r"(?i)\b(?:facebook|twitter|x\.com|tiktok|youtube|instagram|whatsapp)\b"), " "),
    (re.compile(r"(?i)\bakun\b[^.!?\n]{0,140}\bunggah\b[^.!?\n]*"), " "),
    (re.compile(r"(?i)\bbaca juga:\s*[^.!?\n]*"), " "),
    (re.compile(r"(?i)\blihat juga:\s*[^.!?\n]*"), " "),
    (re.compile(r"(?i)\badvertisement\b\s*scroll to continue with content"), " "),
    (re.compile(r"(?i)\bturnbackhoax(?:s)?\b"), " "),
    (re.compile(r"(?i)\bcnn indonesia\b"), " "),
    (re.compile(r"(?i)\bkompas\.com\b"), " "),
    (re.compile(r"(?i)\bdetik(?:com)?\b"), " "),
    (re.compile(r"(?i)\bmafindo\b"), " "),
    (re.compile(r"(?i)\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}\b"), " "),
    (re.compile(r"(?i)\b\d{1,2}\s+\d{1,2}\s+\d{4}\b"), " "),
    (re.compile(r"(?i)\b\d{1,2}:\d{2}\s*wib\b"), " "),
]


def normalize_unit_text(text: str) -> str:
    cleaned = str(text)
    for pattern, replacement in INFERENCE_CLEAN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" -:;,.")
    return cleaned


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "", str(label).strip().lower())


def round6(value: float) -> float:
    return round(float(value), 6)


class InferenceRuntime:
    def __init__(
        self,
        *,
        model_id: str,
        local_model_path: Path,
        max_length: int = 256,
        batch_size: int = 16,
        orange_threshold: float = 0.65,
        hoax_threshold: float = 0.5,
        calibration_path: Path | None = None,
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.local_model_path = Path(local_model_path)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.orange_threshold = float(orange_threshold)
        self.hoax_threshold = float(hoax_threshold)
        self.calibration_path = (
            Path(calibration_path)
            if calibration_path is not None
            else self.local_model_path / "calibration.json"
        )
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = None
        self.tokenizer = None
        self.model_source = "unknown"
        self.calibration_loaded = False

        self.id2label: Dict[int, str] = {0: "Fakta", 1: "Hoaks"}
        self.label2id: Dict[str, int] = {"Fakta": 0, "Hoaks": 1}
        self.num_labels = 2
        self.fakta_class_id = 0
        self.hoax_class_id = 1

    def _hf_auth_kwargs(self) -> Dict:
        kwargs: Dict = {}
        if self.hf_token:
            kwargs["token"] = self.hf_token
        return kwargs

    def _resolve_label_maps(self, model_config) -> None:
        raw_id2label = getattr(model_config, "id2label", None)
        if isinstance(raw_id2label, dict) and raw_id2label:
            parsed: Dict[int, str] = {}
            for key, value in raw_id2label.items():
                try:
                    parsed[int(key)] = str(value)
                except Exception:
                    continue
            if parsed:
                self.id2label = dict(sorted(parsed.items(), key=lambda item: item[0]))
            else:
                self.id2label = {0: "Fakta", 1: "Hoaks"}
        else:
            self.id2label = {0: "Fakta", 1: "Hoaks"}

        self.label2id = {name: idx for idx, name in self.id2label.items()}
        self.num_labels = len(self.id2label)

        hoax_candidates = []
        fakta_candidates = []
        for idx, label_name in self.id2label.items():
            normalized = normalize_label(label_name)
            if any(token in normalized for token in HOAX_LABEL_TOKENS):
                hoax_candidates.append(idx)
            if any(token in normalized for token in FAKTA_LABEL_TOKENS):
                fakta_candidates.append(idx)

        self.hoax_class_id = hoax_candidates[0] if hoax_candidates else (1 if self.num_labels > 1 else 0)
        if fakta_candidates:
            self.fakta_class_id = fakta_candidates[0]
        else:
            self.fakta_class_id = 0 if self.hoax_class_id != 0 else (1 if self.num_labels > 1 else 0)

    def _load_calibration(self) -> None:
        self.calibration_loaded = False
        if not self.calibration_path.exists():
            return
        try:
            payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            candidate = payload.get("best_threshold", payload.get("threshold"))
            if candidate is None:
                return
            value = float(candidate)
            if 0.0 <= value <= 1.0:
                self.hoax_threshold = value
                self.calibration_loaded = True
        except Exception:
            self.calibration_loaded = False

    def load(self) -> None:
        auth_kwargs = self._hf_auth_kwargs()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **auth_kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                use_safetensors=True,
                low_cpu_mem_usage=True,
                **auth_kwargs,
            )
            self.model_source = "hub"
        except Exception as hub_exc:
            if not self.local_model_path.exists():
                raise RuntimeError(
                    f"Model Hub gagal dan fallback lokal tidak ditemukan: {self.local_model_path}"
                ) from hub_exc
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.local_model_path),
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.local_model_path),
                local_files_only=True,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            self.model_source = "local"

        self.model.to(self.device)
        self.model.eval()
        self._resolve_label_maps(self.model.config)
        self._load_calibration()

    def split_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in PARAGRAPH_SPLIT_RE.split(str(text).strip()) if p.strip()]
        if paragraphs:
            return paragraphs
        raw = str(text).strip()
        return [raw] if raw else []

    def split_sentences(self, paragraph: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", str(paragraph)).strip()
        if not normalized:
            return []
        sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(normalized)]
        sentences = [normalize_unit_text(sent) for sent in sentences]
        sentences = [sent for sent in sentences if sent]
        if sentences:
            return sentences
        fallback = normalize_unit_text(normalized)
        return [fallback] if fallback else []

    def predict_batch(self, sentences: List[str]) -> List[Dict]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model belum di-load. Panggil load() dulu.")

        normalized_sentences = [normalize_unit_text(s) for s in sentences]
        normalized_sentences = [s for s in normalized_sentences if s]
        if not normalized_sentences:
            return []

        rows: List[Dict] = []
        with torch.inference_mode():
            for start_idx in range(0, len(normalized_sentences), self.batch_size):
                batch = normalized_sentences[start_idx : start_idx + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu()
                pred_ids = probs.argmax(dim=-1).tolist()

                for text, argmax_id, prob_tensor in zip(batch, pred_ids, probs):
                    values = prob_tensor.tolist()
                    prob_hoax = values[self.hoax_class_id] if self.hoax_class_id < len(values) else 0.0
                    prob_fakta = values[self.fakta_class_id] if self.fakta_class_id < len(values) else 0.0
                    pred_id = 1 if prob_hoax >= self.hoax_threshold else 0
                    label = "Hoaks" if pred_id == 1 else "Fakta"
                    confidence = max(prob_hoax, prob_fakta)
                    color = (
                        "orange"
                        if confidence < self.orange_threshold
                        else ("red" if label == "Hoaks" else "green")
                    )
                    rows.append(
                        {
                            "text": text,
                            "label": label,
                            "pred_id": int(pred_id),
                            "argmax_id": int(argmax_id),
                            "prob_hoax": round6(prob_hoax),
                            "prob_fakta": round6(prob_fakta),
                            "confidence": round6(confidence),
                            "color": color,
                        }
                    )
        return rows

