#!/usr/bin/env python
"""Evaluate classifier on processed test split and store metrics JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference_runtime import InferenceRuntime


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on processed test set.")
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "test.csv",
        help="Path to processed test CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts" / "metrics_local_latest.json",
        help="Output JSON file for metrics.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit number of test rows for faster evaluation.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("MODEL_ID", "fjrmhri/Deteksi_Hoax_TA"),
        help="Hugging Face model id (hub).",
    )
    parser.add_argument(
        "--local-model-path",
        type=Path,
        default=ROOT / "indobert_hoax_ner_model_final",
        help="Fallback local model path.",
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=None,
        help="Optional path ke calibration.json. Default: <local-model>/calibration.json",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("MAX_LENGTH", "256")),
        help="Tokenizer max_length.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "16")),
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, contoh: cpu atau cuda.",
    )
    parser.add_argument(
        "--hoax-threshold",
        type=float,
        default=None,
        help="Override threshold hoax. Jika kosong, gunakan calibration/fallback.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    test_csv = args.test_csv.resolve()
    output_json = args.output_json.resolve()

    if not test_csv.exists():
        print(f"MISSING: {test_csv}")
        return 1

    runtime = InferenceRuntime(
        model_id=args.model_id,
        local_model_path=args.local_model_path.resolve(),
        max_length=args.max_length,
        batch_size=args.batch_size,
        calibration_path=args.calibration_path.resolve() if args.calibration_path else None,
        device=args.device,
    )
    try:
        runtime.load()
    except Exception as exc:
        print(f"Gagal load classifier: {exc}")
        return 1

    if args.hoax_threshold is not None:
        runtime.hoax_threshold = float(args.hoax_threshold)
        runtime.calibration_loaded = False

    df = pd.read_csv(test_csv)
    missing_cols = [col for col in ["text", "label"] if col not in df.columns]
    if missing_cols:
        print(f"Kolom wajib tidak ditemukan di {test_csv}: {missing_cols}")
        return 1

    if args.max_samples and len(df) > args.max_samples:
        parts = []
        total_rows = len(df)
        for label, group in df.groupby("label"):
            frac = len(group) / total_rows if total_rows else 0.0
            n_take = max(1, int(round(args.max_samples * frac)))
            n_take = min(n_take, len(group))
            parts.append(group.sample(n=n_take, random_state=42))
        df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()

    predictions = runtime.predict_batch(texts)
    # backend `pred_id` sudah canonical: 0=Fakta, 1=Hoaks (threshold-based decision)
    raw_pred_ids = [int(row["pred_id"]) for row in predictions]
    y_pred = raw_pred_ids

    if len(y_pred) != len(y_true):
        print(f"Jumlah prediksi ({len(y_pred)}) tidak sama dengan label ({len(y_true)})")
        return 1

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_each, r_each, f_each, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pred_dist = pd.Series(y_pred).value_counts().sort_index().to_dict()
    hoax_precision = float(p_each[1]) if len(p_each) > 1 else 0.0
    hoax_recall = float(r_each[1]) if len(r_each) > 1 else 0.0
    challenge_ready = bool(int(cm[1, 1]) > 0 and hoax_recall > 0.0 and len(pred_dist) > 1)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "test_csv": str(test_csv),
        "rows": int(len(df)),
        "model_source": runtime.model_source,
        "model_id": runtime.model_id,
        "num_labels": runtime.num_labels,
        "fakta_class_id": int(runtime.fakta_class_id),
        "hoax_class_id": int(runtime.hoax_class_id),
        "id2label": {str(k): v for k, v in runtime.id2label.items()},
        "label2id": runtime.label2id,
        "hoax_threshold": float(runtime.hoax_threshold),
        "calibration_loaded": bool(runtime.calibration_loaded),
        "raw_pred_id_distribution": pd.Series(raw_pred_ids).value_counts().sort_index().to_dict(),
        "pred_distribution": {str(k): int(v) for k, v in pred_dist.items()},
        "challenge_ready": challenge_ready,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro_dup": float(f_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f_weighted),
            "fakta_precision": float(p_each[0]) if len(p_each) > 0 else 0.0,
            "fakta_recall": float(r_each[0]) if len(r_each) > 0 else 0.0,
            "fakta_f1": float(f_each[0]) if len(f_each) > 0 else 0.0,
            "hoax_precision": hoax_precision,
            "hoax_recall": hoax_recall,
            "hoax_f1": float(f_each[1]) if len(f_each) > 1 else 0.0,
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[evaluate_model] rows={len(df)} accuracy={accuracy:.6f} f1_macro={f1_macro:.6f}")
    print(
        f"[evaluate_model] hoax_precision={hoax_precision:.6f} hoax_recall={hoax_recall:.6f} "
        f"challenge_ready={challenge_ready}"
    )
    print(f"[evaluate_model] confusion_matrix={cm.tolist()}")
    print(f"[evaluate_model] metrics written: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
