#!/usr/bin/env python
"""Evaluate classifier on processed test split and store metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
        default=ROOT / "artifacts" / "metrics.json",
        help="Output JSON file for metrics.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit number of test rows for faster evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    test_csv = args.test_csv.resolve()
    output_json = args.output_json.resolve()

    if not test_csv.exists():
        print(f"MISSING: {test_csv}")
        return 1

    try:
        from backend import app as backend_app
    except Exception as exc:
        print(f"Gagal import backend.app: {exc}")
        return 1

    try:
        backend_app._load_classifier()
    except Exception as exc:
        print(f"Gagal load classifier: {exc}")
        return 1

    df = pd.read_csv(test_csv)
    missing_cols = [col for col in ["text", "label"] if col not in df.columns]
    if missing_cols:
        print(f"Kolom wajib tidak ditemukan di {test_csv}: {missing_cols}")
        return 1

    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).tolist()

    predictions = backend_app._predict_batch(texts)
    y_pred = [1 if row["label"] == "Hoaks" else 0 for row in predictions]

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
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "test_csv": str(test_csv),
        "rows": int(len(df)),
        "model_source": backend_app.MODEL_SOURCE,
        "model_id": backend_app.MODEL_ID,
        "num_labels": backend_app.NUM_LABELS,
        "id2label": {str(k): v for k, v in backend_app.ID2LABEL.items()},
        "label2id": backend_app.LABEL2ID,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro_dup": float(f_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f_weighted),
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[evaluate_model] rows={len(df)} accuracy={accuracy:.6f} f1_macro={f1_macro:.6f}")
    print(f"[evaluate_model] confusion_matrix={cm.tolist()}")
    print(f"[evaluate_model] metrics written: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    if args.max_samples and len(df) > args.max_samples:
        df = (
            df.groupby("label", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=max(1, int(args.max_samples * len(x) / len(df))),
                    random_state=42,
                )
            )
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
