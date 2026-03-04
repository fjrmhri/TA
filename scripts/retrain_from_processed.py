#!/usr/bin/env python
"""Retrain classifier from processed train/val split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Retrain model from processed train/val data.")
    parser.add_argument("--model-name", default="indolem/indobert-base-uncased")
    parser.add_argument("--train-csv", type=Path, default=root / "data" / "processed" / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=root / "data" / "processed" / "val.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "indobert_hoax_ner_model_final")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=4000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_weighted),
        "recall_weighted": float(r_weighted),
        "f1_weighted": float(f1_weighted),
        "cm_tn": int(cm[0, 0]),
        "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]),
        "cm_tp": int(cm[1, 1]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    train_csv = args.train_csv.resolve()
    val_csv = args.val_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not train_csv.exists():
        print(f"MISSING: {train_csv}")
        return 1
    if not val_csv.exists():
        print(f"MISSING: {val_csv}")
        return 1

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retrain] device={device}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    for name, df in [("train", df_train), ("val", df_val)]:
        missing_cols = [col for col in ["text", "label"] if col not in df.columns]
        if missing_cols:
            print(f"[retrain] {name} missing columns: {missing_cols}")
            return 1

    if args.max_train_samples and len(df_train) > args.max_train_samples:
        df_train = (
            df_train.groupby("label", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=max(1, int(args.max_train_samples * len(x) / len(df_train))),
                    random_state=args.seed,
                )
            )
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )
    if args.max_val_samples and len(df_val) > args.max_val_samples:
        df_val = (
            df_val.groupby("label", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=max(1, int(args.max_val_samples * len(x) / len(df_val))),
                    random_state=args.seed,
                )
            )
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )

    print(f"[retrain] train_rows={len(df_train)} val_rows={len(df_val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        use_safetensors=False,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    ds_train = Dataset.from_pandas(df_train[["text", "label"]], preserve_index=False)
    ds_val = Dataset.from_pandas(df_val[["text", "label"]], preserve_index=False)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    ds_train = ds_train.map(tok, batched=True)
    ds_val = ds_val.map(tok, batched=True)
    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    ds_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if device == "cuda" else None,
    )

    training_args_kwargs = dict(
        output_dir=str(output_dir / "_tmp_retrain"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=float(args.epochs),
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=True if device == "cuda" else False,
        gradient_checkpointing=True,
        auto_find_batch_size=True,
        report_to="none",
        logging_steps=100,
    )
    training_arg_names = TrainingArguments.__init__.__code__.co_varnames
    if "eval_strategy" in training_arg_names:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    train_args = TrainingArguments(**training_args_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.0)],
    )
    trainer_arg_names = Trainer.__init__.__code__.co_varnames
    if "processing_class" in trainer_arg_names:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    eval_metrics = trainer.evaluate(ds_val)

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metric_json = output_dir / "retrain_eval_metrics.json"
    metric_json.write_text(json.dumps(eval_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[retrain] model saved: {output_dir}")
    print(f"[retrain] eval metrics: {metric_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
