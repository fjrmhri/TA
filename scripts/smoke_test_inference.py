#!/usr/bin/env python
"""Smoke test inference: ensure predictions do not collapse to one class."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke test on processed test split.")
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "test.csv",
        help="Path to processed test CSV.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=5,
        help="Number of samples per label (0 and 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--attempt-retrain",
        action="store_true",
        help="If collapse detected, attempt retrain and rerun evaluate/smoke once.",
    )
    parser.add_argument(
        "--_rerun",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def run_retrain_attempt(root: Path) -> int:
    retrain_script = root / "scripts" / "retrain_from_processed.py"
    if not retrain_script.exists():
        print(f"[smoke] Retrain script tidak ditemukan: {retrain_script}")
        return 1

    cmd = [
        sys.executable,
        str(retrain_script),
        "--epochs",
        "0.2",
        "--max-train-samples",
        "3000",
        "--max-val-samples",
        "800",
    ]
    print(f"[smoke] Menjalankan retrain attempt: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(root), check=False)
    return proc.returncode


def run_eval_and_rerun_smoke(root: Path, args: argparse.Namespace) -> int:
    eval_cmd = [sys.executable, str(root / "scripts" / "evaluate_model.py")]
    print(f"[smoke] Menjalankan evaluasi ulang: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, cwd=str(root), check=False)

    smoke_cmd = [
        sys.executable,
        str(root / "scripts" / "smoke_test_inference.py"),
        "--test-csv",
        str(args.test_csv),
        "--samples-per-label",
        str(args.samples_per_label),
        "--seed",
        str(args.seed),
        "--_rerun",
    ]
    print(f"[smoke] Menjalankan smoke ulang: {' '.join(smoke_cmd)}")
    proc = subprocess.run(smoke_cmd, cwd=str(root), check=False)
    return proc.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    test_csv = args.test_csv.resolve()
    root = ROOT

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

    samples = []
    for label in [0, 1]:
        subset = df[df["label"] == label]
        if len(subset) < args.samples_per_label:
            print(
                f"[smoke] Label {label} hanya punya {len(subset)} baris, diminta {args.samples_per_label}. "
                f"Menggunakan seluruh baris label tersebut."
            )
        take = subset.sample(n=min(args.samples_per_label, len(subset)), random_state=args.seed)
        samples.append(take)

    df_sample = pd.concat(samples, ignore_index=True)
    if df_sample.empty:
        print("[smoke] Sampel kosong, tidak dapat menjalankan smoke test.")
        return 1

    preds = backend_app._predict_batch(df_sample["text"].astype(str).tolist())
    pred_labels = [1 if row["label"] == "Hoaks" else 0 for row in preds]
    pred_dist = pd.Series(pred_labels).value_counts().sort_index().to_dict()

    print(f"[smoke] sample_size={len(df_sample)}")
    print(f"[smoke] pred_distribution={pred_dist}")
    print("[smoke] contoh 3 prediksi:")
    for idx, (row, pred) in enumerate(zip(df_sample.itertuples(index=False), preds)):
        if idx >= 3:
            break
        print(
            f"  {idx+1}. true={int(getattr(row, 'label'))} pred={pred['label']} "
            f"prob_hoax={pred['prob_hoax']:.6f} prob_fakta={pred['prob_fakta']:.6f} "
            f"text={str(getattr(row, 'text'))[:140]}"
        )

    if len(set(pred_labels)) == 1:
        print(
            "[smoke] ERROR: prediksi kolaps ke satu kelas. "
            "Periksa mapping label, preprocessing inference, dan model artifact yang dimuat."
        )
        if args.attempt_retrain and not args._rerun:
            retrain_code = run_retrain_attempt(root)
            if retrain_code == 0:
                return run_eval_and_rerun_smoke(root, args)
            print(f"[smoke] Retrain attempt gagal (exit={retrain_code}).")
        return 2

    print("[smoke] PASS: prediksi tidak kolaps ke satu kelas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
