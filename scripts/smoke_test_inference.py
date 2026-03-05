#!/usr/bin/env python
"""Smoke test inference: ensure predictions do not collapse to one class."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inference_runtime import InferenceRuntime

CHALLENGE_ARTICLES = [
    {
        "id": 1,
        "expected_label": 0,
        "text": (
            "Gubernur DKI Jakarta Pramono Anung Wibowo disebut akan mengundang pihak-pihak terkait untuk membahas "
            "perizinan fasilitas padel di Jakarta. Langkah ini muncul setelah ada keluhan warga tentang kebisingan "
            "lapangan padel di dekat permukiman, dan pemerintah daerah disebut akan memetakan lokasi yang melanggar "
            "izin serta menyiapkan tindakan penertiban."
        ),
    },
    {
        "id": 2,
        "expected_label": 1,
        "text": (
            "Beredar unggahan di media sosial yang mengklaim Kementerian Kehutanan membuka rekrutmen CPNS Polisi "
            "Kehutanan tahun 2026 secara besar-besaran. Unggahan itu menyebut kebutuhan personel sangat tinggi dan "
            "menyatakan lulusan SMA diprioritaskan, disertai ajakan agar publik segera mendaftar."
        ),
    },
    {
        "id": 3,
        "expected_label": 0,
        "text": (
            "Sebuah pesawat pengangkut bahan bakar minyak dilaporkan jatuh di kawasan perbukitan Krayan, Kabupaten "
            "Nunukan, Kalimantan Utara. Laporan menyebut kepulan asap hitam terlihat setelah pesawat lepas landas "
            "usai mengantar pasokan BBM, dan saksi mata mengaku melihat pesawat sempat oleng sebelum jatuh pada "
            "kondisi cuaca yang berawan."
        ),
    },
    {
        "id": 4,
        "expected_label": 1,
        "text": (
            "Beredar sebuah video yang memperlihatkan suasana bandara dengan aktivitas pesawat dan iring-iringan "
            "kendaraan mengantar tamu mancanegara. Video itu diklaim sebagai momen kedatangan rombongan Perserikatan "
            "Bangsa-Bangsa ke IKN Nusantara untuk agenda peresmian, dan narasi tersebut ramai dibagikan warganet."
        ),
    },
    {
        "id": 5,
        "expected_label": 0,
        "text": (
            "PT Transjakarta dikabarkan melakukan modifikasi layanan pada empat rute mulai 21 Februari 2026, "
            "mencakup rute 4D, 6M, 9H, dan 11B. Perubahan ini disebut untuk mengoptimalkan layanan dan menjaga "
            "kenyamanan penumpang, dengan penyesuaian pada sebagian titik halte yang dilayani di rute tertentu."
        ),
    },
    {
        "id": 6,
        "expected_label": 1,
        "text": (
            "Sebuah video di TikTok dipublikasikan dengan narasi breaking yang mengklaim terjadi kericuhan saat "
            "ribuan orang turun ke jalan untuk memprotes kebijakan Presiden AS Donald Trump. Unggahan tersebut "
            "memicu banyak komentar dan dibagikan ulang, dengan sebagian warganet menganggapnya sebagai rekaman "
            "aksi massa terbaru."
        ),
    },
]


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
        "--challenge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Jalankan challenge 6 berita untuk cek non-collapse pada input produksi.",
    )
    parser.add_argument(
        "--challenge-output",
        type=Path,
        default=ROOT / "artifacts" / "challenge_metrics.json",
        help="Output JSON challenge metrics.",
    )
    parser.add_argument(
        "--challenge-min-correct",
        type=int,
        default=5,
        help="Batas minimal benar pada challenge 6 berita (untuk flag status).",
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
    eval_cmd = [
        sys.executable,
        str(root / "scripts" / "evaluate_model.py"),
        "--model-id",
        str(args.model_id),
        "--local-model-path",
        str(args.local_model_path),
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
    ]
    if args.calibration_path:
        eval_cmd.extend(["--calibration-path", str(args.calibration_path)])
    if args.device:
        eval_cmd.extend(["--device", str(args.device)])
    if args.hoax_threshold is not None:
        eval_cmd.extend(["--hoax-threshold", str(args.hoax_threshold)])
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
        "--challenge-output",
        str(args.challenge_output),
        "--challenge-min-correct",
        str(args.challenge_min_correct),
        "--model-id",
        str(args.model_id),
        "--local-model-path",
        str(args.local_model_path),
        "--max-length",
        str(args.max_length),
        "--batch-size",
        str(args.batch_size),
        "--_rerun",
    ]
    if args.calibration_path:
        smoke_cmd.extend(["--calibration-path", str(args.calibration_path)])
    if args.device:
        smoke_cmd.extend(["--device", str(args.device)])
    if args.hoax_threshold is not None:
        smoke_cmd.extend(["--hoax-threshold", str(args.hoax_threshold)])
    if args.challenge:
        smoke_cmd.append("--challenge")
    else:
        smoke_cmd.append("--no-challenge")
    print(f"[smoke] Menjalankan smoke ulang: {' '.join(smoke_cmd)}")
    proc = subprocess.run(smoke_cmd, cwd=str(root), check=False)
    return proc.returncode


def run_challenge(runtime: InferenceRuntime, output_path: Path, min_correct: int) -> Dict[str, object]:
    rows = []
    for article in CHALLENGE_ARTICLES:
        text = article["text"]
        paragraphs = runtime.split_paragraphs(text)
        sentence_rows = []
        for paragraph in paragraphs:
            sentences = runtime.split_sentences(paragraph)
            sentence_rows.extend(runtime.predict_batch(sentences))

        if sentence_rows:
            max_prob_hoax = max(item["prob_hoax"] for item in sentence_rows)
            avg_prob_hoax = float(sum(item["prob_hoax"] for item in sentence_rows) / len(sentence_rows))
            avg_prob_fakta = float(sum(item["prob_fakta"] for item in sentence_rows) / len(sentence_rows))
            doc_pred = 1 if max_prob_hoax >= float(runtime.hoax_threshold) else 0
        else:
            max_prob_hoax = 0.0
            avg_prob_hoax = 0.0
            avg_prob_fakta = 0.0
            doc_pred = 0

        rows.append(
            {
                "id": int(article["id"]),
                "expected_label": int(article["expected_label"]),
                "predicted_label": int(doc_pred),
                "pass": bool(int(article["expected_label"]) == int(doc_pred)),
                "doc_prob_hoax_max": round(float(max_prob_hoax), 6),
                "doc_prob_hoax_avg": round(avg_prob_hoax, 6),
                "doc_prob_fakta_avg": round(avg_prob_fakta, 6),
                "n_sentences": int(len(sentence_rows)),
            }
        )

    correct = int(sum(1 for row in rows if row["pass"]))
    pred_dist = pd.Series([row["predicted_label"] for row in rows]).value_counts().sort_index().to_dict()
    expected_dist = pd.Series([row["expected_label"] for row in rows]).value_counts().sort_index().to_dict()
    collapsed = len(pred_dist) < 2
    challenge_result = {
        "size": len(rows),
        "correct": correct,
        "accuracy": round(float(correct / len(rows)), 6) if rows else 0.0,
        "min_correct_target": int(min_correct),
        "pass_min_correct": bool(correct >= min_correct),
        "collapsed_one_class": bool(collapsed),
        "hoax_threshold": float(runtime.hoax_threshold),
        "pred_distribution": {str(k): int(v) for k, v in pred_dist.items()},
        "expected_distribution": {str(k): int(v) for k, v in expected_dist.items()},
        "rows": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(challenge_result, indent=2, ensure_ascii=False), encoding="utf-8")
    return challenge_result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    test_csv = args.test_csv.resolve()
    root = ROOT

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

    preds = runtime.predict_batch(df_sample["text"].astype(str).tolist())
    pred_ids = [int(row["pred_id"]) for row in preds]
    pred_dist = pd.Series(pred_ids).value_counts().sort_index().to_dict()

    print(f"[smoke] sample_size={len(df_sample)}")
    print(
        f"[smoke] class_ids=fakta:{int(runtime.fakta_class_id)} "
        f"hoaks:{int(runtime.hoax_class_id)}"
    )
    print(f"[smoke] pred_distribution={pred_dist}")
    print(
        f"[smoke] model_source={runtime.model_source} "
        f"hoax_threshold={runtime.hoax_threshold:.4f} "
        f"calibration_loaded={runtime.calibration_loaded}"
    )
    print("[smoke] contoh 3 prediksi:")
    for idx, (row, pred) in enumerate(zip(df_sample.itertuples(index=False), preds)):
        if idx >= 3:
            break
        print(
            f"  {idx+1}. true={int(getattr(row, 'label'))} pred_id={int(pred['pred_id'])} "
            f"pred_label={pred['label']} "
            f"prob_hoax={pred['prob_hoax']:.6f} prob_fakta={pred['prob_fakta']:.6f} "
            f"text={str(getattr(row, 'text'))[:140]}"
        )

    if len(set(pred_ids)) == 1:
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

    if args.challenge:
        challenge = run_challenge(
            runtime=runtime,
            output_path=args.challenge_output.resolve(),
            min_correct=args.challenge_min_correct,
        )
        print(
            f"[smoke] challenge accuracy={challenge['accuracy']:.4f} "
            f"correct={challenge['correct']}/{challenge['size']} "
            f"pass_min_correct={challenge['pass_min_correct']}"
        )
        print(f"[smoke] challenge pred_distribution={challenge['pred_distribution']}")
        print(f"[smoke] challenge metrics written: {args.challenge_output.resolve()}")
        if challenge["collapsed_one_class"]:
            print(
                "[smoke] ERROR: challenge 6 berita kolaps ke satu kelas. "
                "Periksa kualitas dataset dan threshold calibration."
            )
            return 3

    print("[smoke] PASS: prediksi tidak kolaps ke satu kelas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
