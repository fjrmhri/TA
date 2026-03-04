#!/usr/bin/env python
"""Build processed dataset for hoax detection with reduced source leakage."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED_COLUMNS = [
    "url",
    "judul",
    "tanggal",
    "isi_berita",
    "Narasi",
    "Clean Narasi",
    "summary",
    "hoax",
]

CSV_SOURCES = [
    ("turnbackhoax", "data_hoaks_turnbackhoaks.csv"),
    ("cnn", "data_nonhoaks_cnn.csv"),
    ("detik", "data_nonhoaks_detik.csv"),
    ("kompas", "data_nonhoaks_kompas.csv"),
]

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

DEBUNK_KEYWORDS_RE = re.compile(
    r"\b(faktanya|cek fakta|klarifikasi|berdasarkan|kesimpulan|tidak benar|hasilnya|fakta)\b",
    flags=re.IGNORECASE,
)

BOILERPLATE_PATTERNS: Sequence[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?i)\b(?:nasional|internasional|ekonomi|olahraga|teknologi|gaya hidup|otomotif|edukasi|hiburan)\b"
            r"(?:\s+\w+){0,8}\s+cnn indonesia\s+\w+,\s*\d{1,2}\s+\w+\s+\d{4}\s+\d{1,2}:\d{2}\s+wib"
        ),
        " ",
    ),
    (re.compile(r"(?i)bagikan:\s*url telah tercopy"), " "),
    (re.compile(r"(?i)\bcnn\s+indonesia\b"), " "),
    (re.compile(r"(?i)\bkompas\.com\b"), " "),
    (re.compile(r"(?i)\bdetik(?:com)?\b"), " "),
    (re.compile(r"(?i)\bmafindo\b"), " "),
    (re.compile(r"(?i)\bturnbackhoax(?:s)?\b"), " "),
    (re.compile(r"(?i)\badvertisement\b\s*scroll to continue with content"), " "),
    (re.compile(r"(?i)\bbaca juga:\s*[^.!?\n]*"), " "),
    (re.compile(r"(?i)\blihat juga:\s*[^.!?\n]*"), " "),
    (re.compile(r"(?i)\bikuti\s+whatsapp\s+channel\b[^.!?\n]*"), " "),
    (re.compile(r"\[\s*arsip\s*\]", flags=re.IGNORECASE), " "),
    (
        re.compile(
            r"\[(?:\s*salah\s*|\s*penipuan\s*|\s*sesat\s*|\s*disinformasi\s*|\s*fitnah\s*)\]",
            flags=re.IGNORECASE,
        ),
        " ",
    ),
    (
        re.compile(
            r"(?i)^\s*(salah|tipu|penipuan|sesat|disinformasi|fitnah)\b[:\-\s]*"
        ),
        " ",
    ),
]


@dataclass
class BuildConfig:
    root: Path
    output_dir: Path
    seed: int = 42
    min_words: int = 6
    nonhoax_lead_sentences: int = 2
    val_test_ratio_from_holdout: float = 0.5


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_text(text: str) -> str:
    cleaned = normalize_whitespace(text)
    for pattern, replacement in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = normalize_whitespace(cleaned)
    cleaned = cleaned.strip(" -–—:;,.")
    return cleaned


def split_sentences(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(normalized) if part.strip()]


def sha1_16(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def hash_fields(url: str, title: str, date: str) -> Tuple[str, str]:
    url_key = normalize_whitespace(url).lower()
    title_date_key = f"{normalize_whitespace(title).lower()}|{normalize_whitespace(date).lower()}"
    if url_key:
        url_hash = sha1_16(url_key)
    else:
        url_hash = sha1_16(title_date_key)
    title_hash = sha1_16(title_date_key)
    return url_hash, title_hash


def valid_text(text: str, min_words: int) -> bool:
    return len(normalize_whitespace(text).split()) >= min_words


def pick_nonhoax_text(row: pd.Series, min_words: int, nonhoax_lead_sentences: int) -> str:
    summary = clean_text(row.get("summary", ""))
    if valid_text(summary, min_words):
        return summary

    isi = clean_text(row.get("isi_berita", ""))
    lead = " ".join(split_sentences(isi)[:nonhoax_lead_sentences])
    if valid_text(lead, min_words):
        return lead
    return ""


def pick_debunk_sentence(isi_berita: str, min_words: int) -> str:
    original_sentences = split_sentences(isi_berita)
    if not original_sentences:
        return ""

    selected = ""
    for sentence in original_sentences:
        if DEBUNK_KEYWORDS_RE.search(sentence):
            selected = sentence
            break

    if not selected:
        # Fallback: sentence from later part of the article (bias toward clarification section).
        fallback_idx = min(len(original_sentences) - 1, max(0, int(len(original_sentences) * 0.66)))
        selected = original_sentences[fallback_idx]

    cleaned = clean_text(selected)
    if not valid_text(cleaned, min_words):
        return ""
    return cleaned


def load_csv_checked(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"MISSING: {path}")
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"MISSING COLUMNS in {path}: {missing_cols}")
    return df


def build_records(config: BuildConfig) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for source, filename in CSV_SOURCES:
        csv_path = config.root / filename
        df = load_csv_checked(csv_path)

        for _, row in df.iterrows():
            url_hash, title_hash = hash_fields(
                str(row.get("url", "")),
                str(row.get("judul", "")),
                str(row.get("tanggal", "")),
            )

            if source == "turnbackhoax":
                claim = clean_text(row.get("Clean Narasi", "") or row.get("Narasi", ""))
                if valid_text(claim, config.min_words):
                    records.append(
                        {
                            "text": claim,
                            "label": 1,
                            "source": source,
                            "url_hash": url_hash,
                            "title_hash": title_hash,
                            "unit_type": "claim",
                        }
                    )

                debunk = pick_debunk_sentence(str(row.get("isi_berita", "")), config.min_words)
                if debunk:
                    records.append(
                        {
                            "text": debunk,
                            "label": 0,
                            "source": source,
                            "url_hash": url_hash,
                            "title_hash": title_hash,
                            "unit_type": "debunk",
                        }
                    )
                continue

            fakta_text = pick_nonhoax_text(
                row=row,
                min_words=config.min_words,
                nonhoax_lead_sentences=config.nonhoax_lead_sentences,
            )
            if fakta_text:
                records.append(
                    {
                        "text": fakta_text,
                        "label": 0,
                        "source": source,
                        "url_hash": url_hash,
                        "title_hash": title_hash,
                        "unit_type": "lead",
                    }
                )

    if not records:
        raise RuntimeError("Tidak ada record valid yang berhasil dibangun.")

    df_records = pd.DataFrame(records)
    df_records["text"] = df_records["text"].astype(str).map(normalize_whitespace)
    df_records = df_records[df_records["text"] != ""].copy()
    df_records["label"] = df_records["label"].astype(int)

    before = len(df_records)
    df_records = df_records.drop_duplicates(subset=["text", "label"], keep="first").reset_index(drop=True)
    dedup_removed = before - len(df_records)
    print(f"[build_dataset] drop_duplicates(text,label): removed={dedup_removed} remaining={len(df_records)}")
    return df_records


def choose_holdout_folds(
    fold_sizes: Dict[int, int], total: int, target_ratio: float
) -> Tuple[int, ...]:
    fold_ids = sorted(fold_sizes.keys())
    target_size = total * target_ratio
    best_combo: Tuple[int, ...] = tuple(fold_ids[:3])
    best_gap = float("inf")

    for combo in itertools.combinations(fold_ids, 3):
        size = sum(fold_sizes[fid] for fid in combo)
        gap = abs(size - target_size)
        if gap < best_gap:
            best_gap = gap
            best_combo = combo
    return best_combo


def stratified_group_split(
    df: pd.DataFrame, seed: int, val_test_ratio_from_holdout: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["url_hash"].astype(str).values
    labels = df["label"].astype(int).values

    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_to_idx: Dict[int, List[int]] = {}
    for fold_id, (_, holdout_idx) in enumerate(outer.split(df["text"], labels, groups=groups)):
        fold_to_idx[fold_id] = holdout_idx.tolist()

    fold_sizes = {fid: len(indices) for fid, indices in fold_to_idx.items()}
    holdout_folds = choose_holdout_folds(fold_sizes=fold_sizes, total=len(df), target_ratio=0.30)

    holdout_set = set()
    for fid in holdout_folds:
        holdout_set.update(fold_to_idx[fid])
    holdout_idx = sorted(holdout_set)
    train_idx = sorted(set(range(len(df))) - holdout_set)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)

    inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    inner_groups = holdout_df["url_hash"].astype(str).values
    inner_labels = holdout_df["label"].astype(int).values
    inner_split = next(inner.split(holdout_df["text"], inner_labels, groups=inner_groups))
    val_rel_idx, test_rel_idx = inner_split

    val_df = holdout_df.iloc[val_rel_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_rel_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def summarize_split(name: str, df: pd.DataFrame) -> Dict[str, object]:
    label_dist = df["label"].value_counts().sort_index().to_dict()
    source_label = pd.crosstab(df["source"], df["label"]).to_dict()
    summary = {
        "rows": int(len(df)),
        "label_dist": {str(k): int(v) for k, v in label_dist.items()},
        "source_label": source_label,
    }
    print(f"\n[{name}] rows={summary['rows']}")
    print(f"[{name}] label_dist={summary['label_dist']}")
    print(f"[{name}] source_label_crosstab:")
    print(pd.crosstab(df["source"], df["label"]))
    return summary


def run(config: BuildConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_records(config)

    train_df, val_df, test_df = stratified_group_split(
        dataset, seed=config.seed, val_test_ratio_from_holdout=config.val_test_ratio_from_holdout
    )

    out_columns = ["text", "label", "source", "url_hash", "title_hash", "unit_type"]
    train_path = config.output_dir / "train.csv"
    val_path = config.output_dir / "val.csv"
    test_path = config.output_dir / "test.csv"

    train_df[out_columns].to_csv(train_path, index=False)
    val_df[out_columns].to_csv(val_path, index=False)
    test_df[out_columns].to_csv(test_path, index=False)

    summary = {
        "output_dir": str(config.output_dir),
        "rows_total": int(len(dataset)),
        "train": summarize_split("train", train_df),
        "val": summarize_split("val", val_df),
        "test": summarize_split("test", test_df),
        "files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }

    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[build_dataset] summary written: {summary_path}")
    print(f"[build_dataset] train={train_path} val={val_path} test={test_path}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed dataset for hoax detection.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path containing the 4 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed",
        help="Output directory for train/val/test CSV.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-words", type=int, default=6, help="Minimum words per text unit.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    cfg = BuildConfig(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        min_words=args.min_words,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
