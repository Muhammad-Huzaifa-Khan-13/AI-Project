"""Shared utilities for Model B distractor and hint pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ProjectConfig
from .utils import ensure_dir

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_WORD_PATTERN = re.compile(r"[^\w\s]")
REQUIRED_COLUMNS = ["id", "article", "question", "A", "B", "C", "D", "answer"]


def normalize_text(text: object) -> str:
    """Lowercase, remove punctuation, and normalize whitespace."""
    if text is None:
        return ""
    cleaned = str(text).lower()
    cleaned = NON_WORD_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def split_into_sentences(article: str) -> List[str]:
    """Split article into approximate sentences and remove empty chunks."""
    if not isinstance(article, str):
        return []
    raw = article.replace("\n", " ").strip()
    if not raw:
        return []
    parts = SENTENCE_SPLIT_PATTERN.split(raw)
    return [p.strip() for p in parts if p and p.strip()]


def compute_cosine_to_query(query: str, candidates: Sequence[str], max_features: int = 5000) -> List[float]:
    """Compute cosine similarity between one query and candidate texts."""
    if not candidates:
        return []
    corpus = [query] + list(candidates)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=max_features)
    tfidf = vectorizer.fit_transform(corpus)
    query_vec = tfidf[0:1]
    cand_vecs = tfidf[1:]
    sims = cosine_similarity(query_vec, cand_vecs)[0]
    return [float(x) for x in sims]


def cosine_between_texts(text_a: str, text_b: str, max_features: int = 2000) -> float:
    """Compute cosine similarity between two texts."""
    texts = [text_a, text_b]
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_features=max_features)
    tfidf = vectorizer.fit_transform(texts)
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


def load_processed_split(config: ProjectConfig, split: str) -> pd.DataFrame:
    """Load processed split by name (train/val/test) from data/processed."""
    split_map = {
        "train": "train_processed.csv",
        "val": "val_processed.csv",
        "test": "test_processed.csv",
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {list(split_map)}")

    path = config.processed_data_dir / split_map[split]
    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}. Run `python -m src.preprocessing` first.")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Processed split missing required columns: {missing}")
    return df


def maybe_sample(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """Optionally down-sample dataframe for quick experiments."""
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def get_correct_answer_text(row: pd.Series) -> str:
    """Return correct option text from answer label."""
    label = str(row["answer"]).strip().upper()
    if label not in {"A", "B", "C", "D"}:
        return ""
    return str(row[label])


def get_wrong_option_texts(row: pd.Series) -> List[str]:
    """Return non-correct option texts."""
    correct_label = str(row["answer"]).strip().upper()
    options = {label: str(row[label]) for label in ["A", "B", "C", "D"]}
    wrong = [text for label, text in options.items() if label != correct_label]
    return wrong


def build_model_b_paths(config: ProjectConfig) -> Dict[str, Path]:
    """Prepare standard model B artifact directories."""
    base_dir = ensure_dir(config.project_root / "models" / "model_b")
    outputs_dir = ensure_dir(base_dir / "outputs")
    reports_dir = ensure_dir(base_dir / "reports")
    return {"base_dir": base_dir, "outputs_dir": outputs_dir, "reports_dir": reports_dir}
