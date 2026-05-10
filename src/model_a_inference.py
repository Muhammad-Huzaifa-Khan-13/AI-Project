"""Shared utilities for option-wise Model A training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

MODEL_LABELS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class OptionRowBatch:
    """Expanded option-wise rows derived from MCQ records."""

    qid: np.ndarray
    option_label: np.ndarray
    text: List[str]
    is_correct: np.ndarray


def _norm_text(text: object) -> str:
    return str(text or "").strip()


def row_qid(row: pd.Series, idx: int) -> str:
    """Build a stable unique key per question row."""
    source_id = str(row.get("id", "")).strip()
    return f"{source_id}__{idx}" if source_id else str(idx)


def build_option_text(row: pd.Series, option_label: str) -> str:
    """Build one candidate text for binary correctness scoring."""
    article = _norm_text(row.get("clean_article", row.get("article", "")))
    question = _norm_text(row.get("clean_question", row.get("question", "")))
    option_text = _norm_text(row.get(f"clean_{option_label}", row.get(option_label, "")))
    return f"{article} [SEP] {question} [SEP] {option_text}"


def expand_mcq_rows(df: pd.DataFrame) -> OptionRowBatch:
    """Expand one-question rows into four option-wise binary rows."""
    qids: List[str] = []
    option_labels: List[str] = []
    texts: List[str] = []
    targets: List[int] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        true_label = str(row["answer"]).strip().upper()
        qid_value = row_qid(row, idx)
        for label in MODEL_LABELS:
            qids.append(qid_value)
            option_labels.append(label)
            texts.append(build_option_text(row, label))
            targets.append(1 if label == true_label else 0)

    return OptionRowBatch(
        qid=np.asarray(qids, dtype=object),
        option_label=np.asarray(option_labels, dtype=object),
        text=texts,
        is_correct=np.asarray(targets, dtype=np.int32),
    )


def score_positive_class(classifier, x_matrix) -> np.ndarray:
    """Return score for positive class for binary classifiers."""
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(x_matrix)
        return probs[:, 1]

    if hasattr(classifier, "decision_function"):
        margin = classifier.decision_function(x_matrix)
        margin = np.asarray(margin, dtype=np.float64)
        if margin.ndim == 1:
            return margin
        if margin.shape[1] == 2:
            return margin[:, 1]
        return margin.max(axis=1)

    preds = classifier.predict(x_matrix)
    return np.asarray(preds, dtype=np.float64)


def predict_option_labels(
    qid: Sequence[str], option_label: Sequence[str], option_score: Sequence[float]
) -> Tuple[List[str], List[str]]:
    """Pick one option label per question using maximum score."""
    tmp = pd.DataFrame({"qid": list(qid), "option": list(option_label), "score": list(option_score)})
    best = tmp.sort_values(["qid", "score"], ascending=[True, False]).groupby("qid", as_index=False).first()
    return best["qid"].astype(str).tolist(), best["option"].astype(str).tolist()


def true_labels_by_qid(df: pd.DataFrame) -> Dict[str, str]:
    """Return true label keyed by question id."""
    base = df.reset_index(drop=True)
    return {row_qid(row, idx): str(row["answer"]).strip().upper() for idx, row in base.iterrows()}


def predict_df_labels(artifact: Dict[str, object], df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Predict one label (A/B/C/D) per row in dataframe using option-wise artifact."""
    batch = expand_mcq_rows(df)
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    x = vectorizer.transform(batch.text)
    scores = score_positive_class(classifier, x)
    pred_qids, pred_labels = predict_option_labels(batch.qid, batch.option_label, scores)

    true_map = true_labels_by_qid(df)
    y_true = [true_map[qid] for qid in pred_qids]
    return pred_qids, y_true, pred_labels


def predict_single_row_label(artifact: Dict[str, object], row: pd.Series) -> str:
    """Predict one answer label for a single MCQ row."""
    items: List[Tuple[str, float]] = []
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    for label in MODEL_LABELS:
        text = build_option_text(row, label)
        x = vectorizer.transform([text])
        score = float(score_positive_class(classifier, x)[0])
        items.append((label, score))
    items.sort(key=lambda x: x[1], reverse=True)
    return items[0][0]
