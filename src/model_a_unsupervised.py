"""Train and evaluate Model A unsupervised baseline using KMeans."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, silhouette_score
from sklearn.pipeline import Pipeline

from .config import DEFAULT_CONFIG, ProjectConfig
from .utils import ensure_dir, save_json, set_random_seed, setup_logger, utc_timestamp

MODEL_LABELS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for unsupervised Model A."""
    parser = argparse.ArgumentParser(description="Train KMeans baseline for Model A verification.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument("--clusters", type=int, default=4, help="Number of KMeans clusters.")
    parser.add_argument("--max-features", type=int, default=20000, help="TF-IDF max feature count.")
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2], help="Maximum n-gram length.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
        help="Max number of rows sampled from train split for silhouette score.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate KMeans mapped labels on test split.",
    )
    return parser.parse_args()


def load_processed_splits(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train/val/test files from data/processed."""
    train_path = config.processed_data_dir / "train_processed.csv"
    val_path = config.processed_data_dir / "val_processed.csv"
    test_path = config.processed_data_dir / "test_processed.csv"

    missing = [str(p) for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing processed dataset file(s). Run `python -m src.preprocessing` first.\n"
            + "\n".join(missing)
        )

    return pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)


def validate_processed_schema(df: pd.DataFrame, split_name: str) -> None:
    """Validate required processed columns for unsupervised training."""
    required = ["verifier_input", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split missing required columns: {missing}")


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return input text and normalized label series."""
    x = df["verifier_input"].astype(str)
    y = df["answer"].astype(str).str.strip().str.upper()
    return x, y


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    """Build TF-IDF vectorizer used before KMeans."""
    return TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=True,
    )


def build_cluster_label_mapping(cluster_ids: List[int], labels: List[str]) -> Dict[int, str]:
    """Map clusters to labels with a one-to-one assignment whenever possible."""
    grouped: Dict[int, List[str]] = defaultdict(list)
    for cluster_id, label in zip(cluster_ids, labels):
        grouped[int(cluster_id)].append(str(label).strip().upper())

    clusters = sorted(grouped.keys())
    labels_sorted = list(MODEL_LABELS)
    count_matrix = np.zeros((len(clusters), len(labels_sorted)), dtype=np.int64)

    for i, cluster_id in enumerate(clusters):
        counts = Counter(grouped[cluster_id])
        for j, label in enumerate(labels_sorted):
            count_matrix[i, j] = int(counts.get(label, 0))

    mapping: Dict[int, str] = {}
    if count_matrix.size > 0:
        # Maximize total agreement via Hungarian assignment (convert to minimization cost).
        cost = count_matrix.max() - count_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            mapping[clusters[int(r)]] = labels_sorted[int(c)]

    # Fallback for any unmatched clusters.
    for cluster_id in clusters:
        if cluster_id not in mapping:
            mapping[cluster_id] = Counter(grouped[cluster_id]).most_common(1)[0][0]
    return mapping


def map_clusters_to_labels(cluster_ids: List[int], mapping: Dict[int, str], default_label: str = "A") -> List[str]:
    """Convert predicted clusters into class labels using trained mapping."""
    return [mapping.get(int(cluster_id), default_label) for cluster_id in cluster_ids]


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    """Compute standard Model A classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": MODEL_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=MODEL_LABELS).tolist(),
    }


def maybe_compute_silhouette(
    x_train_tfidf: csr_matrix, train_cluster_ids: List[int], sample_size: int, seed: int
) -> float | None:
    """Compute silhouette score with optional sampling for speed."""
    n_rows = x_train_tfidf.shape[0]
    if n_rows < 2:
        return None
    if len(set(train_cluster_ids)) < 2:
        return None
    current_sample = min(sample_size, n_rows)
    return float(
        silhouette_score(
            x_train_tfidf,
            train_cluster_ids,
            metric="cosine",
            sample_size=current_sample,
            random_state=seed,
        )
    )


def load_supervised_baseline(report_dir: Path) -> Dict[str, object] | None:
    """Load existing supervised baseline metrics report if present."""
    baseline_path = report_dir / "baseline_metrics.json"
    if not baseline_path.exists():
        return None
    with baseline_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_comparison_section(
    unsupervised_result: Dict[str, object], supervised_report: Dict[str, object] | None
) -> Dict[str, object]:
    """Create a simple comparison table against supervised baselines."""
    table: List[Dict[str, object]] = [
        {
            "model_name": "kmeans_label_mapped",
            "validation_accuracy": unsupervised_result["validation_metrics"]["accuracy"],
            "validation_macro_f1": unsupervised_result["validation_metrics"]["macro_f1"],
        }
    ]

    if supervised_report and "models" in supervised_report:
        for model_name, model_data in supervised_report["models"].items():
            val_metrics = model_data.get("validation_metrics", {})
            table.append(
                {
                    "model_name": model_name,
                    "validation_accuracy": val_metrics.get("accuracy"),
                    "validation_macro_f1": val_metrics.get("macro_f1"),
                }
            )

    table_sorted = sorted(
        table,
        key=lambda row: row["validation_macro_f1"] if row["validation_macro_f1"] is not None else -1.0,
        reverse=True,
    )
    return {"validation_table_sorted_by_macro_f1": table_sorted}


def run_kmeans_pipeline(config: ProjectConfig, args: argparse.Namespace) -> Path:
    """Train KMeans baseline, evaluate, and export artifacts/reports."""
    logger = setup_logger("model_a_unsupervised")
    logger.info("Starting Model A unsupervised pipeline (KMeans)")
    logger.info("Random seed: %s", config.random_seed)
    set_random_seed(config.random_seed)

    train_df, val_df, test_df = load_processed_splits(config)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_processed_schema(df, split_name)

    x_train, y_train = prepare_xy(train_df)
    x_val, y_val = prepare_xy(val_df)
    x_test, y_test = prepare_xy(test_df)
    logger.info("Loaded data | train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))

    vectorizer = build_vectorizer(args)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)
    x_test_tfidf = vectorizer.transform(x_test)
    logger.info("Vectorization complete | train_matrix_shape=%s", x_train_tfidf.shape)

    kmeans = KMeans(n_clusters=args.clusters, random_state=args.seed, n_init=10)
    train_cluster_ids = kmeans.fit_predict(x_train_tfidf)
    mapping = build_cluster_label_mapping(train_cluster_ids.tolist(), y_train.tolist())
    logger.info("Trained KMeans | clusters=%s", args.clusters)

    val_cluster_ids = kmeans.predict(x_val_tfidf)
    val_pred_labels = map_clusters_to_labels(val_cluster_ids.tolist(), mapping)
    val_metrics = evaluate_predictions(y_val.tolist(), val_pred_labels)

    result: Dict[str, object] = {
        "model_name": "kmeans_label_mapped",
        "clusters": args.clusters,
        "cluster_label_mapping": {str(k): v for k, v in mapping.items()},
        "validation_metrics": val_metrics,
    }

    if args.evaluate_test:
        test_cluster_ids = kmeans.predict(x_test_tfidf)
        test_pred_labels = map_clusters_to_labels(test_cluster_ids.tolist(), mapping)
        result["test_metrics"] = evaluate_predictions(y_test.tolist(), test_pred_labels)

    silhouette = maybe_compute_silhouette(
        x_train_tfidf=x_train_tfidf,
        train_cluster_ids=train_cluster_ids.tolist(),
        sample_size=args.silhouette_sample_size,
        seed=args.seed,
    )
    result["silhouette_score"] = silhouette
    logger.info(
        "Validation | accuracy=%.4f macro_f1=%.4f silhouette=%s",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
        f"{silhouette:.4f}" if silhouette is not None else "N/A",
    )

    model_dir = ensure_dir(config.project_root / "models" / "model_a" / "traditional")
    report_dir = ensure_dir(config.project_root / "models" / "model_a" / "reports")
    model_path = model_dir / "kmeans_label_mapped.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "kmeans": kmeans,
            "cluster_label_mapping": mapping,
            "labels": MODEL_LABELS,
        },
        model_path,
    )
    logger.info("Saved unsupervised model artifact: %s", model_path)

    supervised_report = load_supervised_baseline(report_dir)
    comparison = build_comparison_section(result, supervised_report)

    report_payload: Dict[str, object] = {
        "generated_at_utc": utc_timestamp(),
        "seed": config.random_seed,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "vectorizer": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "min_df": args.min_df,
            "stop_words": "english",
            "sublinear_tf": True,
        },
        "kmeans": result,
        "comparison": comparison,
    }

    report_path = report_dir / "unsupervised_metrics.json"
    save_json(report_path, report_payload)
    logger.info("Saved unsupervised metrics report: %s", report_path)
    logger.info("Model A unsupervised pipeline completed")
    return report_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)
    run_kmeans_pipeline(config, args)


if __name__ == "__main__":
    main()
