"""Train and evaluate Model A hard-voting ensemble (Logistic Regression + Linear SVM)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from .config import DEFAULT_CONFIG, ProjectConfig
from .utils import ensure_dir, save_json, set_random_seed, setup_logger, utc_timestamp

MODEL_LABELS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ensemble training."""
    parser = argparse.ArgumentParser(description="Train Model A hard-voting ensemble on processed data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument("--max-features", type=int, default=20000, help="TF-IDF max feature count.")
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2], help="Maximum n-gram length.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate on the test split.",
    )
    return parser.parse_args()


def load_processed_splits(config: ProjectConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed train/val/test splits from disk."""
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


def validate_schema(df: pd.DataFrame, split_name: str) -> None:
    """Validate required columns for ensemble training."""
    required = ["verifier_input", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split missing required columns: {missing}")


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return input feature text and labels."""
    x = df["verifier_input"].astype(str)
    y = df["answer"].astype(str).str.strip().str.upper()
    return x, y


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    """Build shared TF-IDF vectorizer for ensemble members."""
    return TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=True,
    )


def hard_vote(pred_a: List[str], pred_b: List[str], tie_breaker: List[str]) -> List[str]:
    """
    Perform hard voting between two predictors.

    If both agree, keep that label.
    If they disagree, use tie_breaker prediction (Logistic Regression).
    """
    voted: List[str] = []
    for a, b, t in zip(pred_a, pred_b, tie_breaker):
        if a == b:
            voted.append(a)
        else:
            voted.append(t)
    return voted


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    """Compute metrics for multi-class answer verification."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": MODEL_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=MODEL_LABELS).tolist(),
    }


def load_report(report_path: Path) -> Dict[str, object] | None:
    """Load JSON report if it exists."""
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_comparison_table(
    ensemble_val_metrics: Dict[str, object],
    baseline_report: Dict[str, object] | None,
    unsupervised_report: Dict[str, object] | None,
) -> Dict[str, object]:
    """Build validation comparison table across available Model A methods."""
    table: List[Dict[str, object]] = [
        {
            "model_name": "ensemble_hard_vote_lr_svm",
            "validation_accuracy": ensemble_val_metrics["accuracy"],
            "validation_macro_f1": ensemble_val_metrics["macro_f1"],
        }
    ]

    if baseline_report and "models" in baseline_report:
        for model_name, model_data in baseline_report["models"].items():
            val_metrics = model_data.get("validation_metrics", {})
            table.append(
                {
                    "model_name": model_name,
                    "validation_accuracy": val_metrics.get("accuracy"),
                    "validation_macro_f1": val_metrics.get("macro_f1"),
                }
            )

    if unsupervised_report and "kmeans" in unsupervised_report:
        val_metrics = unsupervised_report["kmeans"].get("validation_metrics", {})
        table.append(
            {
                "model_name": "kmeans_label_mapped",
                "validation_accuracy": val_metrics.get("accuracy"),
                "validation_macro_f1": val_metrics.get("macro_f1"),
            }
        )

    sorted_table = sorted(
        table,
        key=lambda row: row["validation_macro_f1"] if row["validation_macro_f1"] is not None else -1.0,
        reverse=True,
    )
    return {"validation_table_sorted_by_macro_f1": sorted_table}


def run_ensemble_pipeline(config: ProjectConfig, args: argparse.Namespace) -> Path:
    """Train hard-voting ensemble, evaluate, and save artifacts."""
    logger = setup_logger("model_a_ensemble")
    logger.info("Starting Model A ensemble pipeline")
    logger.info("Random seed: %s", config.random_seed)
    set_random_seed(config.random_seed)

    train_df, val_df, test_df = load_processed_splits(config)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_schema(df, split_name)

    x_train, y_train = prepare_xy(train_df)
    x_val, y_val = prepare_xy(val_df)
    x_test, y_test = prepare_xy(test_df)

    vectorizer = build_vectorizer(args)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)
    x_test_tfidf = vectorizer.transform(x_test)
    logger.info("Vectorization complete | train_matrix_shape=%s", x_train_tfidf.shape)

    logistic_model = LogisticRegression(max_iter=1000, random_state=args.seed)
    svm_model = LinearSVC(random_state=args.seed)
    logistic_model.fit(x_train_tfidf, y_train)
    svm_model.fit(x_train_tfidf, y_train)
    logger.info("Trained base models: logistic_regression, linear_svm")

    val_lr_pred = logistic_model.predict(x_val_tfidf).tolist()
    val_svm_pred = svm_model.predict(x_val_tfidf).tolist()
    val_ensemble_pred = hard_vote(val_lr_pred, val_svm_pred, val_lr_pred)
    val_metrics = evaluate_predictions(y_val.tolist(), val_ensemble_pred)

    disagreement_count = sum(1 for a, b in zip(val_lr_pred, val_svm_pred) if a != b)
    total_val = len(val_lr_pred)
    disagreement_rate = float(disagreement_count / total_val) if total_val else 0.0

    result: Dict[str, object] = {
        "model_name": "ensemble_hard_vote_lr_svm",
        "voting_strategy": "hard_vote_with_lr_tie_breaker",
        "validation_metrics": val_metrics,
        "validation_disagreement_rate_lr_vs_svm": disagreement_rate,
    }

    if args.evaluate_test:
        test_lr_pred = logistic_model.predict(x_test_tfidf).tolist()
        test_svm_pred = svm_model.predict(x_test_tfidf).tolist()
        test_ensemble_pred = hard_vote(test_lr_pred, test_svm_pred, test_lr_pred)
        result["test_metrics"] = evaluate_predictions(y_test.tolist(), test_ensemble_pred)

    logger.info(
        "Validation | accuracy=%.4f macro_f1=%.4f disagreement_rate=%.4f",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
        disagreement_rate,
    )

    model_dir = ensure_dir(config.project_root / "models" / "model_a" / "traditional")
    report_dir = ensure_dir(config.project_root / "models" / "model_a" / "reports")

    artifact_path = model_dir / "ensemble_hard_vote_lr_svm.joblib"
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "logistic_regression": logistic_model,
            "linear_svm": svm_model,
            "voting_strategy": "hard_vote_with_lr_tie_breaker",
            "labels": MODEL_LABELS,
        },
        artifact_path,
    )
    logger.info("Saved ensemble artifact: %s", artifact_path)

    baseline_report = load_report(report_dir / "baseline_metrics.json")
    unsupervised_report = load_report(report_dir / "unsupervised_metrics.json")
    comparison = build_comparison_table(result["validation_metrics"], baseline_report, unsupervised_report)

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
        "ensemble": result,
        "comparison": comparison,
    }

    report_path = report_dir / "ensemble_metrics.json"
    save_json(report_path, report_payload)
    logger.info("Saved ensemble metrics report: %s", report_path)
    logger.info("Model A ensemble pipeline completed")
    return report_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)
    run_ensemble_pipeline(config, args)


if __name__ == "__main__":
    main()
