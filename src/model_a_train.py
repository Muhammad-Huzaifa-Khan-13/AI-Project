"""Train Model A baselines (Logistic Regression and SVM) for answer verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import DEFAULT_CONFIG, ProjectConfig
from .model_a_inference import MODEL_LABELS, expand_mcq_rows, predict_option_labels, score_positive_class, true_labels_by_qid
from .utils import ensure_dir, save_json, set_random_seed, setup_logger, utc_timestamp


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Model A training."""
    parser = argparse.ArgumentParser(description="Train Model A baselines on processed RACE data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="TF-IDF max features. Lower if you hit MemoryError.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=1,
        choices=[1, 2],
        help="Max n-gram (2 = word bigrams, uses much more RAM on large passages). Default 1 is safer.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=5,
        help="Min document frequency for TF-IDF rows. Higher trims rare tokens and saves RAM.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Force a small vocab (6000 features, unigrams, min_df>=10). Use if MemoryError persists.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate trained models on test split (in addition to validation).",
    )
    return parser.parse_args()


def resolve_memory_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Apply preset for machines that OOM during TF-IDF vocabulary build."""
    if args.low_memory:
        args.max_features = 6000
        args.ngram_max = 1
        args.min_df = max(int(args.min_df), 10)
    return args


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

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def validate_processed_schema(df: pd.DataFrame, split_name: str) -> None:
    """Validate required processed columns for model training."""
    required = ["verifier_input", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split missing required columns: {missing}")


def prepare_binary_xy(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """Return option-wise text and binary targets (correct option vs others)."""
    batch = expand_mcq_rows(df)
    return batch.text, batch.is_correct


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    """Build TF-IDF vectorizer for option-wise training."""
    return TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_logreg_classifier(args: argparse.Namespace) -> LogisticRegression:
    """Build binary logistic regression classifier."""
    return LogisticRegression(max_iter=1000, random_state=args.seed)


def build_svm_classifier(args: argparse.Namespace) -> LinearSVC:
    """Build binary linear SVM classifier."""
    return LinearSVC(random_state=args.seed)


def evaluate_split(artifact: Dict[str, object], df: pd.DataFrame) -> Dict[str, object]:
    """Evaluate model at question level by picking top-scored option."""
    batch = expand_mcq_rows(df)
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    x = vectorizer.transform(batch.text)
    scores = score_positive_class(classifier, x)

    pred_qids, y_pred = predict_option_labels(batch.qid, batch.option_label, scores)
    true_map = true_labels_by_qid(df)
    y_true = [true_map[qid] for qid in pred_qids]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": MODEL_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=MODEL_LABELS).tolist(),
    }


def train_and_evaluate(
    name: str,
    artifact: Dict[str, object],
    x_train: List[str],
    y_train: np.ndarray,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    evaluate_test: bool,
) -> Dict[str, object]:
    """Train one model and return training/evaluation summary."""
    vectorizer = artifact["vectorizer"]
    classifier = artifact["classifier"]
    x_train_tfidf = vectorizer.fit_transform(x_train)
    classifier.fit(x_train_tfidf, y_train)
    result: Dict[str, object] = {"model_name": name, "validation_metrics": evaluate_split(artifact, val_df)}
    if evaluate_test:
        result["test_metrics"] = evaluate_split(artifact, test_df)
    return result


def run_training(config: ProjectConfig, args: argparse.Namespace) -> Path:
    """Run complete Model A baseline training and persist artifacts."""
    logger = setup_logger("model_a_train")
    logger.info("Starting Model A baseline training")
    logger.info("Random seed: %s", config.random_seed)
    set_random_seed(config.random_seed)

    logger.info(
        "TF-IDF settings | max_features=%s ngram_range=(1,%s) min_df=%s low_memory=%s",
        args.max_features,
        args.ngram_max,
        args.min_df,
        getattr(args, "low_memory", False),
    )

    train_df, val_df, test_df = load_processed_splits(config)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_processed_schema(df, split_name)

    x_train, y_train = prepare_binary_xy(train_df)
    logger.info("Loaded data | train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))

    model_dir = ensure_dir(config.project_root / "models" / "model_a" / "traditional")
    report_dir = ensure_dir(config.project_root / "models" / "model_a" / "reports")

    candidates: Dict[str, Dict[str, object]] = {
        "logistic_regression": {
            "kind": "optionwise_binary",
            "labels": MODEL_LABELS,
            "vectorizer": build_vectorizer(args),
            "classifier": build_logreg_classifier(args),
        },
        "linear_svm": {
            "kind": "optionwise_binary",
            "labels": MODEL_LABELS,
            "vectorizer": build_vectorizer(args),
            "classifier": build_svm_classifier(args),
        },
    }

    metrics_report: Dict[str, object] = {
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
            "dtype": "float32",
        },
        "models": {},
    }

    for model_name, artifact in candidates.items():
        logger.info("Training model: %s", model_name)
        result = train_and_evaluate(
            name=model_name,
            artifact=artifact,
            x_train=x_train,
            y_train=y_train,
            val_df=val_df,
            test_df=test_df,
            evaluate_test=args.evaluate_test,
        )
        metrics_report["models"][model_name] = result

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(artifact, model_path)
        logger.info("Saved model: %s", model_path)
        logger.info(
            "%s | val_accuracy=%.4f val_macro_f1=%.4f",
            model_name,
            result["validation_metrics"]["accuracy"],
            result["validation_metrics"]["macro_f1"],
        )

    report_path = report_dir / "baseline_metrics.json"
    save_json(report_path, metrics_report)
    logger.info("Saved metrics report: %s", report_path)
    logger.info("Model A baseline training completed")
    return report_path


def main() -> None:
    """CLI entrypoint for Model A training."""
    args = resolve_memory_defaults(parse_args())
    config = ProjectConfig(random_seed=args.seed)
    run_training(config, args)


if __name__ == "__main__":
    main()
