"""Train and evaluate Model A stacking ensemble (LR + SVM -> Logistic Regression meta model)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import DEFAULT_CONFIG, ProjectConfig
from .model_a_inference import MODEL_LABELS, expand_mcq_rows, predict_option_labels, score_positive_class, true_labels_by_qid
from .utils import ensure_dir, save_json, set_random_seed, setup_logger, utc_timestamp


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for stacking ensemble."""
    parser = argparse.ArgumentParser(description="Train Model A stacking ensemble on processed data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument("--max-features", type=int, default=20000, help="TF-IDF max feature count.")
    parser.add_argument("--ngram-max", type=int, default=2, choices=[1, 2], help="Maximum n-gram length.")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for stacking.")
    parser.add_argument("--evaluate-test", action="store_true", help="Also evaluate on test split.")
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
    """Validate required columns for stacking training."""
    required = ["verifier_input", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split missing required columns: {missing}")


def _option_score_table(artifact: Dict[str, object], df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute one score per option and pivot to question-level feature table."""
    batch = expand_mcq_rows(df)
    x = artifact["vectorizer"].transform(batch.text)
    scores = score_positive_class(artifact["classifier"], x)
    tab = pd.DataFrame(
        {
            "qid": batch.qid,
            "option": batch.option_label,
            f"{prefix}_score": scores,
        }
    )
    wide = tab.pivot(index="qid", columns="option", values=f"{prefix}_score").reindex(columns=MODEL_LABELS)
    wide.columns = [f"{prefix}_{c}" for c in wide.columns]
    return wide.reset_index()


def build_meta_features(lr_artifact: Dict[str, object], svm_artifact: Dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    """Build question-level stacking features from base-model option scores."""
    lr_tab = _option_score_table(lr_artifact, df, "lr")
    svm_tab = _option_score_table(svm_artifact, df, "svm")
    feat = lr_tab.merge(svm_tab, on="qid", how="inner")
    return feat


def labels_for_feature_table(feat_df: pd.DataFrame, source_df: pd.DataFrame) -> List[str]:
    """Align true labels to feature rows by qid."""
    true_map = true_labels_by_qid(source_df)
    return [true_map[str(qid)] for qid in feat_df["qid"].tolist()]


def predict_meta_labels(meta_model: LogisticRegression, feat_df: pd.DataFrame) -> List[str]:
    """Predict question labels from meta features."""
    feature_cols = [c for c in feat_df.columns if c != "qid"]
    x = feat_df[feature_cols].to_numpy(dtype=np.float64)
    return meta_model.predict(x).tolist()


def evaluate_predictions(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    """Compute standard Model A classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": MODEL_LABELS,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=MODEL_LABELS).tolist(),
    }


def load_report(report_path: Path) -> Dict[str, object] | None:
    """Load report JSON if it exists."""
    if not report_path.exists():
        return None
    with report_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_comparison_table(
    stacking_val_metrics: Dict[str, object],
    baseline_report: Dict[str, object] | None,
    unsupervised_report: Dict[str, object] | None,
    ensemble_report: Dict[str, object] | None,
) -> Dict[str, object]:
    """Build validation comparison across available Model A methods."""
    table: List[Dict[str, object]] = [
        {
            "model_name": "stacking_lr_svm_meta_lr",
            "validation_accuracy": stacking_val_metrics["accuracy"],
            "validation_macro_f1": stacking_val_metrics["macro_f1"],
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

    if ensemble_report and "ensemble" in ensemble_report:
        val_metrics = ensemble_report["ensemble"].get("validation_metrics", {})
        table.append(
            {
                "model_name": "ensemble_hard_vote_lr_svm",
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


def run_stacking_pipeline(config: ProjectConfig, args: argparse.Namespace) -> Path:
    """Train stacking ensemble, evaluate, and save model/report artifacts."""
    logger = setup_logger("model_a_stacking")
    logger.info("Starting Model A stacking pipeline")
    logger.info("Random seed: %s", config.random_seed)
    set_random_seed(config.random_seed)

    train_df, val_df, test_df = load_processed_splits(config)
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_schema(df, split_name)

    logger.info("Loaded data | train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))

    model_dir = ensure_dir(config.project_root / "models" / "model_a" / "traditional")
    report_dir = ensure_dir(config.project_root / "models" / "model_a" / "reports")
    lr_path = model_dir / "logistic_regression.joblib"
    svm_path = model_dir / "linear_svm.joblib"
    if not lr_path.exists() or not svm_path.exists():
        raise FileNotFoundError(
            "Missing base Model A artifacts. Run `python -m src.model_a_train --evaluate-test` first."
        )

    lr_artifact = joblib.load(lr_path)
    svm_artifact = joblib.load(svm_path)
    if not (isinstance(lr_artifact, dict) and lr_artifact.get("kind") == "optionwise_binary"):
        raise ValueError("Unsupported logistic_regression artifact format. Re-train with current code.")
    if not (isinstance(svm_artifact, dict) and svm_artifact.get("kind") == "optionwise_binary"):
        raise ValueError("Unsupported linear_svm artifact format. Re-train with current code.")

    train_feat = build_meta_features(lr_artifact, svm_artifact, train_df)
    val_feat = build_meta_features(lr_artifact, svm_artifact, val_df)
    test_feat = build_meta_features(lr_artifact, svm_artifact, test_df)

    y_train = labels_for_feature_table(train_feat, train_df)
    y_val = labels_for_feature_table(val_feat, val_df)
    y_test = labels_for_feature_table(test_feat, test_df)

    feature_cols = [c for c in train_feat.columns if c != "qid"]
    x_train = train_feat[feature_cols].to_numpy(dtype=np.float64)
    x_val = val_feat[feature_cols].to_numpy(dtype=np.float64)
    x_test = test_feat[feature_cols].to_numpy(dtype=np.float64)

    model = LogisticRegression(max_iter=1000, random_state=args.seed)
    model.fit(x_train, y_train)
    logger.info("Trained stacking meta model")

    val_pred = model.predict(x_val).tolist()
    val_metrics = evaluate_predictions(y_val, val_pred)

    result: Dict[str, object] = {
        "model_name": "stacking_lr_svm_meta_lr",
        "stacking_cv": args.cv,
        "validation_metrics": val_metrics,
    }

    if args.evaluate_test:
        test_pred = model.predict(x_test).tolist()
        result["test_metrics"] = evaluate_predictions(y_test, test_pred)

    logger.info(
        "Validation | accuracy=%.4f macro_f1=%.4f",
        val_metrics["accuracy"],
        val_metrics["macro_f1"],
    )

    artifact_path = model_dir / "stacking_lr_svm_meta_lr.joblib"
    joblib.dump(
        {
            "kind": "meta_stacking",
            "labels": MODEL_LABELS,
            "feature_columns": feature_cols,
            "meta_model": model,
            "base_logistic_path": str(lr_path),
            "base_svm_path": str(svm_path),
        },
        artifact_path,
    )
    logger.info("Saved stacking artifact: %s", artifact_path)

    baseline_report = load_report(report_dir / "baseline_metrics.json")
    unsupervised_report = load_report(report_dir / "unsupervised_metrics.json")
    ensemble_report = load_report(report_dir / "ensemble_metrics.json")
    comparison = build_comparison_table(val_metrics, baseline_report, unsupervised_report, ensemble_report)

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
        "stacking": result,
        "comparison": comparison,
    }

    report_path = report_dir / "stacking_metrics.json"
    save_json(report_path, report_payload)
    logger.info("Saved stacking metrics report: %s", report_path)
    logger.info("Model A stacking pipeline completed")
    return report_path


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)
    run_stacking_pipeline(config, args)


if __name__ == "__main__":
    main()
