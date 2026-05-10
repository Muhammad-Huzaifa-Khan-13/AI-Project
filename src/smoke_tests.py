"""Quick, beginner-friendly smoke tests for Models A and B.

Run this after preprocessing and after you have Model A artifacts saved.

Examples:
  venv\\Scripts\\python -m src.smoke_tests --split val --n 200
  venv\\Scripts\\python -m src.smoke_tests --split val --n 200 --skip-model-a
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import DEFAULT_CONFIG, ProjectConfig
from .model_a_inference import predict_df_labels
from .model_b_distractors import generate_row_distractors
from .model_b_hints import generate_row_hints
from .model_b_utils import load_processed_split, normalize_text
from .utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for Model A and Model B on processed data.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Data split.")
    parser.add_argument("--n", type=int, default=200, help="Number of rows to test (default: 200).")
    parser.add_argument("--skip-model-a", action="store_true", help="Skip Model A checks.")
    parser.add_argument("--skip-model-b", action="store_true", help="Skip Model B checks.")
    parser.add_argument(
        "--model-a-artifact",
        type=str,
        default="models/model_a/traditional/linear_svm.joblib",
        help="Path to a saved Model A pipeline artifact.",
    )
    parser.add_argument("--max-features-b", type=int, default=5000, help="TF-IDF max features for Model B checks.")
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.85,
        help="Maximum cosine similarity allowed between selected distractors (Model B).",
    )
    parser.add_argument("--max-sentences", type=int, default=20, help="Max article sentences for distractors test.")
    parser.add_argument("--max-sentences-hints", type=int, default=30, help="Max article sentences for hints test.")
    return parser.parse_args()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def smoke_test_model_a(config: ProjectConfig, split: str, n: int, artifact_path: Path) -> Dict[str, float]:
    logger = setup_logger("smoke_tests.model_a")
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model A artifact not found: {artifact_path}. Train first (e.g. `python -m src.model_a_train`)."
        )

    df = load_processed_split(config, split=split)
    _assert("verifier_input" in df.columns and "answer" in df.columns, "Processed split missing verifier_input/answer.")
    df = df.head(max(1, n)).reset_index(drop=True)

    model = joblib.load(artifact_path)
    y_true = df["answer"].astype(str).str.strip().str.upper()
    if isinstance(model, dict) and model.get("kind") == "optionwise_binary":
        _, y_true_list, y_pred_list = predict_df_labels(model, df)
        y_true = pd.Series(y_true_list)
        y_pred = pd.Series(y_pred_list)
    else:
        x = df["verifier_input"].astype(str)
        y_pred = pd.Series(model.predict(x)).astype(str).str.strip().str.upper()

    allowed = {"A", "B", "C", "D"}
    _assert(y_pred.isin(list(allowed)).all(), "Model A produced invalid labels outside A/B/C/D.")

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    logger.info("Model A OK | rows=%s split=%s accuracy=%.4f macro_f1=%.4f", len(df), split, acc, macro_f1)
    return {"accuracy": acc, "macro_f1": macro_f1}


def _check_distractor_invariants(result: Dict[str, object]) -> Tuple[bool, str]:
    correct_norm = normalize_text(result.get("correct_answer_text", ""))
    distractors = [result.get("distractor_1", ""), result.get("distractor_2", ""), result.get("distractor_3", "")]
    d_norm = [normalize_text(x) for x in distractors]

    if any(not t.strip() for t in map(str, distractors)):
        return False, "Empty distractor text produced."
    if correct_norm and correct_norm in d_norm:
        return False, "Correct answer appeared inside distractors."
    if len(set(d_norm)) != 3:
        return False, "Duplicate distractor in top-3."
    return True, ""


def smoke_test_model_b(config: ProjectConfig, split: str, n: int, args: argparse.Namespace) -> Dict[str, float]:
    logger = setup_logger("smoke_tests.model_b")
    df = load_processed_split(config, split=split)
    df = df.head(max(1, n)).reset_index(drop=True)

    distractor_fail = 0
    hint_empty_count = 0
    for _, row in df.iterrows():
        d = generate_row_distractors(
            row=row,
            max_features=args.max_features_b,
            diversity_threshold=args.diversity_threshold,
            max_sentences=args.max_sentences,
        )
        ok, msg = _check_distractor_invariants(d)
        if not ok:
            distractor_fail += 1
            logger.warning("Distractor invariant failed for id=%s: %s", d.get("id"), msg)

        h = generate_row_hints(row=row, max_features=args.max_features_b, max_sentences=args.max_sentences_hints)
        if not str(h.get("hint_easy", "")).strip() and not str(h.get("hint_medium", "")).strip() and not str(
            h.get("hint_hard", "")
        ).strip():
            hint_empty_count += 1

    fail_rate = float(distractor_fail / len(df)) if len(df) else 0.0
    hint_all_empty_rate = float(hint_empty_count / len(df)) if len(df) else 0.0

    _assert(distractor_fail == 0, f"Model B distractor test failed on {distractor_fail}/{len(df)} rows.")
    logger.info(
        "Model B OK | rows=%s split=%s distractor_fail_rate=%.4f hint_all_empty_rate=%.4f",
        len(df),
        split,
        fail_rate,
        hint_all_empty_rate,
    )
    return {"distractor_fail_rate": fail_rate, "hint_all_empty_rate": hint_all_empty_rate}


def main() -> None:
    args = parse_args()
    logger = setup_logger("smoke_tests")
    config = ProjectConfig(random_seed=args.seed)

    logger.info("Starting smoke tests | split=%s n=%s", args.split, args.n)
    results: Dict[str, object] = {}

    if not args.skip_model_a:
        results["model_a"] = smoke_test_model_a(
            config=config,
            split=args.split,
            n=args.n,
            artifact_path=(config.project_root / args.model_a_artifact),
        )

    if not args.skip_model_b:
        results["model_b"] = smoke_test_model_b(config=config, split=args.split, n=args.n, args=args)

    logger.info("All smoke tests passed")
    # Keep final output short and readable for CLI usage.
    print(results)


if __name__ == "__main__":
    main()

