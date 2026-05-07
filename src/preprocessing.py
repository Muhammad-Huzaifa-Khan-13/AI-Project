"""CLI preprocessing pipeline for RACE data."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import DEFAULT_CONFIG, ProjectConfig
from .data_loader import load_all_splits
from .utils import ensure_dir, save_json, set_random_seed, setup_logger, utc_timestamp

PUNCTUATION_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Normalize text with lowercase, punctuation removal, and whitespace cleanup."""
    if text is None:
        return ""

    cleaned = str(text).lower()
    cleaned = PUNCTUATION_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def add_clean_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Add cleaned versions of text columns using the `clean_` prefix."""
    out = df.copy()
    for col in columns:
        out[f"clean_{col}"] = out[col].apply(clean_text)
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived columns required by Model A."""
    out = df.copy()
    out["verifier_input"] = (
        out["clean_article"]
        + " [SEP] "
        + out["clean_question"]
        + " [SEP] "
        + out["clean_A"]
        + " [SEP] "
        + out["clean_B"]
        + " [SEP] "
        + out["clean_C"]
        + " [SEP] "
        + out["clean_D"]
    )
    out["question_context"] = out["clean_article"] + " [SEP] " + out["clean_question"]
    return out


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """Apply complete preprocessing transformations to one split."""
    text_columns = ["article", "question", "A", "B", "C", "D"]
    out = add_clean_columns(df, text_columns)
    out = add_derived_columns(out)
    return out


def save_processed_split(df: pd.DataFrame, split_name: str, config: ProjectConfig) -> Path:
    """Save processed split to `data/processed/` and return output path."""
    ensure_dir(config.processed_data_dir)
    output_path = config.processed_data_dir / f"{split_name}_processed.csv"
    df.to_csv(output_path, index=False)
    return output_path


def build_report(
    config: ProjectConfig,
    load_report: Dict[str, Dict[str, object]],
    output_paths: Dict[str, Path],
) -> Dict[str, object]:
    """Build JSON-serializable preprocessing report."""
    return {
        "generated_at_utc": utc_timestamp(),
        "random_seed": config.random_seed,
        "required_columns": config.required_columns,
        "answer_labels": config.answer_labels,
        "splits": {
            split: {
                **load_report[split],
                "output_file": str(output_paths[split]),
            }
            for split in load_report
        },
    }


def run_pipeline(config: ProjectConfig) -> Path:
    """Run full load-validate-preprocess-save pipeline and return report path."""
    logger = setup_logger()
    logger.info("Starting preprocessing pipeline")
    logger.info("Using random seed: %s", config.random_seed)
    set_random_seed(config.random_seed)

    ensure_dir(config.processed_data_dir)
    ensure_dir(config.reports_dir)

    splits, load_report = load_all_splits(config)
    output_paths: Dict[str, Path] = {}

    for split_name, split_df in splits.items():
        logger.info("Preprocessing split: %s (rows=%s)", split_name, len(split_df))
        processed = preprocess_split(split_df)
        output_path = save_processed_split(processed, split_name, config)
        output_paths[split_name] = output_path
        logger.info("Saved processed split: %s", output_path)

    report_payload = build_report(config, load_report, output_paths)
    report_path = config.reports_dir / "preprocessing_report.json"
    save_json(report_path, report_payload)
    logger.info("Saved preprocessing report: %s", report_path)
    logger.info("Pipeline completed successfully")

    return report_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run RACE preprocessing pipeline.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)
    run_pipeline(config)


if __name__ == "__main__":
    main()
