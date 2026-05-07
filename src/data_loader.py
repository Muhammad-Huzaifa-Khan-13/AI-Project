"""Data loading and validation utilities for RACE dataset splits."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .config import ProjectConfig


def resolve_split_path(config: ProjectConfig, split_name: str) -> Path:
    """Resolve split file path using required and fallback names."""
    filename = config.split_files[split_name]
    preferred = config.raw_data_dir / filename
    if preferred.exists():
        return preferred

    fallback_filename = config.fallback_split_files.get(split_name)
    if fallback_filename:
        fallback = config.raw_data_dir / fallback_filename
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"Missing split file for '{split_name}'. Expected '{filename}'"
        + (f" or fallback '{fallback_filename}'." if fallback_filename else ".")
    )


def remove_unwanted_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns that are not required for modeling."""
    candidates = [col for col in df.columns if col.startswith("Unnamed")]
    if not candidates:
        return df, []
    return df.drop(columns=candidates), candidates


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate required schema exists."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")


def validate_answer_labels(df: pd.DataFrame, answer_labels: List[str]) -> None:
    """Validate answer column only contains expected labels."""
    normalized = df["answer"].astype(str).str.strip().str.upper()
    invalid = sorted(set(normalized.unique()) - set(answer_labels))
    if invalid:
        raise ValueError(
            f"Answer label validation failed. Found invalid labels: {invalid}. "
            f"Allowed labels: {answer_labels}"
        )


def remove_missing_required_rows(df: pd.DataFrame, required_columns: List[str]) -> Tuple[pd.DataFrame, int]:
    """Drop rows where required fields are null/empty."""
    clean_df = df.copy()
    for col in required_columns:
        clean_df[col] = clean_df[col].astype("string")
        clean_df[col] = clean_df[col].str.strip()
        clean_df[col] = clean_df[col].replace("", pd.NA)

    before_count = len(clean_df)
    clean_df = clean_df.dropna(subset=required_columns)
    dropped = before_count - len(clean_df)
    return clean_df, dropped


def load_split(config: ProjectConfig, split_name: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Load one split and return dataframe plus quality stats."""
    split_path = resolve_split_path(config, split_name)
    df = pd.read_csv(split_path)

    initial_rows = len(df)
    df, removed_columns = remove_unwanted_columns(df)
    validate_schema(df, config.required_columns)
    df, dropped_missing_rows = remove_missing_required_rows(df, config.required_columns)
    validate_answer_labels(df, config.answer_labels)

    # Standardize answer labels once validated.
    df["answer"] = df["answer"].astype(str).str.strip().str.upper()

    stats: Dict[str, object] = {
        "split_name": split_name,
        "source_file": split_path.name,
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "dropped_missing_rows": dropped_missing_rows,
        "removed_columns": removed_columns,
    }
    return df.reset_index(drop=True), stats


def load_all_splits(config: ProjectConfig) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    """Load train/test/val splits and return dataframes + validation stats."""
    splits: Dict[str, pd.DataFrame] = {}
    report: Dict[str, Dict[str, object]] = {}

    for split_name in config.split_files:
        split_df, stats = load_split(config, split_name)
        splits[split_name] = split_df
        report[split_name] = stats

    return splits, report
