"""Project configuration for data loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration used by preprocessing modules."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    random_seed: int = 42
    required_columns: List[str] = field(
        default_factory=lambda: ["id", "article", "question", "A", "B", "C", "D", "answer"]
    )
    answer_labels: List[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    split_files: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "train.csv",
            "test": "test.csv",
            # "val" is required by project spec; "dev" accepted as fallback for local datasets.
            "val": "val.csv",
        }
    )
    fallback_split_files: Dict[str, str] = field(default_factory=lambda: {"val": "dev.csv"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_data_dir", self.project_root / "data" / "raw")
        object.__setattr__(self, "processed_data_dir", self.project_root / "data" / "processed")
        object.__setattr__(self, "reports_dir", self.processed_data_dir / "reports")


DEFAULT_CONFIG = ProjectConfig()
