from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import ProjectConfig
from src.model_b_utils import load_processed_split

LABELS = ["A", "B", "C", "D"]

# Premium dark theme chart colors (matches UI)
_CHART_BG = "#0d1117"
_CHART_AX = "#161b22"
_CHART_TEXT = "#e6edf3"
_CHART_GRID = "#30363d"
_BAR_GRADIENT = ["#58a6ff", "#a371f7", "#3fb950", "#d2a8ff", "#79c0ff"]


@st.cache_data(show_spinner=False)
def load_split_cached(split: str) -> pd.DataFrame:
    config = ProjectConfig()
    return load_processed_split(config, split=split)


def wrap_text(text: str, width: int = 110) -> str:
    return textwrap.fill(str(text), width=width, replace_whitespace=False)


def get_row_by_selector(df: pd.DataFrame, *, mode: str, row_idx: int, example_id: str, seed: int):
    if mode == "By ID" and example_id:
        matches = df.index[df["id"].astype(str) == str(example_id)].tolist() if "id" in df.columns else []
        if matches:
            return df.loc[matches[0]]
    if mode == "By Index":
        row_idx = max(0, min(int(row_idx), len(df) - 1))
        return df.iloc[row_idx]
    return df.sample(n=1, random_state=seed).iloc[0]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def confusion_matrix_figure(cm: List[List[int]], labels: List[str], title: str):
    arr = np.array(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(5.8, 4.6), facecolor=_CHART_BG)
    ax.set_facecolor(_CHART_AX)
    im = ax.imshow(arr, interpolation="nearest", cmap="plasma", vmin=0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.label.set_color(_CHART_TEXT)
    cbar.ax.tick_params(colors=_CHART_TEXT)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    ax.xaxis.label.set_color(_CHART_TEXT)
    ax.yaxis.label.set_color(_CHART_TEXT)
    ax.title.set_color(_CHART_TEXT)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", color=_CHART_TEXT)
    plt.setp(ax.get_yticklabels(), color=_CHART_TEXT)
    ax.tick_params(colors=_CHART_TEXT)
    ax.spines[:].set_color(_CHART_GRID)

    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                int(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "#0d1117",
                fontsize=9,
                fontweight="600",
            )
    fig.tight_layout()
    return fig


def model_comparison_bar_figure(rows: List[Dict[str, Any]], metric: str):
    names = [r["model_name"] for r in rows]
    values = [float(r.get(metric) or 0.0) for r in rows]
    colors = [_BAR_GRADIENT[i % len(_BAR_GRADIENT)] for i in range(len(names))]

    fig, ax = plt.subplots(figsize=(8.0, 4.0), facecolor=_CHART_BG)
    ax.set_facecolor(_CHART_AX)
    bars = ax.bar(names, values, color=colors, edgecolor=_CHART_GRID, linewidth=0.8)
    for b in bars:
        b.set_linewidth(0.8)

    ylabel = metric.replace("_", " ").replace("validation ", "").title()
    ax.set_ylabel(ylabel, color=_CHART_TEXT, fontsize=11)
    ymax = max(0.35, max(values) + 0.06) if values else 0.35
    ax.set_ylim(0.0, ymax)
    ax.set_title(f"Model comparison — {ylabel}", color=_CHART_TEXT, fontsize=12, fontweight="600", pad=12)
    ax.grid(axis="y", alpha=0.35, color=_CHART_GRID)
    ax.tick_params(axis="x", colors=_CHART_TEXT, labelsize=9)
    ax.tick_params(axis="y", colors=_CHART_TEXT, labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    ax.spines[:].set_color(_CHART_GRID)

    fig.tight_layout()
    return fig
