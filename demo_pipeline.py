"""
End-to-end demo pipeline (Model A + Model B) for one processed sample.

Run (recommended on Windows):
  .\\venv\\Scripts\\python demo_pipeline.py --split val

Optional:
  .\\venv\\Scripts\\python demo_pipeline.py --split val --row-idx 5
  .\\venv\\Scripts\\python demo_pipeline.py --split val --id high5060.txt
  .\\venv\\Scripts\\python demo_pipeline.py --skip-model-a
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.config import DEFAULT_CONFIG, ProjectConfig
from src.model_a_inference import predict_single_row_label
from src.model_b_distractors import generate_row_distractors
from src.model_b_hints import generate_row_hints
from src.model_b_utils import load_processed_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo: load 1 processed sample and run Model A + Model B.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Processed split name.")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Seed for random sampling.")
    p.add_argument("--row-idx", type=int, default=-1, help="Row index to use (>=0). If -1, pick a random row.")
    p.add_argument("--id", type=str, default="", help="Example id to fetch (overrides --row-idx if found).")

    p.add_argument("--skip-model-a", action="store_true", help="Skip Model A verifier prediction.")
    p.add_argument("--model-a-artifact", type=str, default="models/model_a/traditional/linear_svm.joblib")

    p.add_argument("--max-features-b", type=int, default=5000, help="TF-IDF max features for Model B.")
    p.add_argument("--diversity-threshold", type=float, default=0.85, help="Model B distractor diversity threshold.")
    p.add_argument("--max-sentences", type=int, default=20, help="Max sentences for distractors.")
    p.add_argument("--max-sentences-hints", type=int, default=30, help="Max sentences for hints.")
    p.add_argument("--wrap", type=int, default=100, help="Text wrap width for terminal output.")
    return p.parse_args()


def hr(char: str = "=", n: int = 90) -> str:
    return char * n


def fmt_block(title: str, text: str, width: int) -> str:
    text = "" if text is None else str(text)
    wrapped = textwrap.fill(text, width=width, replace_whitespace=False)
    return f"{title}\n{wrapped}\n"


def safe_get(row: Any, key: str, default: str = "") -> str:
    try:
        val = row[key]
    except Exception:
        return default
    return default if val is None else str(val)


def pick_row(df, args: argparse.Namespace):
    if args.id:
        matches = df.index[df["id"].astype(str) == str(args.id)].tolist() if "id" in df.columns else []
        if matches:
            return df.loc[matches[0]]
    if args.row_idx is not None and args.row_idx >= 0:
        return df.iloc[int(args.row_idx)]
    return df.sample(n=1, random_state=args.seed).iloc[0]


def run_model_a(row, artifact_path: Path) -> Dict[str, str]:
    if not artifact_path.exists():
        return {
            "status": "missing_artifact",
            "message": f"Model A artifact not found at: {artifact_path}. Train it first: `.\\venv\\Scripts\\python -m src.model_a_train`",
        }

    if "verifier_input" not in row.index:
        return {"status": "missing_column", "message": "Processed row has no `verifier_input` column."}

    model = joblib.load(artifact_path)
    if isinstance(model, dict) and model.get("kind") == "optionwise_binary":
        pred_label = predict_single_row_label(model, row)
    else:
        pred_label = str(model.predict([str(row["verifier_input"])])[0]).strip().upper()
    pred_text = safe_get(row, pred_label, default="")
    return {"status": "ok", "pred_label": pred_label, "pred_text": pred_text}


def main() -> None:
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)

    df = load_processed_split(config, split=args.split)
    row = pick_row(df, args)

    article = safe_get(row, "article")
    question = safe_get(row, "question")
    options = {k: safe_get(row, k) for k in ["A", "B", "C", "D"]}
    correct_label = safe_get(row, "answer").strip().upper()
    correct_text = options.get(correct_label, "")

    print(hr("="))
    print(f"DEMO PIPELINE | split={args.split} | id={safe_get(row, 'id', 'N/A')}")
    print(hr("="))
    print(fmt_block("ARTICLE:", article, args.wrap))
    print(fmt_block("QUESTION:", question, args.wrap))

    print("OPTIONS:")
    for k in ["A", "B", "C", "D"]:
        print(f"  {k}) {textwrap.fill(options[k], width=args.wrap, subsequent_indent='     ')}")
    print()

    print(f"CORRECT ANSWER: {correct_label}) {correct_text}")
    print(hr("-"))

    if not args.skip_model_a:
        artifact = config.project_root / args.model_a_artifact
        a = run_model_a(row, artifact)
        print("MODEL A (Verifier):")
        if a["status"] == "ok":
            print(f"  Predicted: {a['pred_label']}) {a['pred_text']}")
        else:
            print(f"  Skipped: {a['message']}")
        print(hr("-"))

    print("MODEL B (Distractors):")
    d = generate_row_distractors(
        row=row,
        max_features=args.max_features_b,
        diversity_threshold=args.diversity_threshold,
        max_sentences=args.max_sentences,
    )
    print(f"  1) {d['distractor_1']}")
    print(f"  2) {d['distractor_2']}")
    print(f"  3) {d['distractor_3']}")
    print(hr("-"))

    print("MODEL B (Progressive Hints):")
    h = generate_row_hints(row=row, max_features=args.max_features_b, max_sentences=args.max_sentences_hints)
    print(fmt_block("  Easy:", h.get("hint_easy", ""), args.wrap).rstrip())
    print(fmt_block("  Medium:", h.get("hint_medium", ""), args.wrap).rstrip())
    print(fmt_block("  Hard:", h.get("hint_hard", ""), args.wrap).rstrip())
    print(hr("="))


if __name__ == "__main__":
    main()

