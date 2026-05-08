"""Model B progressive hint generation using extractive sentence selection."""

from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from .config import DEFAULT_CONFIG, ProjectConfig
from .model_b_utils import (
    build_model_b_paths,
    compute_cosine_to_query,
    get_correct_answer_text,
    load_processed_split,
    maybe_sample,
    normalize_text,
    split_into_sentences,
)
from .utils import save_json, set_random_seed, setup_logger, utc_timestamp


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for hint generation."""
    parser = argparse.ArgumentParser(description="Generate Model B progressive hints from processed RACE split.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Data split.")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional sample size (0 = full split).")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max feature count.")
    parser.add_argument("--max-sentences", type=int, default=30, help="Maximum article sentences to consider.")
    parser.add_argument("--report-samples", type=int, default=10, help="Number of sample rows in report.")
    return parser.parse_args()


def rank_sentences_for_hints(
    question_text: str,
    answer_text: str,
    sentences: List[str],
    max_features: int,
) -> List[Dict[str, float | str]]:
    """Rank article sentences by relevance to question and answer."""
    q_sims = compute_cosine_to_query(question_text, sentences, max_features=max_features)
    a_sims = compute_cosine_to_query(answer_text, sentences, max_features=max_features)

    ranked: List[Dict[str, float | str]] = []
    for sent, q_sim, a_sim in zip(sentences, q_sims, a_sims):
        score = (0.7 * q_sim) + (0.3 * a_sim)
        ranked.append(
            {
                "sentence": sent,
                "question_similarity": float(q_sim),
                "answer_similarity": float(a_sim),
                "score": float(score),
            }
        )
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def pick_progressive_hints(ranked: List[Dict[str, float | str]]) -> Dict[str, object]:
    """Select easy/medium/hard hints from ranked sentences."""
    if not ranked:
        return {
            "hint_easy": "",
            "hint_medium": "",
            "hint_hard": "",
            "easy_score": 0.0,
            "medium_score": 0.0,
            "hard_score": 0.0,
        }

    # Hard = highest relevance, medium/easy move gradually away from top-ranked sentence.
    hard_idx = 0
    medium_idx = min(2, len(ranked) - 1)
    easy_idx = min(4, len(ranked) - 1)

    chosen_indices: List[int] = []
    for idx in [easy_idx, medium_idx, hard_idx]:
        if idx not in chosen_indices:
            chosen_indices.append(idx)
        else:
            for alt in range(len(ranked)):
                if alt not in chosen_indices:
                    chosen_indices.append(alt)
                    break

    while len(chosen_indices) < 3:
        chosen_indices.append(chosen_indices[-1] if chosen_indices else 0)

    easy = ranked[chosen_indices[0]]
    medium = ranked[chosen_indices[1]]
    hard = ranked[chosen_indices[2]]
    return {
        "hint_easy": str(easy["sentence"]),
        "hint_medium": str(medium["sentence"]),
        "hint_hard": str(hard["sentence"]),
        "easy_score": float(easy["score"]),
        "medium_score": float(medium["score"]),
        "hard_score": float(hard["score"]),
    }


def generate_row_hints(row: pd.Series, max_features: int, max_sentences: int) -> Dict[str, object]:
    """Generate progressive hints for one QA row."""
    question = str(row["question"])
    article = str(row["article"])
    correct = str(get_correct_answer_text(row))

    sentences_raw = split_into_sentences(article)[:max_sentences]
    sentence_candidates = []
    for s in sentences_raw:
        s_norm = normalize_text(s)
        if s_norm and len(s_norm.split()) >= 5:
            sentence_candidates.append(s.strip())

    ranked = rank_sentences_for_hints(
        question_text=normalize_text(question),
        answer_text=normalize_text(correct),
        sentences=sentence_candidates,
        max_features=max_features,
    )
    hints = pick_progressive_hints(ranked)

    return {
        "id": str(row["id"]),
        "question": question,
        "correct_answer_label": str(row["answer"]).strip().upper(),
        "correct_answer_text": correct,
        **hints,
    }


def run_pipeline(config: ProjectConfig, args: argparse.Namespace) -> Dict[str, str]:
    """Run hint generation and save outputs/reports."""
    logger = setup_logger("model_b_hints")
    logger.info("Starting Model B hint generation")
    set_random_seed(config.random_seed)

    df = load_processed_split(config, split=args.split)
    df = maybe_sample(df, args.sample_size, seed=args.seed)
    logger.info("Loaded split=%s rows=%s", args.split, len(df))

    rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        rows.append(generate_row_hints(row=row, max_features=args.max_features, max_sentences=args.max_sentences))

    out_df = pd.DataFrame(rows)
    paths = build_model_b_paths(config)
    output_path = paths["outputs_dir"] / f"hints_{args.split}.csv"
    out_df.to_csv(output_path, index=False)

    n_rows = len(out_df) if len(out_df) > 0 else 1
    progressive_order_count = int(
        ((out_df["easy_score"] <= out_df["medium_score"]) & (out_df["medium_score"] <= out_df["hard_score"])).sum()
    ) if len(out_df) else 0
    report = {
        "generated_at_utc": utc_timestamp(),
        "seed": config.random_seed,
        "split": args.split,
        "row_count": len(out_df),
        "mean_easy_score": float(out_df["easy_score"].mean()) if len(out_df) else 0.0,
        "mean_medium_score": float(out_df["medium_score"].mean()) if len(out_df) else 0.0,
        "mean_hard_score": float(out_df["hard_score"].mean()) if len(out_df) else 0.0,
        "progressive_order_rate": float(progressive_order_count / n_rows),
        "sample_outputs": out_df.head(args.report_samples).to_dict(orient="records"),
        "output_file": str(output_path),
    }
    report_path = paths["reports_dir"] / f"hints_report_{args.split}.json"
    save_json(report_path, report)

    logger.info("Saved hint outputs: %s", output_path)
    logger.info("Saved hint report: %s", report_path)
    logger.info("Model B hint generation completed")
    return {"output_path": str(output_path), "report_path": str(report_path)}


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)
    run_pipeline(config, args)


if __name__ == "__main__":
    main()
