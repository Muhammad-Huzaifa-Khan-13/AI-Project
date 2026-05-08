"""Model B distractor generation using TF-IDF cosine similarity."""

from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd

from .config import DEFAULT_CONFIG, ProjectConfig
from .model_b_utils import (
    build_model_b_paths,
    compute_cosine_to_query,
    cosine_between_texts,
    get_correct_answer_text,
    get_wrong_option_texts,
    load_processed_split,
    maybe_sample,
    normalize_text,
    split_into_sentences,
)
from .utils import save_json, set_random_seed, setup_logger, utc_timestamp


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for distractor generation."""
    parser = argparse.ArgumentParser(description="Generate Model B distractors from processed RACE split.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.random_seed, help="Random seed value.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Data split.")
    parser.add_argument("--sample-size", type=int, default=0, help="Optional sample size (0 = full split).")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max features.")
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=0.85,
        help="Maximum cosine similarity allowed between selected distractors.",
    )
    parser.add_argument("--max-sentences", type=int, default=20, help="Maximum article sentences to consider.")
    parser.add_argument("--report-samples", type=int, default=10, help="Number of sample rows in report.")
    return parser.parse_args()


def score_candidates(
    question_text: str,
    correct_text: str,
    candidates: List[str],
    max_features: int,
) -> List[Dict[str, float | str]]:
    """Compute combined ranking score for distractor candidates."""
    q_sims = compute_cosine_to_query(question_text, candidates, max_features=max_features)
    a_sims = compute_cosine_to_query(correct_text, candidates, max_features=max_features)

    scored: List[Dict[str, float | str]] = []
    for cand, q_sim, a_sim in zip(candidates, q_sims, a_sims):
        # Target moderate similarity to correct answer (plausible but still wrong).
        answer_balance = 1.0 - abs(a_sim - 0.35)
        score = (0.6 * q_sim) + (0.4 * answer_balance)
        scored.append(
            {
                "text": cand,
                "question_similarity": float(q_sim),
                "answer_similarity": float(a_sim),
                "score": float(score),
            }
        )
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def select_diverse_top3(
    ranked: List[Dict[str, float | str]],
    diversity_threshold: float,
) -> List[Dict[str, float | str]]:
    """Select top-3 candidates with diversity filtering."""
    selected: List[Dict[str, float | str]] = []
    for item in ranked:
        text = str(item["text"])
        keep = True
        for chosen in selected:
            sim = cosine_between_texts(text, str(chosen["text"]))
            if sim >= diversity_threshold:
                keep = False
                break
        if keep:
            selected.append(item)
        if len(selected) == 3:
            break
    return selected


def generate_row_distractors(
    row: pd.Series,
    max_features: int,
    diversity_threshold: float,
    max_sentences: int,
) -> Dict[str, object]:
    """Generate 3 distractors for one QA row."""
    question = str(row["question"])
    article = str(row["article"])
    correct = str(get_correct_answer_text(row))
    wrong_options = [normalize_text(x) for x in get_wrong_option_texts(row)]
    correct_norm = normalize_text(correct)
    question_norm = normalize_text(question)

    sentences = split_into_sentences(article)[:max_sentences]
    sentence_candidates = []
    for sent in sentences:
        sent_norm = normalize_text(sent)
        if sent_norm and len(sent_norm.split()) >= 3 and sent_norm != correct_norm:
            sentence_candidates.append(sent_norm)

    # Candidate pool = article sentences + existing wrong options.
    candidate_pool = list(dict.fromkeys(sentence_candidates + wrong_options))
    candidate_pool = [c for c in candidate_pool if c and c != correct_norm]

    if not candidate_pool:
        candidate_pool = [w for w in wrong_options if w and w != correct_norm]

    ranked = score_candidates(
        question_text=question_norm,
        correct_text=correct_norm,
        candidates=candidate_pool,
        max_features=max_features,
    )
    selected = select_diverse_top3(ranked, diversity_threshold=diversity_threshold)

    # Fallback: ensure always 3 distractors by adding remaining wrong options/ranked candidates.
    selected_texts = [str(s["text"]) for s in selected]
    for fallback in wrong_options + [str(x["text"]) for x in ranked]:
        if fallback and fallback != correct_norm and fallback not in selected_texts:
            selected_texts.append(fallback)
        if len(selected_texts) == 3:
            break

    while len(selected_texts) < 3:
        selected_texts.append(f"alternative option {len(selected_texts) + 1}")

    return {
        "id": str(row["id"]),
        "question": question,
        "correct_answer_label": str(row["answer"]).strip().upper(),
        "correct_answer_text": correct,
        "distractor_1": selected_texts[0],
        "distractor_2": selected_texts[1],
        "distractor_3": selected_texts[2],
        "original_wrong_options": wrong_options,
    }


def run_pipeline(config: ProjectConfig, args: argparse.Namespace) -> Dict[str, str]:
    """Run distractor generation and save outputs/reports."""
    logger = setup_logger("model_b_distractors")
    logger.info("Starting Model B distractor generation")
    set_random_seed(config.random_seed)

    df = load_processed_split(config, split=args.split)
    df = maybe_sample(df, args.sample_size, seed=args.seed)
    logger.info("Loaded split=%s rows=%s", args.split, len(df))

    rows: List[Dict[str, object]] = []
    contains_correct_count = 0
    unique_top3_count = 0
    overlap_scores: List[float] = []

    for _, row in df.iterrows():
        result = generate_row_distractors(
            row=row,
            max_features=args.max_features,
            diversity_threshold=args.diversity_threshold,
            max_sentences=args.max_sentences,
        )
        rows.append(result)

        generated = [result["distractor_1"], result["distractor_2"], result["distractor_3"]]
        generated_norm = [normalize_text(x) for x in generated]
        correct_norm = normalize_text(result["correct_answer_text"])
        wrong_norm = [normalize_text(x) for x in result["original_wrong_options"]]

        if correct_norm in generated_norm:
            contains_correct_count += 1
        if len(set(generated_norm)) == 3:
            unique_top3_count += 1

        overlap = len(set(generated_norm).intersection(set(wrong_norm))) / 3.0
        overlap_scores.append(overlap)

    out_df = pd.DataFrame(rows)
    paths = build_model_b_paths(config)
    output_path = paths["outputs_dir"] / f"distractors_{args.split}.csv"
    out_df.to_csv(output_path, index=False)

    n_rows = len(out_df) if len(out_df) > 0 else 1
    report = {
        "generated_at_utc": utc_timestamp(),
        "seed": config.random_seed,
        "split": args.split,
        "row_count": len(out_df),
        "contains_correct_rate": contains_correct_count / n_rows,
        "unique_top3_rate": unique_top3_count / n_rows,
        "mean_overlap_with_original_wrong_options": float(sum(overlap_scores) / n_rows),
        "sample_outputs": out_df.head(args.report_samples).to_dict(orient="records"),
        "output_file": str(output_path),
    }

    report_path = paths["reports_dir"] / f"distractors_report_{args.split}.json"
    save_json(report_path, report)
    logger.info("Saved distractor outputs: %s", output_path)
    logger.info("Saved distractor report: %s", report_path)
    logger.info("Model B distractor generation completed")
    return {"output_path": str(output_path), "report_path": str(report_path)}


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = ProjectConfig(random_seed=args.seed)
    run_pipeline(config, args)


if __name__ == "__main__":
    main()
