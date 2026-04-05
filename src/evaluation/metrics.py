"""
Automatic metrics for Alpaca eval responses.

Computes per-checkpoint:
  - ROUGE-1, ROUGE-2, ROUGE-L  (rouge-score)
  - BERTScore F1               (bert-score, using distilbert-base-uncased)
  - Average output length      (words)
  - Task completion rate       (heuristic: response is non-empty and >= 10 tokens)

Saves per-checkpoint results to results/metrics_ckpt{N}.json
and combined summary to results/metrics_summary.csv.

Usage:
    python src/evaluation/metrics.py --config config.yaml --checkpoint 0
    python src/evaluation/metrics.py --config config.yaml --checkpoint 0,1,2
"""

import argparse
import csv
import json
from pathlib import Path

from src.utils import get_logger, load_config


# ---------------------------------------------------------------------------
# ROUGE helpers
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list, references: list) -> dict:
    """Compute ROUGE-1/2/L scores. Returns mean F1 for each type."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    return {"rouge1": mean(r1), "rouge2": mean(r2), "rougeL": mean(rl)}


# ---------------------------------------------------------------------------
# BERTScore helpers
# ---------------------------------------------------------------------------

def compute_bertscore(predictions: list, references: list) -> float:
    """Compute BERTScore F1 (mean over examples)."""
    from bert_score import score as bert_score
    P, R, F = bert_score(
        predictions,
        references,
        model_type="distilbert-base-uncased",
        lang="en",
        verbose=False,
    )
    return float(F.mean())


# ---------------------------------------------------------------------------
# Length and completion helpers
# ---------------------------------------------------------------------------

def avg_length(responses: list) -> float:
    """Average response length in whitespace-split tokens."""
    if not responses:
        return 0.0
    return sum(len(r.split()) for r in responses) / len(responses)


def task_completion_rate(responses: list, min_tokens: int = 10) -> float:
    """Fraction of responses that are non-empty and at least min_tokens long."""
    if not responses:
        return 0.0
    completed = sum(
        1 for r in responses
        if r and len(r.split()) >= min_tokens
    )
    return completed / len(responses)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint: int, cfg: dict, logger) -> dict:
    response_file = f"{cfg['evaluation']['logs_dir']}/responses_ckpt{checkpoint}_alpaca.jsonl"
    if not Path(response_file).exists():
        logger.warning(f"Response file not found: {response_file}")
        return {}

    with open(response_file) as f:
        records = [json.loads(line) for line in f]

    predictions = [r["response"] for r in records]
    references  = [r["reference"] for r in records]

    logger.info(f"Checkpoint {checkpoint}: computing ROUGE over {len(predictions)} examples ...")
    rouge_scores = compute_rouge(predictions, references)

    logger.info(f"Checkpoint {checkpoint}: computing BERTScore ...")
    bs_f1 = compute_bertscore(predictions, references)

    avg_len = avg_length(predictions)
    completion = task_completion_rate(predictions)

    summary = {
        "checkpoint":            checkpoint,
        "n_total":               len(predictions),
        "rouge1":                rouge_scores["rouge1"],
        "rouge2":                rouge_scores["rouge2"],
        "rougeL":                rouge_scores["rougeL"],
        "bertscore_f1":          bs_f1,
        "avg_output_length":     avg_len,
        "task_completion_rate":  completion,
    }

    # Save per-checkpoint JSON
    out_dir = Path(cfg["evaluation"]["results_dir"])
    out_dir.mkdir(exist_ok=True)
    detail_path = out_dir / f"metrics_ckpt{checkpoint}.json"
    with open(detail_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Checkpoint {checkpoint} Alpaca metrics:")
    logger.info(f"  ROUGE-1:          {summary['rouge1']:.4f}")
    logger.info(f"  ROUGE-2:          {summary['rouge2']:.4f}")
    logger.info(f"  ROUGE-L:          {summary['rougeL']:.4f}")
    logger.info(f"  BERTScore F1:     {summary['bertscore_f1']:.4f}")
    logger.info(f"  Avg length:       {summary['avg_output_length']:.1f} words")
    logger.info(f"  Completion rate:  {summary['task_completion_rate']:.3f}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute automatic metrics for Alpaca responses.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="0",
                        help="Comma-separated checkpoint numbers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("metrics", "logs/metrics.log")
    checkpoints = [int(c.strip()) for c in args.checkpoint.split(",")]

    all_summaries = []
    for ckpt in checkpoints:
        summary = evaluate_checkpoint(ckpt, cfg, logger)
        if summary:
            all_summaries.append(summary)

    # Save combined CSV
    if all_summaries:
        csv_path = Path(cfg["evaluation"]["results_dir"]) / "metrics_summary.csv"
        fields = ["checkpoint", "n_total", "rouge1", "rouge2", "rougeL",
                  "bertscore_f1", "avg_output_length", "task_completion_rate"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({k: s[k] for k in fields})
        logger.info(f"Saved combined metrics summary → {csv_path}")


if __name__ == "__main__":
    main()
