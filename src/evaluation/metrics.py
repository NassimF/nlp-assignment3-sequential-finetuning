"""
Stage 5 — Automatic metrics for Alpaca evaluation.

Computes per-checkpoint:
  - ROUGE-1, ROUGE-2, ROUGE-L   (rouge-score library)
  - BERTScore                   (bert-score library)
  - Average output length       (tokens)
  - Task completion rate        (did the model follow the instruction at all)

Aggregates all metrics (Alpaca + JSON + judge win rates) into:
  results/checkpoint_comparison_table.csv

Also produces forgetting analysis:
  - Absolute change in Alpaca judge win rate from ckpt1 → ckpt2
  - Absolute change in ROUGE-L and BERTScore from ckpt1 → ckpt2
  - Per-category breakdown

Implemented in Stage 5.
"""
# TODO: implement in Stage 5
