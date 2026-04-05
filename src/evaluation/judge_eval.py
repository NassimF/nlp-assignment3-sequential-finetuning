"""
Stage 5 — LLM-as-judge pairwise evaluation.

Runs all three pairwise comparisons using Llama 3.1 70B Instruct as judge:
  - Checkpoint 0 vs Checkpoint 1
  - Checkpoint 1 vs Checkpoint 2
  - Checkpoint 0 vs Checkpoint 2

For each pair and each prompt, the judge scores both responses on 6 dimensions:
  instruction_following, correctness, clarity, completeness,
  structured_output_validity, hallucination_risk

Output schema (per assignment Section 9):
  {
    "prompt_id": str,
    "checkpoint_a": str,
    "checkpoint_b": str,
    "response_a_scores": {dimension: score, ...},
    "response_b_scores": {dimension: score, ...},
    "winner": "A" | "B" | "tie",
    "justification": str
  }

Saves results to:
  results/judge_ckpt0v1.jsonl
  results/judge_ckpt1v2.jsonl
  results/judge_ckpt0v2.jsonl

Implemented in Stage 5.
"""
# TODO: implement in Stage 5
