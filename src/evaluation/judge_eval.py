"""
LLM-as-judge pairwise evaluation.

Runs all three pairwise comparisons using Qwen3-235B as judge:
  - Checkpoint 0 vs Checkpoint 1  →  results/judge_ckpt0v1.jsonl
  - Checkpoint 1 vs Checkpoint 2  →  results/judge_ckpt1v2.jsonl
  - Checkpoint 0 vs Checkpoint 2  →  results/judge_ckpt0v2.jsonl

For each prompt the judge scores both responses on 6 dimensions and declares a winner.
Response order is randomized to reduce position bias.

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

Usage:
    python src/evaluation/judge_eval.py --config config.yaml
    python src/evaluation/judge_eval.py --config config.yaml --pairs 0v1
    python src/evaluation/judge_eval.py --config config.yaml --pairs 0v1,1v2,0v2
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from src.utils import get_logger, get_model_name, load_api_client, load_config


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_judge_prompts(prompts_path: str = "prompts/judge_evaluation.yaml") -> dict:
    with open(prompts_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def call_judge(client, judge_model: str, system_prompt: str,
               user_prompt: str, cfg: dict) -> str:
    """Call the judge model and return the raw text response."""
    ev = cfg["evaluation"]
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=ev["judge_max_new_tokens"],
        temperature=ev["judge_temperature"],
    )
    return response.choices[0].message.content or ""


def parse_judge_response(text: str) -> dict | None:
    """Extract and parse the JSON object from the judge's response."""
    text = text.strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Find JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Single pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(ckpt_a: int, ckpt_b: int, cfg: dict, logger,
                  client, judge_model: str, prompts: dict,
                  eval_type: str = "alpaca") -> list[dict]:
    """
    Run pairwise judge eval on all prompts for a given checkpoint pair.
    eval_type: "alpaca" evaluates general instruction-following ability.
    """
    logs_dir = cfg["evaluation"]["logs_dir"]
    file_a = f"{logs_dir}/responses_ckpt{ckpt_a}_{eval_type}.jsonl"
    file_b = f"{logs_dir}/responses_ckpt{ckpt_b}_{eval_type}.jsonl"

    if not Path(file_a).exists() or not Path(file_b).exists():
        logger.warning(f"Response files not found for ckpt{ckpt_a} vs ckpt{ckpt_b}")
        return []

    with open(file_a) as f:
        records_a = {r["prompt_id"]: r for r in (json.loads(l) for l in f)}
    with open(file_b) as f:
        records_b = {r["prompt_id"]: r for r in (json.loads(l) for l in f)}

    common_ids = sorted(set(records_a) & set(records_b))
    logger.info(f"ckpt{ckpt_a} vs ckpt{ckpt_b}: {len(common_ids)} prompts")

    system_prompt = prompts["system"]
    prompt_template = prompts["pairwise_judge"]

    results = []
    failed = 0

    for pid in tqdm(common_ids, desc=f"  judge ckpt{ckpt_a}v{ckpt_b}"):
        ra = records_a[pid]
        rb = records_b[pid]

        # Randomize A/B assignment to reduce position bias
        if random.random() < 0.5:
            actual_a, actual_b = ra, rb
            label_a, label_b = f"ckpt{ckpt_a}", f"ckpt{ckpt_b}"
            swapped = False
        else:
            actual_a, actual_b = rb, ra
            label_a, label_b = f"ckpt{ckpt_b}", f"ckpt{ckpt_a}"
            swapped = True

        user_prompt = prompt_template.format(
            instruction=actual_a.get("instruction", ""),
            input=actual_a.get("input", ""),
            prompt_id=pid,
            checkpoint_a=label_a,
            checkpoint_b=label_b,
            response_a=actual_a["response"],
            response_b=actual_b["response"],
        )

        # Call judge with retry
        parsed = None
        for attempt in range(3):
            try:
                raw = call_judge(client, judge_model, system_prompt, user_prompt, cfg)
                parsed = parse_judge_response(raw)
                if parsed:
                    break
            except Exception as e:
                logger.warning(f"  Judge call failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)

        if parsed is None:
            logger.warning(f"  Failed to parse judge response for {pid}")
            failed += 1
            continue

        # If we swapped A/B, un-swap the result so it always reflects ckpt_a vs ckpt_b
        if swapped:
            parsed["checkpoint_a"], parsed["checkpoint_b"] = \
                f"ckpt{ckpt_a}", f"ckpt{ckpt_b}"
            parsed["response_a_scores"], parsed["response_b_scores"] = \
                parsed.get("response_b_scores", {}), parsed.get("response_a_scores", {})
            winner = parsed.get("winner", "tie")
            if winner == "A":
                parsed["winner"] = "B"
            elif winner == "B":
                parsed["winner"] = "A"
        else:
            parsed["checkpoint_a"] = f"ckpt{ckpt_a}"
            parsed["checkpoint_b"] = f"ckpt{ckpt_b}"

        parsed["prompt_id"] = pid
        results.append(parsed)

    logger.info(f"  ckpt{ckpt_a}v{ckpt_b}: {len(results)} evaluated, {failed} failed")
    return results


# ---------------------------------------------------------------------------
# Win-rate summary
# ---------------------------------------------------------------------------

def compute_win_rates(results: list[dict]) -> dict:
    """Compute win/tie/loss counts and win rate for checkpoint A."""
    wins_a = sum(1 for r in results if r.get("winner") == "A")
    wins_b = sum(1 for r in results if r.get("winner") == "B")
    ties   = sum(1 for r in results if r.get("winner") == "tie")
    n = len(results)
    return {
        "n":          n,
        "wins_a":     wins_a,
        "wins_b":     wins_b,
        "ties":       ties,
        "win_rate_a": wins_a / n if n else 0.0,
        "win_rate_b": wins_b / n if n else 0.0,
        "tie_rate":   ties   / n if n else 0.0,
    }


def compute_avg_scores(results: list[dict]) -> dict:
    """Compute mean scores per dimension for both checkpoints."""
    dims = ["instruction_following", "correctness", "clarity",
            "completeness", "structured_output_validity", "hallucination_risk"]
    avgs_a = {}
    avgs_b = {}
    for dim in dims:
        scores_a = [r["response_a_scores"][dim] for r in results
                    if "response_a_scores" in r and dim in r["response_a_scores"]]
        scores_b = [r["response_b_scores"][dim] for r in results
                    if "response_b_scores" in r and dim in r["response_b_scores"]]
        avgs_a[dim] = sum(scores_a) / len(scores_a) if scores_a else 0.0
        avgs_b[dim] = sum(scores_b) / len(scores_b) if scores_b else 0.0
    return {"checkpoint_a_avg": avgs_a, "checkpoint_b_avg": avgs_b}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge pairwise evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pairs", default="0v1,1v2,0v2",
                        help="Comma-separated pairs, e.g. '0v1,1v2,0v2'")
    parser.add_argument("--eval_type", default="alpaca",
                        help="Response file suffix: 'alpaca' or 'json'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)
    logger = get_logger("judge_eval", "logs/judge_eval.log")

    client = load_api_client()
    judge_model = get_model_name(cfg, "judge")
    logger.info(f"Judge model: {judge_model}")

    prompts = load_judge_prompts()
    out_dir = Path(cfg["evaluation"]["results_dir"])
    out_dir.mkdir(exist_ok=True)

    pairs = []
    for p in args.pairs.split(","):
        p = p.strip()
        a, b = int(p[0]), int(p[2])
        pairs.append((a, b))

    all_summaries = {}
    for ckpt_a, ckpt_b in pairs:
        label = f"ckpt{ckpt_a}v{ckpt_b}"
        logger.info(f"Running judge eval: {label}")
        results = evaluate_pair(ckpt_a, ckpt_b, cfg, logger,
                                client, judge_model, prompts, args.eval_type)

        if not results:
            continue

        # Save JSONL
        out_path = out_dir / f"judge_{label}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"  Saved → {out_path}")

        # Summary
        win_rates = compute_win_rates(results)
        avg_scores = compute_avg_scores(results)
        summary = {**win_rates, **avg_scores}
        all_summaries[label] = summary

        logger.info(f"  Win rate A (ckpt{ckpt_a}): {win_rates['win_rate_a']:.3f}")
        logger.info(f"  Win rate B (ckpt{ckpt_b}): {win_rates['win_rate_b']:.3f}")
        logger.info(f"  Tie rate:                  {win_rates['tie_rate']:.3f}")

    # Save summary JSON
    if all_summaries:
        summary_path = out_dir / "judge_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"Saved judge summary → {summary_path}")


if __name__ == "__main__":
    main()
