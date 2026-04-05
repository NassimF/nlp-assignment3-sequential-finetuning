"""
JSON structured output evaluation.

For each checkpoint's JSON responses, computes:
  - JSON validity rate       (json.loads() succeeds)
  - Schema compliance rate   (required keys present, basic type check)
  - Exact-match accuracy     (output == reference after normalization)
  - Field-level F1           (precision/recall per field for extraction tasks)
  - Error taxonomy           (categorized failure counts)

Saves per-checkpoint results to results/json_eval_ckpt{N}.json
and combined summary to results/json_eval_summary.csv.

Usage:
    python src/evaluation/json_eval.py --config config.yaml --checkpoint 0
    python src/evaluation/json_eval.py --config config.yaml --checkpoint 0,1,2
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.utils import get_logger, load_config


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def try_parse_json(text: str):
    """Try to parse text as JSON. Returns (parsed_obj, error_type) or (None, error_type)."""
    import re
    if not isinstance(text, str):
        return None, "not_string"
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    # Try to find a JSON object or array
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        msg = str(e).lower()
        if "expecting" in msg and "property name" in msg:
            return None, "invalid_key_format"
        elif "trailing" in msg or "extra" in msg:
            return None, "trailing_content"
        elif "unterminated" in msg or "expecting" in msg:
            return None, "truncated_or_malformed"
        elif "value" in msg:
            return None, "invalid_value"
        else:
            return None, "other_json_error"


def check_schema_compliance(parsed: dict, reference: dict) -> bool:
    """Check that all top-level keys from reference are present in parsed output."""
    if not isinstance(parsed, dict) or not isinstance(reference, dict):
        return False
    return all(k in parsed for k in reference.keys())


def exact_match(parsed, reference) -> bool:
    """Normalized exact match — compares JSON-serialized canonical forms."""
    try:
        parsed_str = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        ref_str = json.dumps(reference, sort_keys=True, ensure_ascii=False)
        return parsed_str == ref_str
    except Exception:
        return False


def field_level_f1(parsed: dict, reference: dict) -> dict:
    """
    Compute field-level precision, recall, and F1 for extraction tasks.
    Only applies when both are flat dicts with string/number values.
    """
    if not isinstance(parsed, dict) or not isinstance(reference, dict):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_keys = set(reference.keys())
    pred_keys = set(parsed.keys())

    # True positives: key present in both AND value matches (string comparison)
    tp = sum(
        1 for k in ref_keys & pred_keys
        if str(parsed[k]).strip().lower() == str(reference[k]).strip().lower()
    )
    precision = tp / len(pred_keys) if pred_keys else 0.0
    recall = tp / len(ref_keys) if ref_keys else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def evaluate_example(response: str, reference_str: str, task_type: str) -> dict:
    """Evaluate a single response against its reference."""
    parsed, error_type = try_parse_json(response)
    ref_parsed, _ = try_parse_json(reference_str)

    valid = parsed is not None
    schema_ok = False
    exact = False
    f1_scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    error = error_type or "none"

    if valid and ref_parsed is not None:
        schema_ok = check_schema_compliance(parsed, ref_parsed)
        exact = exact_match(parsed, ref_parsed)
        if task_type == "json_extraction" and isinstance(parsed, dict):
            f1_scores = field_level_f1(parsed, ref_parsed)

    return {
        "valid": valid,
        "schema_compliant": schema_ok,
        "exact_match": exact,
        "field_precision": f1_scores["precision"],
        "field_recall": f1_scores["recall"],
        "field_f1": f1_scores["f1"],
        "error_type": error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_checkpoint(checkpoint: int, cfg: dict, logger) -> dict:
    response_file = f"{cfg['evaluation']['logs_dir']}/responses_ckpt{checkpoint}_json.jsonl"
    if not Path(response_file).exists():
        logger.warning(f"Response file not found: {response_file}")
        return {}

    with open(response_file) as f:
        responses = [json.loads(line) for line in f]

    results = []
    error_counts = defaultdict(int)
    per_task = defaultdict(lambda: defaultdict(list))

    for r in responses:
        task_type = r.get("task_type", "unknown")
        eval_result = evaluate_example(r["response"], r["reference"], task_type)
        eval_result.update({
            "prompt_id": r["prompt_id"],
            "task_type": task_type,
            "checkpoint": checkpoint,
        })
        results.append(eval_result)
        error_counts[eval_result["error_type"]] += 1
        for metric in ("valid", "schema_compliant", "exact_match", "field_f1"):
            per_task[task_type][metric].append(float(eval_result[metric]))

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0

    # Overall metrics
    summary = {
        "checkpoint": checkpoint,
        "n_total": len(results),
        "json_validity_rate": mean([r["valid"] for r in results]),
        "schema_compliance_rate": mean([r["schema_compliant"] for r in results]),
        "exact_match_rate": mean([r["exact_match"] for r in results]),
        "field_f1_mean": mean([r["field_f1"] for r in results]),
        "error_taxonomy": dict(error_counts),
        "per_task": {
            task: {
                "n": len(vals["valid"]),
                "json_validity_rate": mean(vals["valid"]),
                "schema_compliance_rate": mean(vals["schema_compliant"]),
                "exact_match_rate": mean(vals["exact_match"]),
                "field_f1_mean": mean(vals["field_f1"]),
            }
            for task, vals in per_task.items()
        },
    }

    # Save per-checkpoint detail
    out_dir = Path(cfg["evaluation"]["results_dir"])
    out_dir.mkdir(exist_ok=True)
    detail_path = out_dir / f"json_eval_ckpt{checkpoint}.json"
    with open(detail_path, "w") as f:
        json.dump({"summary": summary, "examples": results}, f, indent=2)

    logger.info(f"Checkpoint {checkpoint} JSON eval:")
    logger.info(f"  Validity:          {summary['json_validity_rate']:.3f}")
    logger.info(f"  Schema compliance: {summary['schema_compliance_rate']:.3f}")
    logger.info(f"  Exact match:       {summary['exact_match_rate']:.3f}")
    logger.info(f"  Field F1 (mean):   {summary['field_f1_mean']:.3f}")
    logger.info(f"  Error taxonomy:    {dict(error_counts)}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="JSON structured output evaluation.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default="0",
                        help="Comma-separated checkpoint numbers")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("json_eval", "logs/json_eval.log")
    checkpoints = [int(c.strip()) for c in args.checkpoint.split(",")]

    all_summaries = []
    for ckpt in checkpoints:
        summary = evaluate_checkpoint(ckpt, cfg, logger)
        if summary:
            all_summaries.append(summary)

    # Save combined CSV
    if all_summaries:
        csv_path = Path(cfg["evaluation"]["results_dir"]) / "json_eval_summary.csv"
        fields = ["checkpoint", "n_total", "json_validity_rate",
                  "schema_compliance_rate", "exact_match_rate", "field_f1_mean"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in all_summaries:
                writer.writerow({k: s[k] for k in fields})
        logger.info(f"Saved combined JSON eval summary → {csv_path}")


if __name__ == "__main__":
    main()
