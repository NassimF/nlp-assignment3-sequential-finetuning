#!/bin/bash
# Generate responses and compute metrics for Checkpoint 2 (after Stage 2 JSON).
set -e
python src/inference/generate_responses.py --config config.yaml --checkpoint 2
python src/evaluation/json_eval.py     --config config.yaml --checkpoint 2
python src/evaluation/metrics.py       --config config.yaml --checkpoint 2
python src/evaluation/judge_eval.py    --config config.yaml
python src/evaluation/aggregate_results.py --config config.yaml
echo "Checkpoint 2 evaluation and final aggregation complete."
