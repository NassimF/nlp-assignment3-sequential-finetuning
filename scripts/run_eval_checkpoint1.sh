#!/bin/bash
# Generate responses and compute metrics for Checkpoint 1 (after Stage 1 Alpaca).
set -e
python src/inference/generate_responses.py --config config.yaml --checkpoint 1
python src/evaluation/json_eval.py     --config config.yaml --checkpoint 1
python src/evaluation/metrics.py       --config config.yaml --checkpoint 1
echo "Checkpoint 1 evaluation complete."
