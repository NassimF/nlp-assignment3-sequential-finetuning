#!/bin/bash
# Generate responses and compute metrics for Checkpoint 0 (untuned base model).
set -e
python src/inference/generate_responses.py --config config.yaml --checkpoint 0
python src/evaluation/json_eval.py     --config config.yaml --checkpoint 0
python src/evaluation/metrics.py       --config config.yaml --checkpoint 0
echo "Checkpoint 0 evaluation complete."
