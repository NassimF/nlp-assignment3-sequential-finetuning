#!/bin/bash
# Prepare both datasets (Alpaca + teacher-generated JSON).
# Run from the project root with the nlp-assignment3 env active.
set -e

echo "=== Stage 2a: Alpaca data preparation ==="
python src/data/prepare_alpaca.py --config config.yaml

echo "=== Stage 2b: Teacher-generated JSON dataset ==="
python src/data/generate_json_dataset.py --config config.yaml

echo "=== Data preparation complete ==="
echo "Alpaca train: data/alpaca_train.jsonl"
echo "Alpaca eval:  data/alpaca_eval.jsonl"
echo "JSON train:   data/json_train.jsonl"
echo "JSON eval:    data/json_eval.jsonl"
