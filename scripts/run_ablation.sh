#!/bin/bash
# Ablation study: Stage 2 learning rate sweep.
#
# Trains three Stage 2 variants from the same Stage 1 checkpoint:
#   - lr=2e-5 (baseline, full data) — already done as ckpt2
#   - lr=1e-5 (half LR)
#   - data_fraction=0.5 (half data, baseline LR)
#
# Each variant saves its adapter, generates responses, and runs eval.
# Results are collected in results/ablation_summary.csv by aggregate_results.py.
#
# Run from the project root with the nlp-assignment3 env active.
# Stage 1 training (checkpoints/stage1/) must be complete before running this.
set -e

echo "=== Ablation: Stage 2 LR=1e-5 ==="
python src/training/stage2_train.py \
    --config config.yaml \
    --lr 1e-5 \
    --output_suffix "_ablr1e5"

echo "=== Ablation: Stage 2 data_fraction=0.5 ==="
python src/training/stage2_train.py \
    --config config.yaml \
    --data_fraction 0.5 \
    --output_suffix "_abfrac50"

echo "=== Ablation inference: LR=1e-5 variant ==="
python src/inference/generate_responses.py \
    --config config.yaml \
    --checkpoint 2 \
    --adapter_override checkpoints/stage2_ablr1e5 \
    --output_suffix "_ablr1e5"

echo "=== Ablation inference: data_fraction=0.5 variant ==="
python src/inference/generate_responses.py \
    --config config.yaml \
    --checkpoint 2 \
    --adapter_override checkpoints/stage2_abfrac50 \
    --output_suffix "_abfrac50"

echo "=== Ablation complete. Run aggregate_results.py to summarize. ==="
