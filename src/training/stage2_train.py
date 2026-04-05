"""
Stage 4 — Stage 2 QLoRA fine-tuning on teacher-generated JSON data.

Loads Phi-3.5 Mini Instruct + Stage 1 LoRA adapter as the starting point,
then continues QLoRA fine-tuning on data/json_train.jsonl.

Saves the Stage 2 LoRA adapter to checkpoints/stage2/.
Logs per-step loss to logs/stage2_train.log and logs/stage2_loss.csv.

Usage (DGX):
    python src/training/stage2_train.py --config config.yaml

Usage (ARC):
    sbatch hpc/stage2_train.sbatch

Implemented in Stage 4.
"""
# TODO: implement in Stage 4
