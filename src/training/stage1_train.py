"""
Stage 3 — Stage 1 QLoRA fine-tuning on Alpaca data.

Loads Phi-3.5 Mini Instruct in 4-bit (bitsandbytes NF4), applies a LoRA
adapter via PEFT, and trains with TRL SFTTrainer on data/alpaca_train.jsonl.

Saves the LoRA adapter to checkpoints/stage1/.
Logs per-step loss to logs/stage1_train.log and logs/stage1_loss.csv.

Usage (DGX):
    python src/training/stage1_train.py --config config.yaml

Usage (ARC):
    sbatch hpc/stage1_train.sbatch

Implemented in Stage 3.
"""
# TODO: implement in Stage 3
