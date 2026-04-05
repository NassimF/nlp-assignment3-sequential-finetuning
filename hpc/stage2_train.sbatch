#!/bin/bash
# =============================================================================
# UTSA ARC HPC — Stage 2 training job (Teacher-generated JSON QLoRA fine-tuning)
# On DGX server run directly: python src/training/stage2_train.py --config config.yaml
# =============================================================================
#SBATCH --job-name=nlp-a3-stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm_stage2_%j.out
#SBATCH --error=logs/slurm_stage2_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"

module load miniconda3
conda activate nlp-assignment3

cd $SLURM_SUBMIT_DIR

python src/training/stage2_train.py --config config.yaml

echo "End: $(date)"
