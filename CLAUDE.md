# Assignment 3 — Claude Code Context

This file gives any new Claude Code session full context about this project.
Read this before doing anything else.

## Project

**Assignment 3: Sequential Instruction Tuning of a Small LLM**
Course: LLM & Agentic Systems (Graduate) — Dr. Peyman Najafirad
Due: April 6, 2026

**Research question:** Does Stage 2 fine-tuning on teacher-generated JSON data
cause catastrophic forgetting of general instruction-following ability gained in Stage 1?

## Hardware

- **DGX server**: 2× NVIDIA A100-SXM4-80GB, driver 535.230.02
- Training runs directly with `python` in a `tmux` session (no SLURM on DGX)
- SLURM `.sbatch` scripts in `hpc/` are for UTSA ARC submission / grading evidence

## Environment

```bash
conda activate nlp-assignment3   # cloned from alpaca-lora, has torch 2.9.1
```

## Models

- **Student**: `microsoft/Phi-3.5-mini-instruct` (fine-tuned locally via QLoRA)
- **Teacher**: `llama-3.3-70b-instruct-awq` — used for data generation (fast, reliable JSON output)
- **Judge**: `qwen3-235b-a22b-thinking-2507-fp8` — used for evaluation scoring (235B reasoning model)

## API Credentials

Stored in `.env` (gitignored). See `.env.example` for template.
- `OPENAI_API_KEY` — API key
- `UTSA_BASE_URL` — UTSA OpenAI-compatible endpoint URL

The model name for teacher/judge in the UTSA API may differ from the HF name.
Check/update `model.teacher` and `model.judge` in `config.yaml` if needed.

## Pipeline Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Scaffold, env, config | ✅ Complete |
| 2 | Data prep (Alpaca + teacher JSON) | 🔄 In progress (Alpaca done, JSON generating) |
| 3 | Stage 1 QLoRA training (Alpaca) | ⬜ Todo |
| 4 | Stage 2 QLoRA training (JSON) | ⬜ Todo |
| 5 | Judge eval + forgetting analysis | ⬜ Todo |
| 6 | Blog post report | ⬜ Todo |

Update the Status column as stages complete.

## Key Files

```
config.yaml                        ← all hyperparameters (edit here, not in code)
.env                               ← API keys (gitignored, create from .env.example)
src/utils.py                       ← load_config(), get_logger(), load_api_client(), log_gpu_info()
prompts/teacher_generation.yaml    ← 5 task-type prompts for imitation learning
prompts/judge_evaluation.yaml      ← pairwise judge prompt
hpc/stage1_train.sbatch            ← SLURM script (ARC)
hpc/stage2_train.sbatch            ← SLURM script (ARC)
logs/                              ← training logs + response outputs (tracked in git)
results/                           ← tables, figures, judge JSONL (tracked in git)
checkpoints/                       ← LoRA adapters (gitignored, large)
data/                              ← datasets (gitignored, regenerated from scripts)
```

## Running Training (DGX)

```bash
# Stage 1
python src/training/stage1_train.py --config config.yaml

# Stage 2
python src/training/stage2_train.py --config config.yaml
```

## Running Evaluation

```bash
bash scripts/run_eval_checkpoint0.sh
bash scripts/run_eval_checkpoint1.sh
bash scripts/run_eval_checkpoint2.sh   # also runs judge + aggregation
```

## Assignment Requirements Checklist

- [ ] 3-checkpoint comparison table (ckpt0, ckpt1, ckpt2)
- [ ] Alpaca eval: 100+ prompts, ROUGE-1/2/L, BERTScore, length, completion rate, judge win rate
- [ ] JSON eval: validity, schema compliance, exact match, field F1, error taxonomy
- [ ] Judge: all 3 pairs (0v1, 1v2, 0v2), 6 dimensions, Section 9 output schema
- [ ] Forgetting analysis: absolute Δ win rate, Δ ROUGE-L, Δ BERTScore, per-category
- [ ] At least 1 ablation (LR sweep or data fraction)
- [ ] SLURM scripts in hpc/ (both stages)
- [ ] Training logs + loss curves in logs/
- [ ] Blog post in docs/blog_post.md (5 pages + appendix)
- [ ] Prompt engineering section with iteration evidence (15% of grade)
