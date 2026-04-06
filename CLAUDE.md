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
- **Teacher**: `llama-3.3-70b-instruct-awq` — data generation. Qwen3-235B was tested first but
  rejected: it is a thinking model that exhausts its token budget on internal reasoning before
  producing visible output (content=None even at max_tokens=4096), making batch generation
  impractical. Llama 3.3 70B produces reliable, well-structured JSON output.
- **Judge**: `qwen3-235b-a22b-thinking-2507-fp8` — evaluation scoring. Thinking model is fine here
  since we make far fewer judge calls and higher reasoning quality produces better-calibrated
  pairwise scores across 6 dimensions.

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
| 2 | Data prep (Alpaca + teacher JSON) | ✅ Complete (981 train + 100 eval JSON examples) |
| 3 | Stage 1 QLoRA training (Alpaca) | ✅ Complete (final loss 0.910, step 9552) |
| 4 | Stage 2 QLoRA training (JSON) | ✅ Complete (final loss 0.751, step 124) |
| 5a | Inference + auto metrics (all 3 ckpts) | ✅ Complete |
| 5b | Judge eval Alpaca (3 pairs) | ✅ Complete |
| 5c | Judge eval JSON (3 pairs) | 🔄 Running (PID 357638, ~5h) |
| 6 | Blog post report | 🔄 ~95% complete (pending JSON judge results) |

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

- [x] 3-checkpoint comparison table (ckpt0, ckpt1, ckpt2)
- [x] Alpaca eval: 100+ prompts, ROUGE-1/2/L, BERTScore, length, completion rate, judge win rate
- [x] JSON eval: validity, schema compliance, exact match, field F1, error taxonomy
- [x] Judge: all 3 pairs (0v1, 1v2, 0v2), 6 dimensions, Section 9 output schema (Alpaca done; JSON running)
- [x] Forgetting analysis: absolute Δ win rate, Δ ROUGE-L, Δ BERTScore, per-category
- [ ] At least 1 ablation (LR sweep or data fraction) — scripts ready, not yet run
- [x] SLURM scripts in hpc/ (both stages)
- [x] Training logs + loss curves in logs/
- [x] Blog post in docs/blog_post.md (5 pages + appendix)
- [x] Prompt engineering section with iteration evidence (15% of grade)
