# Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation

**Assignment 3 — LLM & Agentic Systems (Graduate)**  
Two-stage QLoRA fine-tuning pipeline studying catastrophic forgetting in sequential instruction tuning.

**Research question:** If a small LLM is first fine-tuned on Alpaca-style instruction data and then
continued on teacher-generated JSON structured-output data, does it gain JSON reliability while
maintaining general instruction-following ability, or does Stage 2 cause catastrophic forgetting?

**Result:** No catastrophic forgetting detected. Stage 2 marginally improved general instruction-following (judge win rate +7.9pp ckpt1→ckpt2) while providing mixed JSON specialization gains. See [docs/blog_post.md](docs/blog_post.md) for the full report.

---

## Repository Structure

```
├── config.yaml              # All hyperparameters — edit here, not in source
├── environment.yml          # Conda environment spec
├── requirements.txt         # pip dependencies
├── .env.example             # API credential template
├── prompts/
│   ├── teacher_generation.yaml   # 5 task-type prompts for imitation learning
│   └── judge_evaluation.yaml     # Pairwise judge prompt
├── src/
│   ├── utils.py                  # Config, logging, API client, GPU info
│   ├── data/
│   │   ├── prepare_alpaca.py     # Alpaca train/eval splits
│   │   └── generate_json_dataset.py  # Teacher-generated JSON dataset
│   ├── training/
│   │   ├── stage1_train.py       # QLoRA on Alpaca
│   │   └── stage2_train.py       # QLoRA on teacher JSON (from Stage 1 ckpt)
│   ├── inference/
│   │   └── generate_responses.py # Multi-checkpoint inference
│   └── evaluation/
│       ├── judge_eval.py         # LLM-as-judge pairwise comparison
│       ├── json_eval.py          # JSON validity, schema, F1, error taxonomy
│       ├── metrics.py            # ROUGE, BERTScore, aggregation
│       └── aggregate_results.py  # Reproduce all tables and figures
├── hpc/
│   ├── stage1_train.sbatch       # SLURM script for UTSA ARC (Stage 1)
│   └── stage2_train.sbatch       # SLURM script for UTSA ARC (Stage 2)
├── scripts/
│   ├── run_data_prep.sh
│   ├── run_eval_checkpoint0.sh
│   ├── run_eval_checkpoint1.sh
│   └── run_eval_checkpoint2.sh
├── logs/                    # Training logs + response outputs (tracked)
├── results/                 # Tables, figures, judge JSONL (tracked)
└── docs/
    └── blog_post.md         # 5-page report + appendix
```

---

## Setup

### 1. Clone the repo

```bash
git clone <repo_url>
cd nlp-assignment3-sequential-finetuning
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate nlp-assignment3
```

**Key dependencies:** Python 3.11, PyTorch 2.9.1, transformers 4.46+, peft 0.13+, trl 0.12+, bitsandbytes 0.44+, sentence-transformers, rouge-score, bert-score. See `requirements.txt` for pinned versions.

### 3. Configure API credentials

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY=...
#   UTSA_BASE_URL=...
```

### 4. (Optional) Update model names in config.yaml

If the UTSA API model name strings differ from the defaults, update
`model.teacher` and `model.judge` in `config.yaml`.

---

## Data Preparation

```bash
conda activate nlp-assignment3
bash scripts/run_data_prep.sh
```

Produces:
- `data/alpaca_train.jsonl` (~51k examples)
- `data/alpaca_eval.jsonl` (150 held-out prompts)
- `data/json_train.jsonl` (~1000 teacher-generated examples, 5 task types)
- `data/json_eval.jsonl` (100 held-out JSON prompts)

---

## Training

Training is run on **UTSA ARC** using 80GB NVIDIA A100 GPUs via SLURM.

### Stage 1 — Alpaca QLoRA

```bash
sbatch hpc/stage1_train.sbatch
```

### Stage 2 — Teacher JSON QLoRA

```bash
sbatch hpc/stage2_train.sbatch
```

To run directly (e.g., for debugging):

```bash
conda activate nlp-assignment3
python src/training/stage1_train.py --config config.yaml
python src/training/stage2_train.py --config config.yaml
```

Checkpoints are saved to `checkpoints/stage1/` and `checkpoints/stage2/` (gitignored).
Training logs and loss CSVs are saved to `logs/` (tracked).

---

## Evaluation

```bash
bash scripts/run_eval_checkpoint0.sh   # baseline (untuned)
bash scripts/run_eval_checkpoint1.sh   # after Stage 1
bash scripts/run_eval_checkpoint2.sh   # after Stage 2 + judge + aggregation
```

Final results appear in `results/checkpoint_comparison_table.csv`.

---

## Configuration

All tunable parameters live in `config.yaml`:

| Key | Description |
|-----|-------------|
| `model.student` | HuggingFace model ID for the student |
| `model.teacher` / `model.judge` | Model name on UTSA API |
| `lora.r`, `lora.lora_alpha` | LoRA rank and scaling |
| `stage1.epochs`, `stage2.epochs` | Training epochs per stage |
| `stage1.learning_rate` / `stage2.learning_rate` | Per-stage LR |
| `data.alpaca_eval_size` | Held-out Alpaca eval set size |
| `data.json_examples_per_type` | Teacher-generated examples per task type |
| `ablation.stage2_lr_sweep` | LR values for ablation study |

---

## Reproducing Tables and Figures

```bash
conda activate nlp-assignment3
python src/evaluation/aggregate_results.py --config config.yaml
```

Outputs: `results/checkpoint_comparison_table.csv`, `results/json_metrics_table.csv`, `results/forgetting_curve.png`, `results/loss_curves.png`.

---

## Results Summary

| Metric | Ckpt0 (Base) | Ckpt1 (Stage 1) | Ckpt2 (Stage 2) |
|--------|-------------|-----------------|-----------------|
| ROUGE-L (Alpaca) | 0.279 | 0.280 | 0.283 |
| BERTScore F1 | 0.823 | 0.823 | 0.823 |
| JSON Validity | 48% | 58% | 56% |
| Alpaca Judge Win Rate vs ckpt0 | — | 16.2% | 24.1% |

See [docs/blog_post.md](docs/blog_post.md) for the full report and analysis.

---

## Hardware

Trained on **UTSA ARC** using 80GB NVIDIA A100 GPUs.
SLURM batch scripts are provided in `hpc/` for both training stages.
