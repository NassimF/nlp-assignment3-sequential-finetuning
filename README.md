# Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation

**Assignment 3 — LLM & Agentic Systems (Graduate)**  
Two-stage QLoRA fine-tuning pipeline studying catastrophic forgetting in sequential instruction tuning.

**Research question:** If a small LLM is first fine-tuned on Alpaca-style instruction data and then
continued on teacher-generated JSON structured-output data, does it gain JSON reliability while
maintaining general instruction-following ability, or does Stage 2 cause catastrophic forgetting?

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

### 3. Configure API credentials

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY=...
#   UTSA_BASE_URL=...
```

### 4. (Optional) Update model names in config.yaml

If the UTSA API uses different model name strings for Llama 3.1 70B Instruct,
update `model.teacher` and `model.judge` in `config.yaml`.

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

### Stage 1 — Alpaca QLoRA (DGX)

```bash
conda activate nlp-assignment3
python src/training/stage1_train.py --config config.yaml
# Or in tmux: tmux new -s stage1; python src/training/stage1_train.py --config config.yaml
```

### Stage 1 — Alpaca QLoRA (UTSA ARC)

```bash
sbatch hpc/stage1_train.sbatch
```

### Stage 2 — Teacher JSON QLoRA (DGX)

```bash
python src/training/stage2_train.py --config config.yaml
```

### Stage 2 — Teacher JSON QLoRA (UTSA ARC)

```bash
sbatch hpc/stage2_train.sbatch
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

## Results

See [docs/blog_post.md](docs/blog_post.md) for the full report.

---

## Hardware

Trained on NVIDIA A100-SXM4-80GB (DGX server).
Equivalent SLURM scripts for UTSA ARC provided in `hpc/`.
