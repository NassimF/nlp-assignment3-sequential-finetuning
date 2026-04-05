"""
Stage 3 — Stage 1 QLoRA fine-tuning on Alpaca data.

Loads Phi-3.5 Mini Instruct in 4-bit (NF4), applies a LoRA adapter via PEFT,
and trains with TRL SFTTrainer on data/alpaca_train.jsonl.

Saves the LoRA adapter to checkpoints/stage1/.
Logs per-step loss to logs/stage1_train.log and logs/stage1_loss.csv.

Usage (UTSA ARC):
    sbatch hpc/stage1_train.sbatch

Usage (direct):
    python src/training/stage1_train.py --config config.yaml
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from src.utils import LossCSVWriter, get_logger, load_config, log_gpu_info


# ---------------------------------------------------------------------------
# Loss logging callback
# ---------------------------------------------------------------------------

class LossLoggerCallback(TrainerCallback):
    """Writes (step, loss) rows to a CSV for loss-curve figures."""

    def __init__(self, csv_path: str, logger):
        self._writer = LossCSVWriter(csv_path)
        self._logger = logger

    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._writer.write(state.global_step, logs["loss"])
            self._logger.info(f"step={state.global_step}  loss={logs['loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        self._writer.close()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_alpaca_dataset(path: str, tokenizer) -> Dataset:
    """Load JSONL and format each example with the model's chat template."""
    import json
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 1 QLoRA training on Alpaca data.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    logger = get_logger("stage1_train", s1["log_file"])

    logger.info("=" * 60)
    logger.info("Stage 1 — QLoRA fine-tuning on Alpaca data")
    logger.info("=" * 60)
    log_gpu_info(logger)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    student_model = cfg["model"]["student"]
    logger.info(f"Loading tokenizer: {student_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        student_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ──────────────────────────────────────────────────────────────
    logger.info(f"Loading dataset: {cfg['data']['alpaca_train_path']}")
    train_dataset = load_alpaca_dataset(cfg["data"]["alpaca_train_path"], tokenizer)
    logger.info(f"Training examples: {len(train_dataset)}")

    # ── Quantization config ──────────────────────────────────────────────────
    q = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {student_model}")
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training arguments ───────────────────────────────────────────────────
    Path(s1["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=s1["checkpoint_dir"],
        num_train_epochs=s1["epochs"],
        per_device_train_batch_size=s1["per_device_train_batch_size"],
        gradient_accumulation_steps=s1["gradient_accumulation_steps"],
        learning_rate=s1["learning_rate"],
        lr_scheduler_type=s1["lr_scheduler_type"],
        warmup_ratio=s1["warmup_ratio"],
        bf16=s1["bf16"],
        fp16=s1["fp16"],
        logging_steps=s1["logging_steps"],
        save_steps=s1["save_steps"],
        save_total_limit=2,
        max_seq_length=s1["max_seq_length"],
        dataset_text_field="text",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=4,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[LossLoggerCallback(s1["loss_csv"], logger)],
    )

    logger.info("Starting training ...")
    logger.info(f"  Epochs:            {s1['epochs']}")
    logger.info(f"  Batch size:        {s1['per_device_train_batch_size']}")
    logger.info(f"  Grad accum steps:  {s1['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch:   {s1['per_device_train_batch_size'] * s1['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate:     {s1['learning_rate']}")
    logger.info(f"  Max seq length:    {s1['max_seq_length']}")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    logger.info(f"Saving adapter to {s1['checkpoint_dir']} ...")
    trainer.save_model(s1["checkpoint_dir"])
    tokenizer.save_pretrained(s1["checkpoint_dir"])
    logger.info("Stage 1 training complete.")


if __name__ == "__main__":
    main()
