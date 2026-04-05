"""
Stage 4 — Stage 2 QLoRA fine-tuning on teacher-generated JSON data.

Loads Phi-3.5 Mini Instruct + Stage 1 LoRA adapter as the starting point,
then continues QLoRA fine-tuning on data/json_train.jsonl.

Saves the Stage 2 LoRA adapter to checkpoints/stage2/.
Logs per-step loss to logs/stage2_train.log and logs/stage2_loss.csv.

Usage (UTSA ARC):
    sbatch hpc/stage2_train.sbatch

Usage (direct):
    python src/training/stage2_train.py --config config.yaml
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
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

def load_json_dataset(path: str, tokenizer) -> Dataset:
    """Load JSONL and format each example with the model's chat template."""
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
    parser = argparse.ArgumentParser(description="Stage 2 QLoRA training on teacher-generated JSON data.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    s2 = cfg["stage2"]
    s1 = cfg["stage1"]
    logger = get_logger("stage2_train", s2["log_file"])

    logger.info("=" * 60)
    logger.info("Stage 2 — QLoRA fine-tuning on teacher-generated JSON data")
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
    logger.info(f"Loading dataset: {cfg['data']['json_train_path']}")
    train_dataset = load_json_dataset(cfg["data"]["json_train_path"], tokenizer)
    logger.info(f"Training examples: {len(train_dataset)}")

    # ── Quantization config ──────────────────────────────────────────────────
    q = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    # ── Model: load base + merge Stage 1 adapter ─────────────────────────────
    logger.info(f"Loading base model: {student_model}")
    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model.config.use_cache = False

    logger.info(f"Loading Stage 1 adapter from: {s1['checkpoint_dir']}")
    model = PeftModel.from_pretrained(model, s1["checkpoint_dir"], is_trainable=False)
    model = model.merge_and_unload()
    logger.info("Stage 1 adapter merged into base model.")

    model = prepare_model_for_kbit_training(model)

    # ── Fresh LoRA adapter on top of merged Stage 1 model ────────────────────
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
    Path(s2["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=s2["checkpoint_dir"],
        num_train_epochs=s2["epochs"],
        per_device_train_batch_size=s2["per_device_train_batch_size"],
        gradient_accumulation_steps=s2["gradient_accumulation_steps"],
        learning_rate=s2["learning_rate"],
        lr_scheduler_type=s2["lr_scheduler_type"],
        warmup_ratio=s2["warmup_ratio"],
        bf16=s2["bf16"],
        fp16=s2["fp16"],
        logging_steps=s2["logging_steps"],
        save_steps=s2["save_steps"],
        save_total_limit=2,
        max_length=s2["max_seq_length"],
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
        callbacks=[LossLoggerCallback(s2["loss_csv"], logger)],
    )

    logger.info("Starting training ...")
    logger.info(f"  Epochs:            {s2['epochs']}")
    logger.info(f"  Batch size:        {s2['per_device_train_batch_size']}")
    logger.info(f"  Grad accum steps:  {s2['gradient_accumulation_steps']}")
    logger.info(f"  Effective batch:   {s2['per_device_train_batch_size'] * s2['gradient_accumulation_steps']}")
    logger.info(f"  Learning rate:     {s2['learning_rate']}")
    logger.info(f"  Max seq length:    {s2['max_seq_length']}")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    logger.info(f"Saving adapter to {s2['checkpoint_dir']} ...")
    trainer.save_model(s2["checkpoint_dir"])
    tokenizer.save_pretrained(s2["checkpoint_dir"])
    logger.info("Stage 2 training complete.")


if __name__ == "__main__":
    main()
