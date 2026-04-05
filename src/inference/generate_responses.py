"""
Multi-checkpoint inference — generates model responses for all eval prompts.

Runs Phi-3.5 Mini at each requested checkpoint:
  0  — untuned base model
  1  — Stage 1 adapter (checkpoints/stage1/)
  2  — Stage 2 adapter (checkpoints/stage2/)

For each checkpoint, generates responses on both eval sets:
  data/alpaca_eval.jsonl  →  logs/responses_ckpt{N}_alpaca.jsonl
  data/json_eval.jsonl    →  logs/responses_ckpt{N}_json.jsonl

Usage:
    python src/inference/generate_responses.py --config config.yaml --checkpoint 0
    python src/inference/generate_responses.py --config config.yaml --checkpoint 1
    python src/inference/generate_responses.py --config config.yaml --checkpoint 0,1,2
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

from src.utils import get_logger, load_config, log_gpu_info


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: dict, checkpoint: int):
    """
    Load the student model at a given checkpoint.
      0 → base model only
      1 → base model + stage1 LoRA adapter
      2 → base model + stage2 LoRA adapter
    """
    student_model = cfg["model"]["student"]
    q = cfg["quantization"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        student_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )

    adapter_dirs = {
        1: cfg["stage1"]["checkpoint_dir"],
        2: cfg["stage2"]["checkpoint_dir"],
    }
    if checkpoint in adapter_dirs:
        adapter_dir = adapter_dirs[checkpoint]
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()  # merge for faster inference

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Response generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, example: dict, cfg: dict) -> str:
    """Generate a single response for an eval example."""
    inf = cfg["inference"]
    messages = example["messages"][:1]  # user turn only

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inf["max_new_tokens"],
            temperature=inf["temperature"],
            do_sample=inf["do_sample"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_inference(model, tokenizer, eval_path: str, output_path: str,
                  checkpoint: int, split_name: str, cfg: dict, logger):
    """Generate responses for all examples in an eval JSONL and save results."""
    with open(eval_path) as f:
        examples = [json.loads(line) for line in f]

    results = []
    for example in tqdm(examples, desc=f"  ckpt{checkpoint} {split_name}"):
        response = generate_response(model, tokenizer, example, cfg)
        results.append({
            "prompt_id": example.get("prompt_id", f"{split_name}_{len(results):04d}"),
            "checkpoint": checkpoint,
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "reference": example.get("output", ""),
            "response": response,
            "task_type": example.get("task_type", "general"),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"  Saved {len(results)} responses → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate responses at one or more checkpoints.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint", default="0",
        help="Comma-separated checkpoint numbers to evaluate, e.g. '0' or '0,1,2'"
    )
    parser.add_argument("--adapter_override", default=None,
                        help="Override adapter directory (for ablation variants)")
    parser.add_argument("--output_suffix", default="",
                        help="Suffix appended to output filenames (for ablation variants)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = get_logger("generate_responses", "logs/inference.log")
    log_gpu_info(logger)

    checkpoints = [int(c.strip()) for c in args.checkpoint.split(",")]
    eval_cfg = cfg["evaluation"]

    alpaca_eval = cfg["data"]["alpaca_eval_path"]
    json_eval   = cfg["data"]["json_eval_path"]

    for ckpt in checkpoints:
        logger.info(f"Loading model at checkpoint {ckpt} ...")
        # Support adapter_override for ablation runs
        if args.adapter_override and ckpt == 2:
            cfg_copy = {**cfg}
            cfg_copy["stage2"] = {**cfg["stage2"], "checkpoint_dir": args.adapter_override}
            model, tokenizer = load_model_and_tokenizer(cfg_copy, ckpt)
        else:
            model, tokenizer = load_model_and_tokenizer(cfg, ckpt)

        sfx = args.output_suffix
        alpaca_out = f"{eval_cfg['logs_dir']}/responses_ckpt{ckpt}_alpaca{sfx}.jsonl"
        json_out   = f"{eval_cfg['logs_dir']}/responses_ckpt{ckpt}_json{sfx}.jsonl"

        logger.info(f"Checkpoint {ckpt} — Alpaca eval")
        run_inference(model, tokenizer, alpaca_eval, alpaca_out, ckpt, "alpaca", cfg, logger)

        logger.info(f"Checkpoint {ckpt} — JSON eval")
        run_inference(model, tokenizer, json_eval, json_out, ckpt, "json", cfg, logger)

        # Free memory before loading next checkpoint
        del model
        torch.cuda.empty_cache()

    logger.info("All inference complete.")


if __name__ == "__main__":
    main()
