"""
Multi-checkpoint inference — generates responses for all eval prompts.

Runs Phi-3.5 Mini at each of the three checkpoints:
  0: untuned base model
  1: after Stage 1 (Alpaca adapter from checkpoints/stage1/)
  2: after Stage 2 (JSON adapter from checkpoints/stage2/)

For each checkpoint, generates responses on both eval sets:
  - data/alpaca_eval.jsonl  → logs/responses_ckpt{N}_alpaca.jsonl
  - data/json_eval.jsonl    → logs/responses_ckpt{N}_json.jsonl

Usage:
    python src/inference/generate_responses.py --config config.yaml --checkpoints 0,1,2

Implemented in Stage 3 (checkpoints 0+1) and Stage 4 (checkpoint 2).
"""
# TODO: implement in Stage 3
