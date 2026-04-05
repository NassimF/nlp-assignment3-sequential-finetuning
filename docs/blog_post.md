# Sequential Instruction Tuning of a Small LLM: Does Stage 2 Cause Catastrophic Forgetting?

> **Course:** LLM & Agentic Systems — Graduate  
> **Assignment 3** | April 2026

---

## 1. Methodology

### 1.1 Overview

We study sequential instruction tuning of a small language model across two stages:
**Stage 1** fine-tunes on a large general-purpose instruction dataset to build broad instruction-following ability; **Stage 2** continues training on a domain-specific structured-output dataset generated via imitation learning from a stronger teacher model. The core question is whether Stage 2 specialization comes at the cost of the general competence gained in Stage 1 — a form of catastrophic forgetting.

### 1.2 Student Model

We use **Phi-3.5 Mini Instruct** (`microsoft/Phi-3.5-mini-instruct`) as the student: a 3.8B-parameter transformer that balances capability with trainability under compute constraints. Phi-3.5 Mini supports long contexts (128K tokens), uses a combined `qkv_proj` projection (not separate Q/K/V), and has an active open-source community with a well-documented chat template.

### 1.3 Stage 1 — Alpaca Fine-Tuning

We load the `yahma/alpaca-cleaned` dataset (52,002 examples after removing empty-output entries) and hold out 150 examples for evaluation (seed=42). The training set (~51K examples) covers diverse general instruction types: open-ended generation, question answering, summarization, classification, and creative writing.

**QLoRA setup:** The model is loaded in 4-bit NF4 quantization via `bitsandbytes`. A LoRA adapter (r=16, α=32, dropout=0.05) is applied to all linear layers (`target_modules="all-linear"`) via PEFT, yielding **25.2M trainable parameters** (0.65% of 3.85B total). Training uses TRL's `SFTTrainer` with `max_length=1024`, cosine LR schedule, and 3% warmup.

### 1.4 Stage 2 — Imitation Learning from Teacher JSON

**Teacher model:** We use **Llama 3.3 70B Instruct** (`llama-3.3-70b-instruct-awq`) via the UTSA OpenAI-compatible API. Our initial choice was Qwen3-235B, but that model exhausts its full 4096-token budget on internal chain-of-thought reasoning, returning `content=None` — making batch generation impractical (see Section 4).

**Dataset construction:** We generate 1,081 examples across five structured-output task types (Table 1.1). Each task type uses a dedicated prompt template with a rotatable domain or error-type slot to maximize lexical diversity. All outputs are double-validated with `json.loads()`: outer wrapper plus inner `output` field. After discarding 100 failures (mostly `json_repair`), 981 examples form the training set; 100 additional examples are held out for evaluation.

| Task Type | Description | Train | Eval |
|---|---|---|---|
| `json_extraction` | Extract fields from unstructured text into JSON | 196 | 20 |
| `schema_constrained_generation` | Generate JSON conforming to a given schema | 200 | 20 |
| `classification_json_output` | Classify input, output as JSON label+confidence | 200 | 20 |
| `json_repair` | Fix malformed JSON (type mismatches, missing keys) | 185 | 20 |
| `tool_call_generation` | Generate tool-call argument JSON for function APIs | 200 | 20 |
| **Total** | | **981** | **100** |

*Table 1.1: Teacher-generated JSON dataset composition.*

The Stage 2 adapter is trained on the merged Stage 1 model (base + Stage 1 LoRA merged via `merge_and_unload()`), then a fresh LoRA adapter (same hyperparameters) is applied and trained for 2 epochs.

### 1.5 Training Infrastructure

All training runs on **80GB NVIDIA A100 GPUs on UTSA's ARC**. SLURM batch scripts for both stages are provided in `hpc/`. GPU utilization is logged at startup via `nvidia-smi`. Training loss is recorded per logging step to CSV for loss-curve figures.

### 1.6 Hyperparameters

| Parameter | Stage 1 | Stage 2 |
|---|---|---|
| Base model | Phi-3.5 Mini Instruct (3.8B) | Phi-3.5 Mini + Stage 1 merged |
| Training data | Alpaca-cleaned (~51K) | Teacher JSON (981) |
| Epochs | 3 | 2 |
| Effective batch size | 16 (4 × 4 accum) | 16 |
| Learning rate | 2e-5 | 2e-5 |
| LR schedule | Cosine + 3% warmup | Cosine + 3% warmup |
| Max sequence length | 1024 | 1024 |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| LoRA dropout | 0.05 | 0.05 |
| Quantization | NF4 4-bit (bfloat16 compute) | NF4 4-bit (bfloat16 compute) |

*Table 1.2: Training hyperparameters.*

### 1.7 Evaluation Protocol

We evaluate three checkpoints: **ckpt0** (untuned base), **ckpt1** (after Stage 1), and **ckpt2** (after Stage 2). For each checkpoint, responses are generated on both the Alpaca eval set (150 prompts) and the JSON eval set (100 prompts) using greedy decoding (temperature=0.1, do_sample=False, max_new_tokens=512).

**Automatic metrics (Alpaca):** ROUGE-1/2/L, BERTScore F1 (distilbert-base-uncased rescaler), average output length, task completion rate (≥10 tokens).

**JSON metrics:** Validity rate (`json.loads()` success), schema compliance, exact-match accuracy, field-level F1 (for extraction tasks), error taxonomy (truncated/malformed, invalid key format, trailing content, invalid value).

**LLM-as-judge:** We use **Qwen3-235B** (`qwen3-235b-a22b-thinking-2507-fp8`) to perform pairwise comparisons across all three checkpoint pairs (0v1, 1v2, 0v2). The judge scores each response on 6 dimensions (1–5 scale): instruction following, correctness, clarity, completeness, structured output validity, and hallucination risk. Winner is determined by average score differential with a 0.3 tie threshold. Response order is randomized per call to reduce position bias.

---

## 2. Experiments

<!-- ~3 pages -->

### 2.1 Three-Checkpoint Comparison

<!-- Table 4.1 from assignment requirements -->

| Checkpoint | Alpaca Judge Win Rate | ROUGE-L | BERTScore | JSON Validity | Schema Compliance | Exact Match |
|---|---|---|---|---|---|---|
| 0: Untuned base | — | — | — | — | — | — |
| 1: After Stage 1 (Alpaca) | — | — | — | — | — | — |
| 2: After Stage 2 (Teacher JSON) | — | — | — | — | — | — |

### 2.2 Alpaca Evaluation Results

_To be filled after Stage 5._

### 2.3 JSON Structured Output Evaluation

_To be filled after Stage 5._

### 2.4 Forgetting Analysis

_To be filled after Stage 5._

### 2.5 Ablation Study

_To be filled after Stage 5._

---

## 3. Analysis

### 3.1 Does Stage 2 Cause Catastrophic Forgetting?

<!-- Fill in after results: reference Table 4.1 ΔROUGE-L and ΔBERTScore ckpt1→ckpt2,
     and judge win rate shift. Discuss whether the drop (if any) is "catastrophic"
     or within acceptable bounds. -->

_Results pending — to be filled after Stage 5._

### 3.2 Qualitative Examples

Below are representative examples illustrating how responses evolve across checkpoints.

**Example 1 — General instruction (Alpaca eval):**

| Checkpoint | Response (excerpt) |
|---|---|
| ckpt0 (base) | *(to be filled)* |
| ckpt1 (Stage 1) | *(to be filled)* |
| ckpt2 (Stage 2) | *(to be filled)* |

**Example 2 — JSON extraction task:**

| Checkpoint | Response | Valid JSON? |
|---|---|---|
| ckpt0 (base) | *(to be filled)* | — |
| ckpt1 (Stage 1) | *(to be filled)* | — |
| ckpt2 (Stage 2) | *(to be filled)* | — |

### 3.3 Failure Mode Analysis

<!-- Summarize the error taxonomy from json_eval — which error types dominate at ckpt0 and ckpt1,
     and whether ckpt2 eliminates them or shifts the failure distribution. -->

Common failure modes observed across checkpoints:

- **Truncated/malformed JSON** — model generates plausible JSON structure but closes brackets incorrectly or hits max_tokens mid-object.
- **Invalid key format** — model uses Python dict syntax (`'key'` with single quotes) instead of JSON-compliant double quotes.
- **Trailing content** — model appends explanation text after the JSON object, breaking `json.loads()`.
- **Schema drift** — model generates valid JSON but uses different key names than the reference schema.

### 3.4 Implications for Sequential Fine-Tuning

<!-- Write after seeing forgetting delta. Discuss:
     - Whether a single LR across both stages is optimal
     - Whether the data ratio (51K Alpaca vs 981 JSON) matters
     - What the ablation results suggest about the forgetting-specialization tradeoff
     - Practical guidance: when is sequential fine-tuning safe vs. when to use multi-task training -->

_To be completed after ablation results are available._

---

## 4. Prompt Engineering

### 4.1 Teacher Model Selection and Iteration

The assignment recommended Llama 3.1 70B Instruct as the teacher model. The UTSA API offered a stronger alternative — **Qwen3-235B** (`qwen3-235b-a22b-thinking-2507-fp8`), a 235-billion-parameter reasoning model. We initially selected it expecting higher-quality training examples.

**What failed:** Qwen3-235B is a *thinking* model that performs extended internal chain-of-thought reasoning before producing visible output. During testing, we observed that even with `max_tokens=4096`, the model exhausted its entire token budget on internal reasoning and returned `content=None` — no usable output at all. Disabling thinking mode via `chat_template_kwargs: {enable_thinking: false}` did not resolve the issue.

**Decision:** We switched the teacher to **Llama 3.3 70B Instruct** (`llama-3.3-70b-instruct-awq`). This is a standard instruction-following model with no thinking overhead. In testing it produced well-structured, valid JSON on the first attempt across all five task types, with a 0% failure rate on four of five task types and 8.6% failure rate on the harder `json_repair` task.

**Judge model:** Qwen3-235B was retained as the judge. For evaluation we make far fewer API calls (~300 pairwise comparisons vs ~1100 generation calls), so the slower thinking overhead is acceptable. The model's extended reasoning capability is an advantage for producing well-calibrated, nuanced scores across six evaluation dimensions.

### 4.2 Teacher Prompt Design

Each of the five task-type prompts follows a consistent structure:
1. **Role framing** — system prompt establishes the teacher as a dataset creator
2. **Task specification** — describes the structured-output task type
3. **Domain injection** — a `{domain}` or `{error_type}` slot filled per call for diversity
4. **Output schema** — explicit three-field schema (`instruction`, `input`, `output`) with requirements
5. **Validity constraint** — explicitly states the `output` field must be parseable by `json.loads()`

The domain/error-type rotation across 12–15 values per task type ensures lexical and semantic diversity in the training set. All prompts instruct the model to return raw JSON with no markdown fences.

### 4.3 Judge Prompt Design

The judge prompt went through three iterations before producing reliable, self-consistent output.

**v1 — "Which is better?":** A simple prompt asking the judge to pick the better response and explain why. The judge produced verbose natural-language justifications but no structured scores, making aggregation impossible and preventing per-dimension analysis.

**v2 — Explicit 1–5 scale:** We added explicit scoring instructions for all 6 dimensions. The judge now returned numerical scores, but winner declarations were often inconsistent with the scores themselves — the model would assign Response A higher average scores across all dimensions but still declare Response B the winner, apparently overriding its own scores with a holistic impression.

**v3 (current) — Deterministic winner derivation:** We resolved the inconsistency by giving the judge an explicit winner computation rule: calculate `avg_A` and `avg_B`, declare the winner based on a 0.3 differential threshold (ties if `|avg_A - avg_B| < 0.3`). This removes ambiguity — the judge no longer has to "decide" the winner; it follows from the scores it already assigned. We also clarified `hallucination_risk` direction (higher = better = fewer hallucinations, to match the other dimensions) and required the justification to cite specific evidence from both responses rather than generic observations.

The `structured_output_validity` dimension was given a neutral default of 3 for non-structured-output prompts so it doesn't penalize general instruction-following comparisons where JSON validity is irrelevant.

---

## Appendix: Full Prompt Templates

### A. Teacher-Generation Prompts (5 task types)

_See prompts/teacher_generation.yaml — filled in Stage 2._

### B. Judge Evaluation Prompt

_See prompts/judge_evaluation.yaml — filled in Stage 5._
