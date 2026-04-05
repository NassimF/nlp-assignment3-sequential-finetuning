# Sequential Instruction Tuning of a Small LLM: Does Stage 2 Cause Catastrophic Forgetting?

> **Course:** LLM & Agentic Systems — Graduate  
> **Assignment 3** | April 2026

---

## 1. Methodology

<!-- ~1 page -->
<!-- Cover: student model choice, Alpaca data source, imitation learning pipeline,
     teacher model setup, training design for both stages, hardware (DGX A100-80GB),
     judge model choice, evaluation protocol, hyperparameters table -->

_To be written after experiments are complete._

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

<!-- ~1 page -->
<!-- Qualitative comparison across checkpoints, failure cases,
     forgetting vs retention discussion, implications for sequential fine-tuning -->

_To be written after experiments are complete._

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

_To be written in Stage 5 after judge prompt iteration._

---

## Appendix: Full Prompt Templates

### A. Teacher-Generation Prompts (5 task types)

_See prompts/teacher_generation.yaml — filled in Stage 2._

### B. Judge Evaluation Prompt

_See prompts/judge_evaluation.yaml — filled in Stage 5._
