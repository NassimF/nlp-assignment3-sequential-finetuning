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

<!-- Design process for teacher-generation prompts and judge prompts.
     Show evidence of iteration — what failed first and why. -->

_To be written in Stage 5._

---

## Appendix: Full Prompt Templates

### A. Teacher-Generation Prompts (5 task types)

_See prompts/teacher_generation.yaml — filled in Stage 2._

### B. Judge Evaluation Prompt

_See prompts/judge_evaluation.yaml — filled in Stage 5._
