"""
Stage 5 — JSON structured output evaluation.

Evaluates responses at each checkpoint on the JSON eval set:
  - JSON validity rate      (json.loads() succeeds)
  - Schema compliance rate  (required keys + correct value types)
  - Exact-match accuracy    (output == expected)
  - Field-level F1          (precision/recall per field for extraction tasks)
  - Error taxonomy          (missing fields, wrong types, invalid nesting, truncation)

Saves results to results/json_eval.csv

Implemented in Stage 5.
"""
# TODO: implement in Stage 5
