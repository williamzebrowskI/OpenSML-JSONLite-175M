---
language:
- en
library_name: mlx
tags:
- information-extraction
- structured-output
- json
- small-language-model
- apple-silicon
datasets:
- HuggingFaceTB/smollm-corpus
- HuggingFaceTB/dclm-edu
- HuggingFaceFW/finewiki
- Avelina/python-edu-cleaned
- databricks/databricks-dolly-15k
---
# OpenSML-JSONLite-175M

Author: William Zebrowski. Artifact: recreated SFT-093/update 160, 175,611,648 parameters. Weight SHA-256: `d552c1380c71c7c0569e9d692ec9c46c5c94bb87a7b024e56df0f0e36df4d798`.

This is a source/documentation release for an English field extractor plus JSON assembly. Weights are preserved locally, not published by this repository. Source uses Apache-2.0; no model-weight license is assigned by this card.

The model is a 24-layer decoder of width 768, 12 query heads and 4 KV heads, SwiGLU width 2,048, tied embeddings and a frozen 32K byte BPE. Context is 2,048 tokens per field call. Pretraining from random initialization processed 17.30B positions; the earlier 150M model contributed its tokenizer, not initial weights.

Lineage: Stage A/132005 → grounded QA SFT-065/469 → SNIPS extraction SFT-068/379 → focused extraction SFT-093/160. All SFT stages update the full weights using output-only loss; no RL, LoRA or direct native JSON targets occur in this inherited chain.

The adapter supports a closed object with 1–8 required properties, known definitions and arrays of strings. It runs one labeled-extraction call per field, maps `NOT_STATED` to `[]`, assembles JSON and validates structure. Arbitrary definitions, nested records and scalars are unsupported. JSON schema compliance does not establish correct values.

Observed adapted SNIPS held-out exact is 1,132/1,388 (81.56%), field accuracy 91.04%, value F1 90.58%; development exact is 168/196 (85.71%). The local diagnostic is 29/40, including 6/12 authored requests. These reused panels are not a blind general extraction benchmark or official SNIPS intent score. Precision improves while present-field recall regresses relative to the same-route parent. Predeclared category retention gates fail; preservation is an owner-selected experimental decision.

Intended use: controlled extraction experiments on short English requests in the documented domains. Known failures include omitted entities, incorrect associations/fields, unsupported inferred facts, corrections and punctuation changes. General chat, broad web extraction, safety, prompt-injection robustness, confidence calibration and other-language quality remain unestablished.

Complete methods, source revisions, all metric groups, actual failure cases and deployment boundaries are in the [technical report](../docs/TECHNICAL_REPORT.md). [Run guide](../RUN_GUIDE.md), [results](../results/JSONLITE_RESULTS.json), [notices](../THIRD_PARTY_NOTICES.md).
