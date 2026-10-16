# Training lineage and reproducibility

The latest full-weight SFT jointly mixes original source/schema/JSON training replay, controlled multiple-value examples and authored natural-language passages across the seven SNIPS domains. The prepared corpus contains **21,144 presentations**: 15,000 original JSON views, 768 cardinality replay views and 5,376 controlled prose presentations (1,753 distinct new texts). The released main checkpoint inherits **19,200 presentations** from one deterministic shuffle: 13,640 original JSON views, 689 cardinality views and 4,871 prose views. It processes **6,153,194 serialized positions**, including **633,561 supervised assistant positions**. Later updates from the completed experiment are not included.

The run uses full-weight BF16 compute with FP32-master AdamW, batch 64, microbatch 2, betas (0.9, 0.95), epsilon 1e-8, zero weight decay and gradient clipping at 1. A fresh optimizer warms up over 20 updates to 5e-6, then follows a cosine schedule toward 1e-6 over the full 331-update experiment. Only assistant JSON and the turn-ending token contribute to mean per-request token cross-entropy. There is no teacher KL, LoRA or RL in this latest stage. Source text, schema and instruction tokens are masked. The inherited earlier cardinality stage used training-only distillation; no teacher is used at inference.

Inherited lineage: Stage A/132005 → SFT-065/469 → SFT-068/379 → SFT-093/160 → SFT-100/469 → SFT-105/64 → SFT-128/300. Later updates are not inherited. Full source and prepared rows are provided; original versus sanitized hashes are separated in provenance/SOURCE_FILES.json. Training assumes the original workspace layout and preserved parent weights. Recreate preparation/preflight receipts locally; published sanitized receipts are audit evidence, not resumable optimizer state. No optimizer binaries are distributed.

| Metric | Main checkpoint | Previous release |
| --- | ---: | ---: |
| Entire request correct | **88.02%** | 83.03% |
| Field accuracy | **94.17%** | 91.36% |
| Extracted-value F1 | **92.88%** | 90.40% |
| Extracted-value precision | **92.42%** | 90.53% |
| Extracted-value recall | **93.35%** | 90.26% |
| Missing-field accuracy | **95.22%** | 92.79% |
| Schema-valid JSON | **100.00%** | 99.79% |

Complete correctness is **1,271/1,444**, compared with 1,199/1,444 for the previous release: **96 improved requests and 24 regressions**, a net gain of 72. All seven domain aggregate correctness totals improve. Paired-source correctness rises from 512/694 to **561/694**. False abstentions fall from 107 to **53**, missed missing fields from 110 to **73**, and unsupported predicted values from 27 to **24**. All 1,444 outputs parse, validate and stop naturally; there are no output-budget failures.

The separate observed 60-case playground scores **49/60 (81.67%)**, compared with 45/60. Its original 54-case subset scores 46/54; the six added three/four/five-item probes score 3/6. Three larger-list probes still fail. The compact 240-request development panel scores 178/240 (74.17%). These panels measure different scopes and must not be pooled. They have prior research exposure and are not fresh blind benchmarks or an external leaderboard.

The main checkpoint was selected for aggregate improvement despite failing the strict zero-lost-case development gate. Individual regressions remain; improved aggregate scores do not establish universal retention or production reliability.
