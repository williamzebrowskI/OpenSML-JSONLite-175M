# Training and reproducibility

The [single detailed technical report](docs/TECHNICAL_REPORT.md) is the canonical methods document. This file indexes the reproduction records rather than restating the full report.

## Preserved lineage

Stage A/132005 → SFT-065/469 → SFT-068/379 → recreated SFT-093/160.

See [LINEAGE.json](provenance/LINEAGE.json) for the original selected weight/metadata hashes and sanitized metadata hashes. Each stage's public metadata is under `provenance/checkpoints/`. The source repository carries no model or optimizer weights. Weight identity does not change when a private path is removed from metadata; sanitized metadata necessarily gets a new file hash.

## Source and data contracts

- Pretraining mixture and pins: `configs/corpus_20b.json`.
- Grounded QA source/batching: `pretraining/qa_064/`, `pretraining/qa_065/`.
- SNIPS curation, exact chunk reconstruction, requested-field views and splits: `pretraining/extract_068/`.
- SFT-093 query selection, focused/replay views, exact order, optimizer, evaluator and verification: `pretraining/json_approaches_090/`.
- SFT-093's TRAIN-only verified bank: `provenance/SFT093_VERIFIED_BANK.json`.
- Prepared data manifests: `pretraining/data/sft_065_targeted_qa_20261013/`, `sft_068_snips_extraction_20261013/`, `sft_093_json_approaches_20261014/`.
- Pinned SNIPS source hashes: `provenance/sources/snips-source-manifest.json`.
- Source snapshot and original-versus-published hashes: `provenance/SOURCE_FILES.json`.

Supporting package names do not establish that those experiments trained the selected weights. SFT-064 supplies data for SFT-065; SFT-066 supplies helper definitions/evaluation code, not an inherited trained checkpoint. SFT-093 directly forks SFT-068/379.

The original SFT-093/160 was deleted, then recreated under an identical frozen contract at the owner's request. The original had 167/196 development exact; the preserved recreation has 168/196 and a different weight hash. Bit-identical GPU training reproduction is not established. Releasing or reproducing a different hash requires its own evaluation.

## What is reproducible here

The repository verifies its own file hashes and tokenizer. Its native inference adapter is tested against the 40 recorded fixtures, with actual values and stop reasons compared to saved playground outputs. The clean-runtime verification records its exact coverage, output agreement and versions; it does not establish full-suite or cross-framework parity.

The historical training implementations retain external prerequisites and source guards. Raw SQuAD/Dolly/SNIPS pools, prepared token/mask arrays, ancestor weight/optimizer files and private five-Mac launch/network receipts are excluded. Selected metadata and prepared manifests document their identities. Some guards reference historical assets not packaged here. This is an auditable source archive, not a one-command reconstruction of all training.

## Results and evaluation independence

`results/JSONLITE_RESULTS.json` indexes the actual completed panels and predictions. The adapted SNIPS test and development panels have prior research exposure. The newly authored playground portion is only 12 requests. None is described as a freshly sealed general extraction benchmark. Failed retention gates remain recorded. Output NLL, JSON validity and aggregate gains do not replace correct extraction or per-category retention.
