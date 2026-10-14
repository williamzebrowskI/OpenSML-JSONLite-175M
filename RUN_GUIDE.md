# Run guide

## Verify the source snapshot

```bash
python scripts/verify_snapshot.py
```

This checks the published source/documentation/tokenizer file manifest. It does not download weights or train a model.

## Install the native runtime

Use Python 3.11+ on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The initial repository's clean inference check used stock MLX and records the exact resolved versions in `inference_native/INFERENCE_VERIFICATION.json`. Training used a different custom development build. Results on other hardware, runtime releases or quantized exports are not established.

## Supply the preserved weights

The matching BF16 weight file is 351,242,342 bytes and has SHA-256:

`d552c1380c71c7c0569e9d692ec9c46c5c94bb87a7b024e56df0f0e36df4d798`

Weights and their optimizer state are not committed to this GitHub source repository. A public download URL and weight license are pending. The portable runner only requires weights, `config.json`, tokenizer and implementation; it does not require original workspace metadata or report directories.

## Run a fixture

```bash
python inference.py --weights /path/to/model.safetensors --example 29
```

There are 40 fixtures. The example reference is used only after generation to score complete parsed JSON equality. `success: true` means natural completion and structural validity; `exact: true` is available only when a reference is present. Neither indicates correctness of unrelated custom input.

## Run custom text and schema

```bash
python inference.py --weights /path/to/model.safetensors \
  --text-file request.txt --schema-file schema.json --max-tokens 160
```

Schemas must follow the supported contract documented in the report's Section 11 and glossary. No silent truncation, schema simplification, new field mapping or fuzzy output repair is performed. Greedy defaults are fixed to the evaluated protocol. Errors reject unsupported requests. One call per field means a multi-field request may generate more than 160 total tokens.

## Training source

`pretraining/qa_065`, `pretraining/extract_068`, `pretraining/json_approaches_090` and supporting modules preserve the original recipe implementations. Prepared manifests and sanitized checkpoint contracts record original file hashes and stage boundaries. Training requires the external source datasets, parent weights, packed arrays, exact runtime and historical readiness/control receipts. Archived helpers can refer to former workspace paths; this archive is not a verified clean-machine training launch. Do not launch a full recipe expecting it to recreate the published selected checkpoint without those assets. See `TRAINING_AND_REPRODUCIBILITY.md`.

## Re-run the observed extraction benchmark

```bash
python scripts/evaluate_snips.py --weights /path/to/model.safetensors \
  --output ./new-evaluation.json
```

The included 1,388 requests preserve the original source text and gold field lists from the adapted SNIPS held-out panel. This is evaluation material under the included SNIPS CC0 notice, not a training corpus. The runner prints progress, produces full predictions and metrics, refuses to overwrite results, and never feeds gold values into model generation. The existing scores were measured in the historical workspace runtime; the new portable runner has separate 40-case parity verification, not a claimed full-suite stock-runtime rerun.
