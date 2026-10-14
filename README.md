# OpenSML-JSONLite-175M

**A small, fast field-extraction model with a schema-guided JSON adapter.**

**Author:** William Zebrowski · **Parameters:** 175,611,648 · **Selected artifact:** recreated SFT-093/update 160

[Technical report](docs/TECHNICAL_REPORT.md) · [Model card](model-card/README.md) · [Native inference](inference_native/) · [Results](results/JSONLITE_RESULTS.json) · [Training and reproducibility](TRAINING_AND_REPRODUCIBILITY.md) · [Run guide](RUN_GUIDE.md) · [Source notices](SOURCE_NOTICE.md)

JSONLite extracts exact values from short English requests. Its software adapter reads a supported JSON Schema, asks the model about each field separately, parses the generated labeled values and assembles the JSON object. Missing fields become empty arrays.

```text
Text + supported schema → fieldwise model extraction → strict parser → validated JSON
```

The model learned labeled extraction, rather than direct JSON Schema → JSON generation. **The checkpoint and adapter together are the evaluated system.** It currently supports documented field definitions and flat objects whose values are arrays of strings. Arbitrary schemas, nested records and typed scalars are outside this release's supported interface.

This repository contains source, tokenizer, recipes, provenance, the detailed technical report, actual benchmark predictions and a tested native inference entry point. **Model weights are preserved locally and are not yet published in this GitHub repository.** A public weight artifact and its license are separate remaining release items. No Hugging Face repository was created for this model.

## At a glance

| Component | Configuration |
| --- | --- |
| Architecture | Causal decoder, 24 layers, width 768, SwiGLU width 2,048 |
| Attention | 12 query heads / 4 KV heads; RoPE, RMSNorm and Q/K normalization |
| Tokenizer | Frozen 32,000-token byte-level BPE, reused from the first OpenSML project |
| Context | 2,048 tokens per field call, including output budget |
| Precision | BF16 weights; no quantization |
| Pretraining exposure | 17,300,111,360 processed token positions |
| Post-training | Full-weight grounded QA, requested-field extraction, focused extraction with replay |
| Inference | Native MLX on Apple Silicon; greedy, one model call per field |
| Schema contract | Closed object, 1–8 required known fields, arrays of strings, `[]` for absence |

## Exact training lineage

```text
Stage A/132005 → SFT-065/469 → SFT-068/379 → recreated SFT-093/160
```

| Stage | Prepared data | Actually inherited |
| --- | --- | --- |
| Pretraining | FineWeb-Edu 50%, DCLM-Edu 18%, FineWiki 13%, Cosmopedia 9%, Python-Edu text 10% | 17.30B processed positions |
| SFT-065 | 25,000 grounded QA records from SQuAD, Dolly and checked fictional examples | Selected update 469: 15,008 presentations / 3,097,331 serialized tokens |
| SFT-068 | 12,115 curated original SNIPS extraction queries | Update 379: 12,115 presentations / 1,842,593 tokens |
| SFT-093 | 2,560 SNIPS source queries, each with replay and focused views | Update 160: 5,120 presentations / 706,399 tokens |

Only these selected stages are ancestors. The many supporting research implementations in the source archive are not additional training stages in these weights. No RL or adapter-only training is part of this chain. Counts represent processed positions/presentations, not globally unique passages. The [report](docs/TECHNICAL_REPORT.md) gives source pins, selected versus completed boundaries, masks, hyperparameters and failed retention checks.

## Results

| Evaluation panel | Parent SFT-068/379, same adapter | JSONLite SFT-093/160 |
| --- | --- | --- |
| Observed development complete JSON exact | 139/196 — 70.92% | **168/196 — 85.71%** |
| Observed SNIPS held-out complete JSON exact | 953/1,388 — 68.66% | **1,132/1,388 — 81.56%** |
| Held-out field accuracy | 83.54% | **91.04%** |
| Held-out value precision / recall / F1 | 77.05% / 93.61% / 84.53% | **88.64% / 92.60% / 90.58%** |
| Held-out missing-field accuracy | 70.64% | **89.03%** |
| Local playground diagnostic | — | **29/40 — 72.50%** |
| Newly authored portion of playground diagnostic | — | **6/12 — 50.00%** |

![Complete-request correctness through the same JSON adapter](docs/figures/complete-request-exact.svg)

These are observed, adapted extraction panels, not an official SNIPS intent-classification leaderboard or a fresh blind general text-to-JSON benchmark. The held-out panel comprises 694 source queries with two requested-field views each. The 40-case diagnostic includes 28 observed development views and 12 authored requests. Golds are not model inputs. Decoding is temperature 0, repetition penalty 1, up to 160 tokens per field; no prompt truncation.

Improved aggregate exact correctness coexists with regressions: held-out present-field correctness falls from 1,759/1,879 to 1,740/1,879, and false abstentions increase from 18 to 69. The recreated checkpoint fails the predeclared per-category retention gates and was retained at the owner's request. Schema validity is not factual correctness. General chat, coding, IFEval and likelihood scores from other OpenSML checkpoints are not scores for JSONLite.

## Example

Input: `Play the track Blue Lantern by Mira Sol.`

```json
{
  "type": "object",
  "properties": {
    "track": {"type": "array", "items": {"type": "string"}, "description": "track name"},
    "artist": {"type": "array", "items": {"type": "string"}, "description": "performer or musical artist"},
    "album": {"type": "array", "items": {"type": "string"}, "description": "album name"}
  },
  "required": ["track", "artist", "album"],
  "additionalProperties": false
}
```

Actual result:

```json
{"track":["Blue Lantern"],"artist":["Mira Sol"],"album":[]}
```

The [40 examples](examples/cases.json) include schemas, expected answers and disclosed failures. Examples and expected answers are caller-side test material; generation receives only source text and schema-derived field prompts.

## Native inference

Use Apple Silicon and Python 3.11+. Once you have the matching local weight file:

```bash
git clone https://github.com/williamzebrowskI/OpenSML-JSONLite-175M.git
cd OpenSML-JSONLite-175M
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_snapshot.py
python inference.py --weights /path/to/model.safetensors --example 29
```

The runner verifies the weight SHA-256, architecture and frozen tokenizer. It reports extracted JSON, labeled intermediate outputs, stop reason and exact equality when using a built-in example. Custom requests use `--text-file` and `--schema-file`. Correctness references never enter generation. See [RUN_GUIDE.md](RUN_GUIDE.md) and [inference verification](inference_native/INFERENCE_VERIFICATION.json).

There is no public weight download URL yet. This native bundle is not a Transformers AutoModel, mlx-lm or quantized export. Historical training used a custom MLX build; portable inference checks record their own runtime version and scope.

## Repository contents

| Location | Contents |
| --- | --- |
| `docs/TECHNICAL_REPORT.md` | Full pretraining/SFT report, schema contract, all metric groups and failures |
| `model-card/README.md` | Task, intended use, boundaries and selected checkpoint identity |
| `pretraining/` | Preserved model, optimization, curation, serialization and selected-stage training/evaluation source |
| `tokenizer/` | Frozen tokenizer and integrity manifest |
| `inference.py`, `inference_native/` | Portable fieldwise extraction plus JSON assembly |
| `examples/` | 40 source/schema/reference fixtures |
| `results/`, `reports/model-design/` | Summary metrics and actual observed extraction predictions |
| `provenance/` | Lineage, source pins, original/published hashes, runtime and verification |
| `scripts/verify_snapshot.py` | Repository file integrity verification |
| `LICENSE`, notices | Apache-2.0 source license and separate dataset/component notices |

Raw training datasets, packed arrays, optimizer states, private cluster receipts and weights are excluded. Historical source requires its recorded external assets and workspace prerequisites; it is not advertised as a one-command training recreation.

## Intended use and known limitations

Use JSONLite as an experimental extractor for short English requests with documented fields. It can omit additional entities, confuse fields, infer unsupported countries, mishandle corrections and alter exact spelling. It has not demonstrated reliable paragraph extraction, arbitrary field definitions, nested record binding, broad multilingual quality or calibrated confidence. The adapter's structural checks do not certify the source supports a value.

Author-controlled source contributions use [Apache-2.0](LICENSE). Dataset and dependency notices remain distinct; **a model-weight release license has not been assigned here**. See [SOURCE_NOTICE.md](SOURCE_NOTICE.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [CITATION.cff](CITATION.cff).
