---
language:
- en
pipeline_tag: text-generation
library_name: mlx
base_model: OpenSML/OpenSML-Base-175M
base_model_relation: finetune
tags:
- information-extraction
- structured-output
- json
- small-language-model
- apple-silicon
- from-scratch
- post-training
- research
datasets:
- HuggingFaceTB/smollm-corpus
- HuggingFaceTB/dclm-edu
- HuggingFaceFW/finewiki
- Avelina/python-edu-cleaned
- rajpurkar/squad_v2
- databricks/databricks-dolly-15k
---

# OpenSML-JSONLite-175M-Instruct

**Author:** William Zebrowski · **175M parameters**

[Run locally](RUN_GUIDE.md) · [Technical report](docs/TECHNICAL_REPORT.md) · [Results](results/JSONLITE_RESULTS.json) · [Provenance](provenance/LINEAGE.json)

**Main checkpoint: a small language model that generates complete JSON directly from source text and a supplied JSON Schema.** It makes one autoregressive call; parsing/schema validation follows generation. No JSON assembly, output repair, constrained decoder or secondary model is used.

This release retains the grounded-QA, exact/focused extraction and native JSON foundation, then adds a joint JSON extraction continuation with original TRAIN replay, multiple-value examples and authored prose. The display name and standard `model.safetensors` filename stay unchanged. The pretrained ancestor is [OpenSML-Base-175M](https://huggingface.co/OpenSML/OpenSML-Base-175M).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf download OpenSML/OpenSML-JSONLite-175M --local-dir OpenSML-JSONLite-175M
cd OpenSML-JSONLite-175M
python -m pip install -r requirements.txt
python inference.py --example 35
python inference.py --text-file request.txt --schema-file schema.json
python inference.py --example 35 --json
```

The runner inserts the fixed instruction/native role serialization. Default stdout is the unmodified generated text; `--json` adds validation/stop metadata. Example golds are checked afterward and never passed to generation. Inference is offline after download/dependency installation. This custom architecture uses the supplied native MLX runner; Transformers AutoModel and mlx-lm auto-loading are not supported exports. Verified platform: Apple Silicon macOS, Python 3.11, stock MLX 0.32.3.

## Inputs and supported schema

Supply source text and a flat object JSON Schema with 1–8 required properties and `additionalProperties:false`. Supported properties are strings, nullable strings and arrays of strings. Use clear field descriptions. Copy exact spelling/case and complete spans; preserve source order in arrays. Missing arrays target []; missing nullable strings target null. There is no date/number normalization. Nested objects, numeric/boolean outputs, enums/references/advanced constraints are outside the verified runner contract. Overlong prompts reject without truncation.

|Model setting|Value|
|---|---|
|Parameters|175,611,648|
|Layers / hidden / FFN|24 /768 /2,048|
|Attention|GQA: 12  query / 4 KV heads|
|Normalization / MLP / positions|RMSNorm+Q/K norm /SwiGLU /RoPE10,000|
|Vocabulary / shared context|Frozen 32,000-byte-BPE /2,048 tokens|
|Weights|BF16 unquantized safetensors; tied embeddings|
|Default generation|Temperature 0, repetition penalty 1, max 256 tokens|

## Verified source/schema/output examples

These four observed examples are checked against the selected model and standalone runner. They demonstrate successes within a limited research scope; other requests can fail. The runner supplies this exact instruction:

```text
Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.
```

The user message is the instruction, a blank line, then compact `{"source_text":<query>,"schema":<schema>}`. No expected answer is included.

### Track and artist

Source text:

```text
Play the track Blue Lantern by Mira Sol.
```

<details>
<summary>View JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "track": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "track name"
    },
    "artist": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "performer or musical artist"
    },
    "album": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "album name"
    }
  },
  "required": [
    "track",
    "artist",
    "album"
  ],
  "additionalProperties": false
}
```

</details>

<details>
<summary>View generated JSON</summary>

```json
{"track":["Blue Lantern"],"artist":["Mira Sol"],"album":[]}
```

</details>

Run: `python inference.py --example 29`.

### Restaurant request

Source text:

```text
Book River Moon in Denver for four people tomorrow.
```

<details>
<summary>View JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "restaurant_name": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "restaurant name"
    },
    "city": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "city or town"
    },
    "party_size_number": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "explicit party size"
    },
    "timeRange": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "date, time, or time-range expression"
    }
  },
  "required": [
    "restaurant_name",
    "city",
    "party_size_number",
    "timeRange"
  ],
  "additionalProperties": false
}
```

</details>

<details>
<summary>View generated JSON</summary>

```json
{"restaurant_name":["River Moon"],"city":["Denver"],"party_size_number":["four"],"timeRange":["tomorrow"]}
```

</details>

Run: `python inference.py --example 30`.

### Multiple artists

Source text:

```text
Play songs by Mira Sol and Theo Lane.
```

<details>
<summary>View JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "artist": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "performer or musical artist"
    },
    "album": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "album name"
    }
  },
  "required": [
    "artist",
    "album"
  ],
  "additionalProperties": false
}
```

</details>

<details>
<summary>View generated JSON</summary>

```json
{"artist":["Mira Sol","Theo Lane"],"album":[]}
```

</details>

Run: `python inference.py --example 35`.

### Nullable fields

Source text:

```text
Play the last theme by Behzad Mirkhani.
```

<details>
<summary>View JSON Schema</summary>

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "service": {
      "type": [
        "string",
        "null"
      ],
      "description": "music service"
    },
    "music_item": {
      "type": [
        "string",
        "null"
      ],
      "description": "word specifying the requested music category, such as artist, track, song, or album"
    },
    "playlist": {
      "type": [
        "string",
        "null"
      ],
      "description": "playlist name"
    }
  },
  "required": [
    "service",
    "music_item",
    "playlist"
  ],
  "additionalProperties": false
}
```

</details>

<details>
<summary>View generated JSON</summary>

```json
{"playlist":null,"music_item":"theme","service":null}
```

</details>

Run: `python inference.py --example 48`.

## Training


The base was pretrained from scratch on 17,300,111,360 token positions using five Apple Silicon Macs with Thunderbolt/RDMA distributed training. Intended mixture: 50% FineWeb-Edu, 18% DCLM-Edu, 13% English FineWiki, 9% Cosmopedia v2,10% educational Python. These are configured sampling shares, not unique-document counts. Base held-out validation loss: 2.5563359552.

![Training loss and held-out validation](model-card/pretraining-training-loss.png)

*Training loss and held-out validation during pretraining.*

The latest full-weight SFT jointly mixes original source/schema/JSON training replay, controlled multiple-value examples and authored natural-language passages across the seven SNIPS domains. The prepared corpus contains **21,144 presentations**: 15,000 original JSON views, 768 cardinality replay views and 5,376 controlled prose presentations (1,753 distinct new texts). The released main checkpoint inherits **19,200 presentations** from one deterministic shuffle: 13,640 original JSON views, 689 cardinality views and 4,871 prose views. It processes **6,153,194 serialized positions**, including **633,561 supervised assistant positions**. Later updates from the completed experiment are not included.

The run uses full-weight BF16 compute with FP32-master AdamW, batch 64, microbatch 2, betas (0.9, 0.95), epsilon 1e-8, zero weight decay and gradient clipping at 1. A fresh optimizer warms up over 20 updates to 5e-6, then follows a cosine schedule toward 1e-6 over the full 331-update experiment. Only assistant JSON and the turn-ending token contribute to mean per-request token cross-entropy. There is no teacher KL, LoRA or RL in this latest stage. Source text, schema and instruction tokens are masked. The inherited earlier cardinality stage used training-only distillation; no teacher is used at inference.

The full inherited lineage and earlier SFT data are documented in the technical report.

## Native JSON evaluation

Same original 1,444 SNIPS-derived source/schema requests and raw greedy protocol as the previous release:

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

![Complete-request correctness on separate observed panels](model-card/complete-request-exact.svg)

## Limitations

A focused English extraction research model for documented flat schemas and mostly SNIPS fields. It can shorten, omit, invent or misassign values, confuse roles, miss list members, and fail on corrections or unfamiliar definitions. Valid JSON does not prove correct extraction. Long documents, nested objects, numeric/boolean outputs, multilingual tasks, general chat and prompt-injection robustness are outside the verified scope. `success` means structural validity and natural completion, not semantic correctness.

## Main checkpoint identity and package

Selected BF16 weight SHA-256:

`781597baeb9b51d535fc27cd6f941dbc202e01d62c8d963587a3f3d565128560`

Includes `model.safetensors`, unchanged tokenizer, native MLX runner, source bundle, 60 source/schema/reference examples, raw evaluation records, checksums and provenance. Standalone verification checks generated strings and stop reasons against the saved 60-case protocol; this is packaging parity, not an independent accuracy benchmark. Previous revisions remain available. Source uses Apache-2.0; upstream terms apply separately; no trained-weight license is assigned by this update.
