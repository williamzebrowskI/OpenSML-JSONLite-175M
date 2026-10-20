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

**A small language model that generates complete JSON directly from source text and a supplied JSON Schema.** It makes one autoregressive call; parsing/schema validation follows generation. No JSON assembly, output repair, constrained decoder or secondary model is used.

The model was fine-tuned for grounded question answering, exact and focused field extraction, and direct JSON generation. Its final joint JSON extraction stage combines original training replay, multiple-value examples and authored prose. Weights are provided as `model.safetensors`. The pretrained ancestor is [OpenSML-Base-175M](https://huggingface.co/OpenSML/OpenSML-Base-175M).

## Intended use and limitations

This is a research model for **extracting information that is explicitly stated in short English text** and placing it into one flat JSON object. Supply a schema with clear field descriptions. The verified runner accepts 1–8 required fields containing strings, nullable strings or arrays of strings. Dates and numbers are copied as text rather than normalized or converted.

**Expect extraction mistakes.** The model can shorten a name or title, miss or merge list items, confuse who or what a value belongs to, or supply a value when the right answer is empty. Unfamiliar field names and domains need their own testing. There is no established number of entities it will always extract correctly; only 3 of the 6 observed three/four/five-item probes were completely correct.

**Outside the verified scope:** nested objects, arrays of objects, numeric or Boolean outputs, optional-property logic, enums, references and advanced schema constraints. General chat, multilingual input and long-document extraction are also unverified. This describes the released runner's tested contract, not a claim that the weights can never produce these formats.

**Check both structure and meaning.** Parse the output, validate it against the schema, and check the values against the source. The reported 100% schema validity applies to the observed 1,444-request evaluation; complete correctness there was 88.02%. It is not a promise for new inputs. The instruction, source, schema and generated answer must share the 2,048-token context; the supplied runner allows up to 256 output tokens.

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

Each example below is a complete walkthrough: **Prompt instruction + Source text + JSON Schema → Model output**. Open an example to see all four parts together.

The instruction, source text and schema are the model's inputs. The model output is its observed response—not part of the prompt. These examples were verified with the released model; other requests can fail.

### Track and artist

<details open>
<summary><strong>Example 1: Track and artist — prompt, source text, schema and output</strong></summary>

#### 1. Prompt instruction

> Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.

#### 2. Source text

> Play the track Blue Lantern by Mira Sol.

#### 3. JSON Schema

This defines the requested fields and their types.

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

#### 4. Model output

The JSON below is generated by the model. It is shown here for reference and is not sent as an input.

```json
{
  "track": [
    "Blue Lantern"
  ],
  "artist": [
    "Mira Sol"
  ],
  "album": []
}
```

**Try this example locally:**

```bash
python inference.py --example 29
```

This command supplies the instruction, source text and schema automatically.

</details>


---

### Restaurant request

<details>
<summary><strong>Example 2: Restaurant request — prompt, source text, schema and output</strong></summary>

#### 1. Prompt instruction

> Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.

#### 2. Source text

> Book River Moon in Denver for four people tomorrow.

#### 3. JSON Schema

This defines the requested fields and their types.

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

#### 4. Model output

The JSON below is generated by the model. It is shown here for reference and is not sent as an input.

```json
{
  "restaurant_name": [
    "River Moon"
  ],
  "city": [
    "Denver"
  ],
  "party_size_number": [
    "four"
  ],
  "timeRange": [
    "tomorrow"
  ]
}
```

**Try this example locally:**

```bash
python inference.py --example 30
```

This command supplies the instruction, source text and schema automatically.

</details>


---

### Multiple artists

<details>
<summary><strong>Example 3: Multiple artists — prompt, source text, schema and output</strong></summary>

#### 1. Prompt instruction

> Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.

#### 2. Source text

> Play songs by Mira Sol and Theo Lane.

#### 3. JSON Schema

This defines the requested fields and their types.

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

#### 4. Model output

The JSON below is generated by the model. It is shown here for reference and is not sent as an input.

```json
{
  "artist": [
    "Mira Sol",
    "Theo Lane"
  ],
  "album": []
}
```

**Try this example locally:**

```bash
python inference.py --example 35
```

This command supplies the instruction, source text and schema automatically.

</details>


---

### Nullable fields

<details>
<summary><strong>Example 4: Nullable fields — prompt, source text, schema and output</strong></summary>

#### 1. Prompt instruction

> Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.

#### 2. Source text

> Play the last theme by Behzad Mirkhani.

#### 3. JSON Schema

This defines the requested fields and their types.

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

#### 4. Model output

The JSON below is generated by the model. It is shown here for reference and is not sent as an input.

```json
{
  "playlist": null,
  "music_item": "theme",
  "service": null
}
```

**Try this example locally:**

```bash
python inference.py --example 48
```

This command supplies the instruction, source text and schema automatically.

</details>


## Training


The base was pretrained from scratch on 17,300,111,360 token positions using five Apple Silicon Macs with Thunderbolt/RDMA distributed training. Intended mixture: 50% FineWeb-Edu, 18% DCLM-Edu, 13% English FineWiki, 9% Cosmopedia v2,10% educational Python. These are configured sampling shares, not unique-document counts. Base held-out validation loss: 2.5563359552.

![Training loss and held-out validation](model-card/pretraining-training-loss.png)

*Training loss and held-out validation during pretraining.*

The latest full-weight SFT jointly mixes original source/schema/JSON training replay, controlled multiple-value examples and authored natural-language passages across the seven SNIPS domains. The prepared corpus contains **21,144 presentations**: 15,000 original JSON views, 768 cardinality replay views and 5,376 controlled prose presentations (1,753 distinct new texts). The released main checkpoint inherits **19,200 presentations** from one deterministic shuffle: 13,640 original JSON views, 689 cardinality views and 4,871 prose views. It processes **6,153,194 serialized positions**, including **633,561 supervised assistant positions**. Later updates from the completed experiment are not included.

The run uses full-weight BF16 compute with FP32-master AdamW, batch 64, microbatch 2, betas (0.9, 0.95), epsilon 1e-8, zero weight decay and gradient clipping at 1. A fresh optimizer warms up over 20 updates to 5e-6, then follows a cosine schedule toward 1e-6 over the full 331-update experiment. Only assistant JSON and the turn-ending token contribute to mean per-request token cross-entropy. There is no teacher KL, LoRA or RL in this latest stage. Source text, schema and instruction tokens are masked. The inherited earlier cardinality stage used training-only distillation; no teacher is used at inference.

The full inherited lineage and earlier SFT data are documented in the technical report.

## Native JSON evaluation

The released model was evaluated on 1,444 SNIPS-derived source-text and JSON Schema requests using raw greedy generation:

| Metric | Released model |
| --- | ---: |
| Entire request correct | **88.02%** |
| Field accuracy | **94.17%** |
| Extracted-value F1 | **92.88%** |
| Extracted-value precision | **92.42%** |
| Extracted-value recall | **93.35%** |
| Missing-field accuracy | **95.22%** |
| Schema-valid JSON | **100.00%** |

The model answers **1,271/1,444 requests completely correctly**. Paired-source correctness is **561/694**. The evaluation records **53 false abstentions**, **73 missed missing fields**, and **24 unsupported predicted values**. All 1,444 outputs parse, validate and stop naturally; there are no output-budget failures.

The separate observed 60-case playground scores **49/60 (81.67%)**. Its original 54-case subset scores 46/54; the six added three/four/five-item probes score 3/6. Three larger-list probes fail. The compact 240-request development panel scores 178/240 (74.17%). These panels measure different scopes and must not be pooled. They have prior research exposure and are not fresh blind benchmarks or an external leaderboard.

These aggregate scores do not establish universal retention or production reliability. Checkpoint-selection checks and remaining failures are documented in the technical report.

![Complete-request correctness on separate observed panels](model-card/complete-request-exact.svg)

## Limitations

A focused English extraction research model for documented flat schemas and mostly SNIPS fields. It can shorten, omit, invent or misassign values, confuse roles, miss list members, and fail on corrections or unfamiliar definitions. Valid JSON does not prove correct extraction. Long documents, nested objects, numeric/boolean outputs, multilingual tasks, general chat and prompt-injection robustness are outside the verified scope. `success` means structural validity and natural completion, not semantic correctness.

## Released model identity and package

Selected BF16 weight SHA-256:

`781597baeb9b51d535fc27cd6f941dbc202e01d62c8d963587a3f3d565128560`

Includes `model.safetensors`, tokenizer, native MLX runner, source bundle, 60 source/schema/reference examples, raw evaluation records, checksums and provenance. Standalone verification checks generated strings and stop reasons against the saved 60-case protocol; this is packaging parity, not an independent accuracy benchmark. Source uses Apache-2.0; upstream terms apply separately; no trained-weight license is assigned by this release.
