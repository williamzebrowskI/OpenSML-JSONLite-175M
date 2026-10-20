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
- AmazonScience/massive
---

# OpenSML-JSONLite-175M-Instruct

**Author:** William Zebrowski · **175M parameters**

[Run locally](RUN_GUIDE.md) · [Technical report](docs/TECHNICAL_REPORT.md) · [Results](results/JSONLITE_RESULTS.json) · [Provenance](provenance/LINEAGE.json)

**A small language model that generates complete JSON directly from source text and a supplied JSON Schema.** It makes one autoregressive call; parsing/schema validation follows generation. No JSON assembly, output repair, constrained decoder or secondary model is used.

The model combines structured-data continued pretraining with one joint SFT for grounded extraction and direct schema-guided JSON generation. Weights are provided as `model.safetensors`. The pretrained ancestor is [OpenSML-Base-175M](https://huggingface.co/OpenSML/OpenSML-Base-175M).

## Intended use and limitations

This is a research model for **extracting information that is explicitly stated in short English text** and placing it into one flat JSON object. Supply a schema with clear field descriptions. The verified runner accepts 1–8 required fields containing strings, nullable strings or arrays of strings. Dates and numbers are copied as text rather than normalized or converted.

**Expect extraction mistakes.** The model can shorten a name or title, miss or merge list items, confuse who or what a value belongs to, or supply a value when the right answer is empty. Unfamiliar field names and domains need their own testing. There is no established number of entities it will always extract correctly; only 3 of the 6 observed three/four/five-item probes were completely correct.

**Outside the verified scope:** nested objects, arrays of objects, numeric or Boolean outputs, optional-property logic, enums, references and advanced schema constraints. General chat, multilingual input and long-document extraction are also unverified. This describes the released runner's tested contract, not a claim that the weights can never produce these formats.

**Check both structure and meaning.** Parse the output, validate it against the schema, and check the values against the source. The reported 99.93% schema validity applies to the observed 1,444-request evaluation; complete correctness there was 89.54%. It is not a promise for new inputs. The instruction, source, schema and generated answer must share the 2,048-token context; the supplied runner allows up to 256 output tokens.

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

Each example groups **Prompt instruction, Source text, JSON Schema and Model output** in one expandable section. Only the first three are model inputs.

### Track and artist

<details open>
<summary><strong>Example 1: Track and artist — prompt, source text, schema and output</strong></summary>

#### 1. Prompt instruction

> Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.

#### 2. Source text

> Play the track Blue Lantern by Mira Sol.

#### 3. JSON Schema

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

Recorded generation; the output is not supplied as input.

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

```bash
python inference.py --example 29
```

</details>

---

### Restaurant request

<details>
<summary><strong>Example 2: Restaurant request — prompt, source text, schema and output</strong></summary>

#### 1. Prompt instruction

> Extract information from the supplied source_text according to the supplied JSON Schema. Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; do not normalize dates or numbers. Include every required key and no other keys. For string arrays, include all matching source spans in source order, or [] when absent. For nullable strings, use the exact source span or null when absent.

#### 2. Source text

> Find a Thai restaurant in Boston.

#### 3. JSON Schema

```json
{
  "type": "object",
  "properties": {
    "cuisine": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "cuisine style"
    },
    "city": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "city or town"
    },
    "restaurant_name": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "restaurant name"
    }
  },
  "required": [
    "cuisine",
    "city",
    "restaurant_name"
  ],
  "additionalProperties": false
}
```

#### 4. Model output

Recorded generation; the output is not supplied as input.

```json
{
  "cuisine": [
    "Thai"
  ],
  "city": [
    "Boston"
  ],
  "restaurant_name": []
}
```

```bash
python inference.py --example 37
```

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

Recorded generation; the output is not supplied as input.

```json
{
  "artist": [
    "Mira Sol",
    "Theo Lane"
  ],
  "album": []
}
```

```bash
python inference.py --example 35
```

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

Recorded generation; the output is not supplied as input.

```json
{
  "playlist": null,
  "music_item": "theme",
  "service": null
}
```

```bash
python inference.py --example 48
```

</details>

## Training

The model starts from the from-scratch 175,611,648-parameter base at update 132005 (17,300,111,360 processed positions), then inherits **100,096,883 additional tokens of structured-data continued pretraining**, followed by **one joint full-weight SFT**. It does not inherit weights from earlier extraction SFT experiments.

The continuation samples 25% web, 9% DCLM, 6.5% English wiki, 4.5% Cosmopedia, 5% educational Python, 5% JSON records, 15% schema/instance documents, 20% prose/schema/JSON documents and 10% Python working with JSON. These are configured token-mixture shares. Continuation uses next-token loss over all real document/EOS tokens, BF16 compute and imported FP32-master AdamW state. It warms from 1e-5 to 3.221225472e-5 over 5M additional tokens, then holds that rate. The selected continuation checkpoint contains 100,096,883 additional tokens; later continuation checkpoints are not inherited. Small fixed next-token validation panels are not extraction-accuracy benchmarks.

The joint SFT prepares **36,000 presentations** covering original SNIPS extraction (15,000), schema variation (3,000), grounded QA rendered as JSON (3,000), official MASSIVE TRAIN slot extraction (3,000), authored prior tasks (2,400), atomic JSON tasks (3,200) and composite JSON tasks (6,400). Atomic/composite tasks cover types, missing and optional values, lists, role assignment, corrections, nested records, references, composition and constraints. Coverage in training does not establish reliable mastery of each feature.

The main checkpoint is selected at update 500 of a 563-update experiment, inheriting **32,000 presentations from 21,392 distinct source texts**. Checkpoint metadata records **1,027,385 supervised assistant/EOS positions**; this is not a count of all serialized input positions. The exact selected-prefix component counts and source hashes are in `provenance/SELECTED_PREFIX.json`.

SFT uses a fresh full-weight AdamW optimizer, BF16 compute with FP32 master weights/moments, batch 64, microbatch 2, betas (0.9, 0.95), epsilon 1e-8, no weight decay, clipping at 1, and seed 202610180142. The learning rate warms for 25 updates to 3e-5, then follows cosine decay toward 3e-6 over 563 updates. Only assistant JSON and EOS contribute to mean per-request token loss; source, schema and instruction tokens are masked. The training sequence limit is 1,536 within the 2,048-token model context. There is no teacher distillation, LoRA or RL in this SFT, and no teacher is needed for inference. The development evaluation selected update 500 before the broader four-suite evaluation.

![Original base training history](model-card/pretraining-training-loss.png)

*This figure describes the original base run. Continued pretraining and SFT are separately documented above.*

## Native JSON evaluation

| Metric | Main checkpoint |
| --- | ---: |
| Entire request correct | **89.54%** |
| Field accuracy | **94.91%** |
| Extracted-value F1 | **93.75%** |
| Extracted-value precision | **93.54%** |
| Extracted-value recall | **93.96%** |
| Missing-field accuracy | **96.13%** |
| Schema-valid JSON | **99.93%** |

The original SNIPS-derived panel contains **1,444 requests**, of which **1,293 (89.54%)** are completely correct. All 1,444 outputs parse and stop naturally; **1,443/1,444 (99.93%)** validate against the supplied schema. One output misses a required city property. Recorded counts include 49 false abstentions, 59 missed missing fields and 20 unsupported predicted values. Valid JSON is not proof of correct extraction.

The original 60-case playground scores **49/60 (81.67%)**. The expanded 108-case playground scores **78/108 (72.22%)**: 49/60 original extraction cases and 29/48 added structural probes. The separate dedicated 384-case JSON diagnostic scores **183/384 (47.66%)**, with **81.51% schema validity**. These broader structural diagnostics use a different generic instruction and include schemas outside the supplied flat-schema runner contract. They demonstrate remaining weaknesses, not verified support for arbitrary JSON Schema.

The adapted English MASSIVE TEST slot-list evaluation scores **1,565/2,974 (52.62%)** complete requests, with **50.11% slot F1** and **99.73% schema validity**. Official MASSIVE TRAIN examples were included in SFT; TEST examples were evaluation-only. This custom schema/JSON rendering is not the official MASSIVE leaderboard protocol.

Each panel measures a different scope; do not pool their scores or interchange structural-path F1, extracted-value F1 and slot F1. The authored diagnostics and original extraction panel have prior research exposure and are not fresh blind benchmarks. Some requests share source text or construction frames. References and expected answers are used only after generation. No answer hints, output repair, JSON assembly, secondary model or constrained decoding are used. The original panel, expanded playground and dedicated diagnostic allow 256 output tokens; adapted MASSIVE allows 384. All share a 2,048-token context and reject overflow without truncation.

![Complete-request correctness on separate observed panels](model-card/complete-request-exact.svg)

## Detailed limitations

The intended public runner contract is short English evidence extraction into one flat closed JSON object with 1–8 required string, nullable-string or string-array fields. Dates and numbers stay as copied text. Accepted input does not guarantee correct output. The model can omit or merge entities, shorten spans, confuse roles and invent unsupported values. Unfamiliar key names, descriptions and domains require their own validation. No reliable maximum entity count has been established.

Training includes broader JSON tasks, but their measured reliability remains limited. On the dedicated diagnostic, nested objects score **13/24**, role binding **5/24**, parallel groups **4/24**, numeric types **6/24**, and the missing/optional family **4/24** completely correct requests. The last number is whole-request correctness for that challenging family, not raw missing-field accuracy. Canonical and renamed schema variants can behave differently. Deep nesting, record association and optional-property decisions remain research weaknesses. The supplied flat-schema runner rejects broader schemas rather than implying verified support.

General chat, coding, multilingual quality, long documents, calibrated confidence and prompt-injection robustness are not established. Check both structure and meaning: parse, validate the schema, then compare values and roles with source evidence. `success` reports valid structure plus natural completion, not factual correctness. Constrained-decoding scores from other checkpoints do not apply to these weights; no matched constrained/unconstrained result is claimed here.

## Main checkpoint identity and package

BF16 weight SHA-256:

`25b406a7b86e3e8ac4dd0b7f32e27590d9d11349a0148a7815c32304150ac725`

Includes weights, the frozen tokenizer, native MLX runner, 60 source/schema/reference examples, recorded generations, four-suite evaluation evidence, training source and provenance. The package uses one raw greedy generation per request. Source uses Apache-2.0; data and dependencies retain upstream terms. No trained-weight license is assigned. See [technical report](docs/TECHNICAL_REPORT.md) and [run guide](RUN_GUIDE.md).
