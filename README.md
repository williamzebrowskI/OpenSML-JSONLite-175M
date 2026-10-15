# OpenSML-JSONLite-175M-Instruct

**Author:** William Zebrowski · **175,611,648 parameters**

[Run locally](RUN_GUIDE.md) · [Technical report](docs/TECHNICAL_REPORT.md) · [Results](results/JSONLITE_RESULTS.json) · [Provenance](provenance/LINEAGE.json)

**Main checkpoint: a small language model that generates complete JSON directly from source text and a supplied JSON Schema.** It makes one autoregressive call; parsing/schema validation follows generation. No JSON assembly, output repair, constrained decoder or secondary model is used.

This release retains the grounded-QA, exact/focused extraction and native JSON foundation, then adds a focused extraction continuation with original TRAIN replay, matched zero/one/two/three-value examples and frozen-parent distillation. The display name and standard `model.safetensors` filename stay unchanged. The pretrained ancestor is [OpenSML-Base-175M](https://huggingface.co/wzebrowski/OpenSML-Base-175M).

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
hf download wzebrowski/OpenSML-JSONLite-175M --local-dir OpenSML-JSONLite-175M
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

JSON Schema:

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

Actual model output:

```json
{"track":["Blue Lantern"],"artist":["Mira Sol"],"album":[]}
```

Run: `python inference.py --example 29`.

### Restaurant request

Source text:

```text
Book River Moon in Denver for four people tomorrow.
```

JSON Schema:

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

Actual model output:

```json
{"restaurant_name":["River Moon"],"city":["Denver"],"party_size_number":["four"],"timeRange":["tomorrow"]}
```

Run: `python inference.py --example 30`.

### Multiple artists

Source text:

```text
Play songs by Mira Sol and Theo Lane.
```

JSON Schema:

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

Actual model output:

```json
{"artist":["Mira Sol","Theo Lane"],"album":[]}
```

Run: `python inference.py --example 35`.

### Nullable fields

Source text:

```text
Play the last theme by Behzad Mirkhani.
```

JSON Schema:

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

Actual model output:

```json
{"playlist":null,"music_item":"theme","service":null}
```

Run: `python inference.py --example 48`.

## Training

The base was pretrained from scratch on 17,300,111,360 token positions using five Apple Silicon Macs with Thunderbolt/RDMA distributed training. Intended mixture: 50% FineWeb-Edu, 18% DCLM-Edu, 13% English FineWiki, 9% Cosmopedia v2,10% educational Python. These are configured sampling shares, not unique-document counts. Base held-out validation loss: 2.5563359552.

![Training loss and held-out validation](docs/figures/pretraining-training-loss.png)

*Training loss and held-out validation during pretraining.*

|Adaptation stage|What is inherited|
|---|---|
|Grounded evidence QA|15,008 presentations through the selected ancestor|
|Requested-field extraction|12,115 SNIPS requests|
|Focused extraction plus replay|5,120 presentations|
|Native JSON foundation|15,000 source/schema/JSON presentations|
|Additional extraction SFT|2,048 inherited presentations: 1,536 TRAIN replay + 512 controlled cardinality views|

The additional stage completed a 3,072-presentation experiment; the main checkpoint inherits only its selected 2,048-presentation prefix and 600,558 serialized positions. All stages update full model weights. The add-on uses batch 32/micro 2, fresh AdamW, 6-update warmup, peak 2e-6 and a 96-update cosine schedule toward 2e-7. On replay assistant tokens it adds KL from the frozen native JSON foundation; new rows use gold JSON CE alone. No teacher, RL, LoRA or adapter is needed at inference. Exact settings, data separation and stage hashes are in the report/provenance. Newly researched SGD/MIT datasets are not in these weights.

## Native JSON evaluation

Same original 1,444 SNIPS-derived source/schema requests and raw greedy protocol as the preceding release:

|Metric|Main checkpoint|Previous native JSON release|
|---|---:|---:|
|Entire request correct|**83.03%**|82.48%|
|Field accuracy|**91.36%**|91.31%|
|Extracted-value F1|**90.40%**|90.48%|
|Extracted-value precision|**90.53%**|90.55%|
|Extracted-value recall|**90.26%**|90.41%|
|Missing-field accuracy|**92.79%**|92.46%|
|Schema-valid JSON|**99.79%**|99.79%|

Complete correctness is 1,199/1,444 versus 1,191/1,444: 27 improved requests and 19 regressions. Unsupported values decrease 30→27. Canonical string-array views score 1,163/1,388 (83.79%) with 100% schema validity; renamed-key 21/28, nullable 15/28. The separate 54-example playground scores 44/54 (81.48%), versus 42/54; all 54 are schema-valid. The controlled new two/three-value development panel scores 14/24, versus 0/24. These are observed research panels, not fresh blind benchmarks or a public leaderboard.

The main checkpoint was owner-selected despite strict automatic retention checks not passing. Playlist/weather exactness declines on the larger panel; aggregate precision/recall/F1 decline slightly while exact/field/missing correctness improve. The new 48-pattern panel has only 75% schema validity. Full tradeoffs and failures are documented.

![Complete-request correctness on separate observed panels](docs/figures/complete-request-exact.svg)

## Limitations

The model remains a focused SNIPS-domain extraction research model. Valid JSON does not prove correct values. It can invent, shorten, omit or misassign names, confuse venue/movie or state/country, and mishandle multiple values or nullable fields. For example, Bergen and Oslo can generate Bergen, Orange and Lake. Broader domains, long documents, corrections, arbitrary schemas, multilingual/chat/coding quality, calibration and prompt-injection robustness are unverified. `success` only means structure-valid natural completion. No blanket reliability/nonregression claim is made.

## Main checkpoint identity and package

Selected BF16 weight SHA-256:

`655fb5cfba009163cd8a7f7fe7b170b998bb452c217b1b473cfe10b33de0ccd1`

The package includes standard model.safetensors(351,242,342 bytes), config, unchanged tokenizer, source archive, examples, evaluation records and provenance. Source/archive/weight checksums are verified by the runner. The standalone verification receipt records raw-output and stop parity across 54 local fixtures; it is not an independent blind accuracy result. Historical revisions are retained for reproducibility. Source uses Apache-2.0; upstream terms apply separately; no trained-weight license is assigned by this update.
