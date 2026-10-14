# OpenSML-JSONLite-175M-Instruct: native JSON technical report

**Author:** William Zebrowski · **Parameters:** 175,611,648 · **Date:** 2026-10-14

## 1. Current artifact and behavior

The current weights directly generate JSON from source text and a supplied JSON Schema. A single native autoregressive request emits the entire object. There is no fieldwise extraction-to-JSON assembler, answer repair, constrained decoder or gold-conditioned filling in the current inference path. Parsing and schema validation happen only after generation; the raw generated string is retained even when incorrect or invalid.

Current artifact: SFT-100/update 469, BF16 `model.safetensors`, 351,242,342 bytes.
SHA-256: `656f45015aab548465d93bed0b57e687497163741885da7364872851fa17e9a4`.
This is a custom native MLX architecture, not a Transformers/MLX-LM auto-loader export.

This report supersedes the earlier fieldwise SFT-093 release. Its historical report is retained as `HISTORICAL_SFT093_REPORT.md`; those adapter-assisted scores are not current native JSON scores. No release-readiness or arbitrary-schema reliability claim is made.

## 2. Exact inherited lineage

`Stage A/132005 → SFT-065/469 → SFT-068/379 → SFT-093/160 → SFT-100/469`

Only those selected checkpoints are inherited. Other trials and failed RL/JSON experiments are research history, not hidden additional ancestors. The architecture and frozen tokenizer remain unchanged. Each SFT stage resets its local counters and AdamW optimizer while continuing the inherited neural weights.

| Stage | Inherited update | Presentations through selected artifact | Serialized positions |
| --- | ---: | ---: | ---: |
| Stage A pretrained | 132005 | Document stream | 17,300,111,360 lifetime pretraining positions |
| SFT-065 grounded QA | 469 of 782 | 15,008 | 3,097,331 |
| SFT-068 requested-field extraction | 379 | 12,115 | 1,842,593 |
| SFT-093 focused extraction/replay | 160 | 5,120 | 706,399 |
| SFT-100 native JSON | 469 | 15,000 | 4,634,126 |
| Inherited post-training total | 1,477 updates | 47,243 | 10,280,449 |

Presentation counts include repeated source views. Serialized positions include masked prompt tokens; they are not supervised-output-token counts. The earlier 150M model weights did not initialize this model; the earlier frozen OpenSML tokenizer was reused.

## 3. Architecture and parameterization

The same architecture is retained across pretraining and all inherited SFT stages. No new extraction head, adapter or vocabulary was introduced.

| Setting | Value |
| --- | --- |
| Architecture | Causal decoder-only transformer |
| Parameters | 175,611,648 |
| Layers | 24 |
| Hidden size | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Head size | 64 |
| Attention | Grouped-query causal attention |
| Position encoding | RoPE, base 10,000 |
| Normalization | Pre-norm RMSNorm; query/key RMSNorm enabled |
| Feed-forward | SwiGLU |
| FFN hidden size | 2048 |
| Configured MLP ratio / rounding | 4.0 / multiple 256; implementation uses 2/3 sizing |
| Embedding / output projection | Tied weights |
| Projection biases | Disabled |
| Vocabulary | 32000 |
| Context per model call | 2048 |
| Attention implementation | MLX fast attention |
| FFN implementation | Packed Metal SwiGLU |
| Training CE implementation | Metal cross entropy, FP32 reductions |

Implementation evidence: [`model.py`](../provenance/LINEAGE.json), [`precision.py`](../provenance/LINEAGE.json), and the final bundle's `recipe.model` metadata. Weight count and file size are consistent with BF16 parameters plus the safetensors header. GPU allocation readings are not equivalent to file size, total process RSS or system RAM requirements.

The base initialization is recorded as `normal-0.02-residual-scaled-v1`. Earlier OpenSML-150M **weights** were not used to initialize the base; its frozen tokenizer was reused.

## 4. Tokenizer and serialization

The model uses the project's frozen 32K byte-level BPE tokenizer at `reference/opensml-150m/tokenizer/bytebpe32k_v1`. No tokenizer retraining occurs in the inherited SFT stages. Its own manifest has format `sml-v2-tokenizer-v1`; the format name is not a model-weight lineage claim.

| Tokenizer artifact | SHA-256 |
| --- | --- |
| Tokenizer manifest / contract fingerprint | 1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb |
| tokenizer.json | e095f0413a11fa91fbb58f37d00e4c6bcda59beb89b28f6f680db1436a7e8891 |
| tokenizer_config.json | 619f500553367eb76f5be00a98625130ec857b9c367561334984096c029ba2a9 |
| tokenizer corpus.json | e4c47b7b6aba08f948de7c2c7cb94c58dbd09dde2ded6ac185c60856ee686d27 |

Structural token IDs:

| ID | Token | Meaning |
| --- | --- | --- |
| 0 | `<\|pad\|>` | Padding |
| 1 | `<\|doc_end\|>` | Document/end-of-sequence marker |
| 2 | `<\|turn_start\|>` | Role/turn header |
| 3 | `<\|turn_end\|>` | End of a conversational-format turn |

`pretraining/sft/data.py` constructs explicit IDs rather than assuming a generic Hugging Face chat template. A user turn is `turn_start + encode("user\n") + content + turn_end`. An assistant target has the corresponding assistant header, response and turn-end marker; training additionally appends document-end. Generation ends its prompt at the assistant header. User text and role headers are masked from the supervised loss. Assistant content, assistant turn-end and the appended document-end are supervised.

Ordinary text must round-trip through the tokenizer and cannot emit reserved structural IDs. Special-token-looking content can be rejected by the tokenizer; the API does not promise unrestricted literal handling of all structural strings. Prompt/output limits are checked without truncating source text.

## 5. Pretraining corpus and curation

The selected base is the best retained checkpoint at update 132005 after **17,300,111,360 processed token positions**. The original budget was 20B; a continuation was configured toward 30B, but this inherited checkpoint is not a completed 20B or 30B artifact.

Source weights below are intended processed-token sampling weights, not measured unique-document shares. No claim of 17.3B globally unique corpus tokens is made.

| Source | Sampling weight (%) | Repository / configuration | Pinned revision | Additional selection |
| --- | --- | --- | --- | --- |
| FineWeb-Edu | 50 | HuggingFaceTB/smollm-corpus / fineweb-edu-dedup | 3ba9d605774198c5868892d7a8deda78031a781f | Deduplicated educational subset |
| DCLM-Edu | 18 | HuggingFaceTB/dclm-edu | dbad8ad71224482740cd9c9d353591adbf62fe04 | edu_int_score≥3 |
| FineWiki English | 13 | HuggingFaceFW/finewiki / en | 8bd13e72e6a002407649b3e898535f42ceb1aeb9 | English |
| Cosmopedia v2 | 9 | HuggingFaceTB/smollm-corpus / cosmopedia-v2 | 3ba9d605774198c5868892d7a8deda78031a781f | Format contains textbook/tutorial/blog/educational |
| Python text mirror | 10 | Avelina/python-edu-cleaned / default | 3fedf55e02e72dd1fd68963f991f5e91adcb0f8d | int_score≥4 |

Public source cards: [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md), [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/blob/dbad8ad71224482740cd9c9d353591adbf62fe04/README.md), [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki/blob/8bd13e72e6a002407649b3e898535f42ceb1aeb9/README.md). The Python-Edu upstream subset identifies files; this run used the configured text mirror. The mirror's pinned README could not be retrieved during report preparation, so its provenance statement here comes from the local corpus specification, not a newly completed external audit.

The exact corpus specification is [`configs/corpus_20b.json`](../provenance/LINEAGE.json). Training preserves source text and code indentation. NFC/whitespace normalization is used for identity hashing and partitioning, not to rewrite model input.

The content-split policy hashes identity into 10,000 buckets: 97% training, 1% validation, 1% final and 1% tokenizer development. Identity checks do not establish that a supervised benchmark article never appeared elsewhere in pretraining web data.

Recorded stream settings are a 256-document accepted-document shuffle buffer, bounded 100,000-document deduplication window, 4 prefetched batches and up to 5 retry attempts. The corpus JSON's generic 10,000 shuffle value is not the active stream override. Filtering rejects documents below 200 characters, above 200,000 UTF-8 bytes, with null characters, excess replacement characters or an excessive repeated-line pattern. Source exhaustion fails rather than silently recycling. Deduplication is bounded; it is not global semantic or fuzzy deduplication.

## 6. Pretraining execution, optimization and validation

Pretraining runs on the recorded five-Mac Apple Silicon cluster. Mac1's current local hardware was independently checked during report preparation as an Apple M5 Ultra with 96 GB unified memory. Other worker descriptions below are historical cluster records, not newly queried hardware inventories.

| Mac | Recorded chip | RAM | Microbatch | MLX cache cap |
| --- | --- | --- | --- | --- |
| 1 | M5 Ultra | 96 GB | 7 | 16 GiB |
| 2 | M3 Ultra | 96 GB | 3 | 16 GiB |
| 3 | M4 Max | 36 GB | 2 | 4 GiB |
| 4 | M4 Max | 36 GB | 2 | 4 GiB |
| 5 | M4 Max | 36 GB | 2 | 4 GiB |

The physical ring is 1→2→3→5→4→1. Four accumulation microsteps and 2,048-token sequences produce `(7+3+2+2+2)×4×2048=131072` positions/update. Unequal-rank gradients are weighted by token counts. Early batch transitions mean the lifetime token count is not simply `132005×131072`.

- First 125 updates: microbatches [4, 4, 2, 2, 2], 114,688 positions/update, 14,336,000 tokens.
- From 125: microbatches [6, 4, 2, 2, 2], 131,072 positions/update.
- From 963 / 124,174,336 tokens: [7, 3, 2, 2, 2], same global batch; per-Mac cache caps recorded.
- Stage A continuation starts from 113313 / 14,850,113,536 tokens, retaining weights, FP32 optimizer state, source cursors, counters and controller history.
- Selected base: 132005 / 17,300,111,360 tokens.

Ethernet supports SSH orchestration; the training ring uses verified Thunderbolt/RDMA links. Sequential microbatch materialization, FP32 gradient accumulation and custom Metal paths were used. Public documentation must not include private SSH identities, addresses, access credentials or machine serial numbers.

| Pretraining optimizer/schedule | Recorded setting |
| --- | --- |
| Optimizer | AdamW |
| Nominal peak LR | 3e-4 |
| Original warmup | 200M tokens, 1% of the 20B budget |
| Base schedule | Warmup/stable interval/cosine; continuation stretches nominal decay to 24B–30B |
| Adaptive controller | Factor 0.8; patience 12 evaluations; cooldown 10; minimum delta 0.002 |
| Controller LR floor | 3e-5 |
| Effective cap at selected 132005 | 3.221225472e-5 |
| Recorded controller reductions / evaluations | 9 / 242 |
| Betas | (0.9,0.95) |
| Epsilon | 1e-8 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Seed | 7337 |
| Loss | Next-token cross entropy; FP32 reductions |

The nominal 3e-4 schedule is not the effective late-run learning rate: the retained controller cap bounds it. The record does not justify a claim that the run trained at 3e-4 for all 17.3B positions.

Validation and qualitative samples occur at 50M-token boundaries. The fixed validation panel has 32 batches/source, batch 2, five sources, context 2,048: nominally 655,360 positions/event. Aggregate loss is source-mixture weighted. Validation fingerprint: `4ddd461c131d4d27d07bb07f5aab3fcb5a00d4b1202066e3d9b6ee6b3129087b`. Selected-base validation loss is **2.5563359551876785**, perplexity **12.8885**. This measures document prediction, not extraction or chat reliability.

Regular saves occur every 100M tokens, plus best-validation and clean-stop saves. Exact checkpoint metadata is authoritative. The initial October 2 technical report's early observations, 20B target and old schedule describe history; they are superseded here for this selected artifact. Contemporary qualitative logs often showed roughly 41k tokens/s, but no reproducible end-to-end cluster performance/energy benchmark is established by this report. The separate 42.7k profiling candidate is not claimed as the configuration that trained this checkpoint.

### Training run

The 175.6M-parameter base was trained from random initialization using next-token prediction across the educational web, Wikipedia, Cosmopedia and Python mixture. Five Apple Silicon Macs carried out distributed pretraining over Thunderbolt/RDMA. Validation was measured every 50M tokens. The curves show a steep early loss reduction followed by slower improvement; the published base was selected using held-out validation.

![Stage A pretraining training loss, EMA and held-out validation loss](figures/pretraining-training-loss.png)

*Training loss and held-out validation during pretraining. Left: raw training loss and exponential moving average. Right: held-out validation loss.*


## 7. Inherited supervised stages and objective

SFT-065 starts directly from Stage A/132005. Its prepared 25,000-record grounded-QA corpus contains 12,000 natural SQuAD 2.0/Dolly QA records, 5,000 contrasting questions over shared evidence, 3,000 answerability/missing-information cases, 3,000 multipart questions and 2,000 corrections. Unsupported answers target `Not stated.`. The whole run reached 782 updates; only update 469 and its 15,008 presentations are inherited.

SFT-068 uses 12,115 curated original SNIPS TRAIN requests across seven intent domains. Each supplies requested field names/definitions and source text. Assistant targets are labeled exact source spans, with `NOT_STATED` for absent fields. It completes 379 updates. Original joint-labeled accuracy is a different protocol from native JSON and is not attributed to the current weights.

SFT-093 uses 2,560 original SNIPS TRAIN queries (1,920 accepted by a TRAIN-only parent-output verification bank and 640 additional queries). Each contributes a labeled replay view and a focused single-field view, yielding 5,120 presentations. Every batch has 16 replay and 16 focused examples. This stage completes 160 updates. It was not a native JSON target stage; its later software-assisted JSON route is historical.

All four SFT stages update full weights on Mac1 with assistant-output-only, conversation-mean cross entropy. User/schema/source tokens and role headers are masked; assistant content and turn/document stops are supervised. BF16 model computation uses FP32 master AdamW weights/moments and FP32 gradient accumulation. No LoRA, frozen-layer training, RL, added entity head or weight merge is part of this lineage.

| Setting | SFT-065 | SFT-068 | SFT-093 | SFT-100 |
| --- | --- | --- | --- | --- |
| Batch / microbatch | 32 / 2 | 32 / 2 | 32 / 2 | 32 / 2 |
| Planned updates | 782 | 379 | 160 | 469 |
| Inherited selected update | 469 | 379 | 160 | 469 |
| Peak / final LR | 1e-5 / 2e-6 | 5e-6 / 1e-6 | 5e-6 / 1e-6 | 5e-6 / 1e-6 |
| Warmup updates | 20 | 20 | 12 | 20 |
| Weight decay / clip | 0.01 / 1 | 0.01 / 1 | 0.01 / 1 | 0.01 / 1 |
| Adam betas / epsilon | (0.9, 0.95) / 1e-8 | Same | Same | Same |
| LR decay | Cosine | Cosine | Cosine | Cosine |

## 8. SFT-100: source text + schema → JSON targets

The final one-pass continuation starts from preserved SFT-093/update 160 with a fresh optimizer. Every assistant target is an actual JSON object. No labeled-line replay target is mixed into this stage.

| JSON training view | Records |
| --- | ---: |
| Canonical string-array schemas | 12,115 |
| Renamed property keys with field descriptions | 1,443 |
| Nullable-string schema views | 1,442 |
| Total presentations | 15,000 |
| Distinct source requests | 12,115 |

The additional views reuse TRAIN-only sources. Original SNIPS annotation chunks at revision `b86ac7f1577868c42158d0dec77db50956046696` provide exact values and offsets. All targets are reconstructed from those annotations, compared with the source text and validated against the supplied schema. This is annotation fidelity, not a claim that every upstream annotation is independently perfect semantic truth.

Schemas are closed objects with all properties required. Arrays contain all annotated source spans in source order, or `[]` when absent. Nullable strings are used only when the gold has at most one value, with `null` for absence. Multi-value fields remain arrays. Spelling and case are copied exactly; numbers/dates are not normalized. TRAIN alias keys use `out_`; DEV/TEST use `selected_`. Gold key maps are curation/evaluation metadata only and never enter inference.

The seven domains are AddToPlaylist, BookRestaurant, GetWeather, PlayMusic, RateBook, SearchCreativeWork and SearchScreeningEvent; there are 39 trained canonical field definitions. TRAIN/DEV/TEST source-text groups do not overlap within this prepared dataset. The frozen development/test panels had previous research exposure; pretraining decontamination and perfect semantic golds are not claimed.

Packed TRAIN has 4,634,126 serialized positions, max prompt length 425 and max target length 75. No record is truncated. Scheduled DEV checks at 0, 100, 200, 300 and 469 use 252 requests: 196 canonical, 28 alias and 28 nullable views of 98 source requests. TEST contains 1,444 requests: 1,388 canonical plus 28 alias and 28 nullable views of 694 distinct sources. Packed arrays are not in the source repository; JSONL records and the preparation/masking code are included. The original data/checkpoint manifests record workspace files (including excluded packed arrays/optimizer binaries); use provenance/SNAPSHOT_MANIFEST.json for the actual distributed file inventory.

The local preflight checked target/schema/annotation agreement, evidence offsets, tokenization and assistant masks, split separation and frozen hashes. Eleven adversarial scorer/retention tests passed. GPU forward/backward accumulation was checked without optimizer updates or new checkpoints. Batch 32 uses actual microbatch 2, padding each microbatch to a multiple of 128 with no unused 64-slot fake samples. Training completed 469 updates, 15,000 presentations and 4,634,126 positions.

## 9. Native inference contract

Input is source text plus the supplied JSON Schema, serialized in a fixed extraction instruction followed by compact `{"source_text": ..., "schema": ...}`. Native role IDs construct a user turn and the assistant-generation header. The entire source/schema prompt reaches the model together. There is no canonical-field compiler that extracts one field at a time.

The portable runner accepts one to eight required properties in a closed flat object: strings, nullable strings (`["string","null"]`) or arrays with `items: {"type":"string"}`. `$schema`, when present, is draft 2020-12. Nested records, numbers, booleans, enums, references and advanced constraints are outside this verified runner contract. Unknown names/descriptions are passed directly to the neural model, but reliable unfamiliar-domain semantics are not established.

Default generation is greedy BF16 native MLX, temperature 0, repetition penalty 1, max 256 output tokens. Prompt plus output budget must fit 2,048 tokens; overflow rejects without truncation. Stop IDs are turn-end 3 or document-end 1; unexpected structural tokens or token exhaustion fail complete correctness. JSON object key order is irrelevant; array order and exact source strings matter.

The default CLI writes raw model text. `--json` returns a result record with raw `text`, parsed `value`, schema/stop status and diagnostics. Parsing is strictly post-generation; duplicate keys and nonfinite JSON numbers reject. No generated value is filled, normalized, repaired or reserialized to create the model's raw output. The example reference stays caller-side and is compared only after inference.

```bash
python inference.py --weights model.safetensors --example 29
python inference.py --weights model.safetensors --text-file request.txt --schema-file schema.json
python inference.py --weights model.safetensors --example 29 --json
```

Example 29 source: `Play the track Blue Lantern by Mira Sol.` The actual current native output is `{"track":["Blue Lantern"],"artist":["Mira Sol"],"album":[]}`. The schema names/describes those fields as arrays of strings. The model generates all keys and values in one request.

## 10. Evaluation and checkpoint selection

Selection was frozen from development before TEST generation. Rank prioritizes domain-macro complete correctness, then overall complete correctness, field accuracy, F1, schema validity, paired correctness, unsupported-value count and NLL. Eligibility requires no loss of complete-request or field correctness on any fixed domain/schema panel relative to the same-protocol raw parent at update 0. This is a limited observed-panel gate, not protection against every regression and not a comparison with the parent's old assembler.

Complete correctness requires natural completion, strict JSON parsing, schema validity and exact agreement with every gold value. Fences, trailing prose, duplicate keys, nonfinite numbers, missing/extra keys, wrong types or values, wrong array ordering and exhausted budgets fail. Field accuracy is exact per requested field. Value precision/recall/F1 use `(field,value)` multisets. Unsupported values use literal source-substring checks and do not detect every association or boundary error.

| Development update | Complete correct / 252 | Field accuracy | Value F1 | Schema-valid rate |
| --- | --- | --- | --- | --- |
| 0 | 0 (0%) | 0% | 0% | 0% |
| 100 | 88 (34.92%) | 59.11% | 63.64% | 97.22% |
| 200 | 162 (64.29%) | 80.30% | 81.03% | 99.60% |
| 300 | 177 (70.24%) | 85.76% | 84.71% | 99.21% |
| 469 selected | 200 (79.37%) | 90.40% | 89.99% | 99.60% |

### Selected candidate held-out results: native generation

| Metric | All 1,444 requests | Canonical 1,388 requests |
| --- | ---: | ---: |
| Entire request correct | 1,191 (82.48%) | 1,156 (83.29%) |
| Field accuracy | 3,193/3,497 (91.31%) | 3,074/3,347 (91.84%) |
| Value precision | 90.55% | 90.95% |
| Value recall | 90.41% | 90.90% |
| Value F1 | 90.48% | 90.92% |
| Present-field accuracy | 1,782/1,971 (90.41%) | 1,708/1,879 (90.90%) |
| Missing-field accuracy | 1,411/1,526 (92.46%) | 1,366/1,468 (93.05%) |
| False abstentions | 111 | 103 |
| Missed abstentions | 115 | 102 |
| Unsupported values | 30 | 24 |
| Strict JSON parse | 1,442 (99.86%) | 1,388 (100%) |
| Schema-valid output | 1,441 (99.79%) | 1,388 (100%) |
| Natural stops / token limits | 1,444 / 0 | 1,388 / 0 |
| Output NLL | 0.028335 | 0.026668 |

Both canonical requested-field views are correct for 501/694 source pairs (72.19%). Alias stress is 19/28 (67.86%); nullable stress is 16/28 (57.14%). These extra schemas reuse held-out source texts, not additional independent documents.

| Domain, including stress views | Complete-request exact |
| --- | ---: |
| AddToPlaylist | 168/206 (81.55%) |
| BookRestaurant | 156/208 (75.00%) |
| GetWeather | 168/206 (81.55%) |
| PlayMusic | 149/208 (71.63%) |
| RateBook | 188/204 (92.16%) |
| SearchCreativeWork | 197/208 (94.71%) |
| SearchScreeningEvent | 165/204 (80.88%) |

The automatic same-protocol raw-parent comparison completes with parent 0/1,444 exact versus candidate 1,191/1,444. The parent had not been trained to generate native JSON from this combined schema/source prompt. Its historical fieldwise-plus-assembler score is a separate protocol: 1,132/1,388 (81.56%), compared with current native canonical 1,156/1,388 (83.29%). Current canonical false abstentions and unsupported values are higher (103 and 24 versus 69 and 5), despite improved complete correctness. Do not claim universal retention or equate these different inference protocols.

### All 54 actual playground examples

The complete live SSE test keeps the displayed schema ordering, instruction, source and settings unchanged. Model outputs are stored and rescored verbatim; reference answers never enter a generation POST.

| Metric / group | Result |
| --- | ---: |
| Whole request exact | 42/54 (77.78%) |
| Strict parse / schema validity | 52/54 (96.30%) |
| Field accuracy | 118/135 (87.41%) |
| Value precision / recall / F1 | 91.03% / 84.52% / 87.65% |
| Present / missing-field correctness | 70/82 (85.37%) / 48/53 (90.57%) |
| False / missed abstentions | 3 / 5 |
| Unsupported values | 2 |
| Natural stop / token limits | 54 / 0 |
| Original observed SNIPS views | 25/28 (89.29%) |
| Previously authored natural diagnostics | 7/12 (58.33%) |
| Additional schema stress views | 10/14 (71.43%) |

These are observed adapted SNIPS and local diagnostic panels, not official SNIPS intent classification or a fresh blind general text-to-JSON benchmark. No Paraloq, SOB, JSONSchemaBench, ARC, IFEval or coding score is attributed to these exact weights. General language quality is not established by JSON schema validity. Playground output NLL was not measured through the API.

## 11. Failure evidence and intended scope

The retained live outputs include Oslo changed to unsupported Odense, two performers combined into one string, a second city omitted, a maximum rating omitted, missing Thai cuisine, restaurant type confused with restaurant name, incomplete spans, a duplicate-key object and malformed nullable-string JSON. The model is a focused small-model extraction research candidate; it is not established as a reliable arbitrary-schema or general web-document extractor.

Schema validity does not verify values. Important gaps remain in exact copying, multiple values, boundaries/associations, correction/current-value reasoning, nullable fields, unfamiliar descriptions, long passages and schema complexity. General dialogue, coding, multilingual transfer, calibrated confidence, prompt-injection robustness and comprehensive safety/privacy behavior are unverified. Release readiness remains false in the research record.

## 12. Runtime, packaging and verification

The current portable inference package is tested independently of the training workspace on Apple Silicon macOS, Python 3.11 with stock MLX 0.32.3, NumPy 2.4.2, Tokenizers 0.22.2 and jsonschema 4.26.0. Greedy raw output text and stop reasons are compared with all 54 live playground outputs. The verification artifact records coverage and measured correctness; this is packaging parity, not a fresh blind accuracy result or full 1,444-case cross-runtime parity claim.

Hugging Face provides canonical `model.safetensors`, config/tokenizer files, a root runner and `native_bundle.zip`; the archive includes the native model implementation, prompt/validator, tokenizer companions, examples and reproducibility records. GitHub contains source/documentation and provenance, with weights hosted separately. Optimizer binaries, packed NumPy arrays, host network logs and private workspace paths are excluded. Original/published metadata hashes are separated in lineage provenance.

## 13. Reproduction and evidence

The frozen source/schema/JSON records, preparation/masking code, manifests, selected training metadata, DEV evaluations, parent/candidate TEST outputs and all 54 local predictions are included. Training uses the original workspace modules and the preserved SFT-093 parent; reproducing training requires reconstructing the corresponding workspace inputs/checkpoint paths and regenerating a local preflight receipt. The downloadable inference package does not require that workspace or optimizer state. Publishing training source is not a claim that all historic pretraining raw data or discarded optimizer checkpoints are included.

See `../provenance/LINEAGE.json`, `../results/JSONLITE_RESULTS.json`, `../reports/model-design/SFT-100_native_json_20261014/`, `../pretraining/native_json_100/`, `../examples/cases.json` and `../inference_native/INFERENCE_VERIFICATION.json` in the source archive. `scripts/verify_snapshot.py` checks the source manifest; the HF root runner checks the source archive and weight identities. Root tokenizer artifacts mirror the archive's frozen tokenizer.

## 14. Attribution and licensing

Source code carries the repository's Apache-2.0 license; upstream data/dependency notices retain their terms. SNIPS original entity annotations are CC0-1.0. SQuAD/Dolly and the pretraining mixture have separate source notices recorded in `THIRD_PARTY_NOTICES.md`. Source licensing does not automatically assign a trained-weight license; the owner has not assigned one in this update. No blanket commercial-use or redistribution determination is claimed.
