# OpenSML-JSONLite-175M-Instruct: native JSON technical report

Date: 2026-10-20. Main checkpoint: BF16 `model.safetensors`, 175,611,648 parameters, 351,242,342 bytes.

Weight SHA-256: `25b406a7b86e3e8ac4dd0b7f32e27590d9d11349a0148a7815c32304150ac725`.

## 1. Intended use and limitations



This is a research model for **extracting information that is explicitly stated in short English text** and placing it into one flat JSON object. Supply a schema with clear field descriptions. The verified runner accepts 1–8 required fields containing strings, nullable strings or arrays of strings. Dates and numbers are copied as text rather than normalized or converted.

**Expect extraction mistakes.** The model can shorten a name or title, miss or merge list items, confuse who or what a value belongs to, or supply a value when the right answer is empty. Unfamiliar field names and domains need their own testing. There is no established number of entities it will always extract correctly; only 3 of the 6 observed three/four/five-item probes were completely correct.

**Outside the verified scope:** nested objects, arrays of objects, numeric or Boolean outputs, optional-property logic, enums, references and advanced schema constraints. General chat, multilingual input and long-document extraction are also unverified. This describes the released runner's tested contract, not a claim that the weights can never produce these formats.

**Check both structure and meaning.** Parse the output, validate it against the schema, and check the values against the source. The reported 99.93% schema validity applies to the observed 1,444-request evaluation; complete correctness there was 89.54%. It is not a promise for new inputs. The instruction, source, schema and generated answer must share the 2,048-token context; the supplied runner allows up to 256 output tokens.


## 2. Exact inherited lineage

| Stage | Selected checkpoint | Inherited training |
| --- | --- | --- |
| Original from-scratch pretraining | 132005 | 17,300,111,360 processed positions |
| Structured continued pretraining | 132838 | 100,096,883 additional real tokens |
| One joint extraction/JSON SFT | 500 of 563 | 32,000 presentations; 1,027,385 supervised assistant/EOS positions |

Original base weight SHA-256: `5151567e266244c9e32fd6e01775192633abe227320269ecdefe28ce43e9ccc3`.
Continued-pretraining parent weight SHA-256: `57d0933d5d806e7f589a60170b5080ff46895b732be480aa953650701f4e19cc`.
Main weight SHA-256: `25b406a7b86e3e8ac4dd0b7f32e27590d9d11349a0148a7815c32304150ac725`.

The main weights inherit one SFT. Earlier SFT research recipes informed the data design but their weights and optimizers are not ancestors of this artifact. All other continued-pretraining endpoints are outside this selected lineage. Supervised SFT counters and pretraining token counters measure different things and must not be added as if equivalent.

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

Implementation evidence: [`model.py`](checkpoint_provenance.json), [`precision.py`](checkpoint_provenance.json), and the final bundle's `recipe.model` metadata. Weight count and file size are consistent with BF16 parameters plus the safetensors header. GPU allocation readings are not equivalent to file size, total process RSS or system RAM requirements.

The base initialization is recorded as `normal-0.02-residual-scaled-v1`. Earlier OpenSML-150M **weights** were not used to initialize the base; its frozen tokenizer was reused.

## 4. Tokenizer and serialization

The model uses the project's frozen 32K byte-level BPE tokenizer at `reference/opensml-150m/tokenizer/bytebpe32k_v1`. No tokenizer retraining occurs in the inherited SFT stages. Its own manifest has format `sml-v2-tokenizer-v1`; the format name is not a model-weight lineage claim.

| Tokenizer artifact | SHA-256 |
| --- | --- |
| Tokenizer manifest / contract fingerprint | 1194292 c 6 d 906 ac 19e 6 f 01 c 6 ad 2 b 42825143 fad 426 a 22 aa 8856 d 903776701ecb |
| tokenizer.json | e 095 f 0413 a 11 fa 91 fbb 58 f 37 d 00e 4 c 6 bcda 59 beb 89 b 28 f 6 f 680 db 1436 a 7e 8891 |
| tokenizer_config.json | 619 f 500553367eb 76 f 5 be 00 a 98625130ec 857 b 9 c 367561334984096 c 029 ba 2 a 9 |
| tokenizer corpus.json | e 4 c 47 b 7 b 6 aba 08 f 948 de 7 c 2 c 7 cb 94 c 58 dbd 09 dde 2 ded 6 ac 185 c 60856ee 686 d 27 |

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
| FineWeb-Edu | 50 | HuggingFaceTB/smollm-corpus / fineweb-edu-dedup | 3 ba 9 d 605774198 c 5868892 d 7 a 8 deda 78031 a 781 f | Deduplicated educational subset |
| DCLM-Edu | 18 | HuggingFaceTB/dclm-edu | dbad 8 ad 71224482740 cd 9 c 9 d 353591 adbf 62 fe 04 | edu_int_score≥3 |
| FineWiki English | 13 | HuggingFaceFW/finewiki / en | 8 bd 13e 72e 6 a 002407649 b 3e 898535 f 42 ceb 1 aeb 9 | English |
| Cosmopedia v2 | 9 | HuggingFaceTB/smollm-corpus / cosmopedia-v2 | 3 ba 9 d 605774198 c 5868892 d 7 a 8 deda 78031 a 781 f | Format contains textbook/tutorial/blog/educational |
| Python text mirror | 10 | Avelina/python-edu-cleaned / default | 3 fedf 55e 02e 72 dd 1 fd 68963 f 991 f 5e 91 adcb 0 f 8 d | int_score≥4 |

Public source cards: [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus/blob/3ba9d605774198c5868892d7a8deda78031a781f/README.md), [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/blob/dbad8ad71224482740cd9c9d353591adbf62fe04/README.md), [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki/blob/8bd13e72e6a002407649b3e898535f42ceb1aeb9/README.md). The Python-Edu upstream subset identifies files; this run used the configured text mirror. The mirror's pinned README could not be retrieved during report preparation, so its provenance statement here comes from the local corpus specification, not a newly completed external audit.

The exact corpus specification is [`configs/corpus_20b.json`](checkpoint_provenance.json). Training preserves source text and code indentation. NFC/whitespace normalization is used for identity hashing and partitioning, not to rewrite model input.

The content-split policy hashes identity into 10,000 buckets: 97% training, 1% validation, 1% final and 1% tokenizer development. Identity checks do not establish that a supervised benchmark article never appeared elsewhere in pretraining web data.

Recorded stream settings are a 256-document accepted-document shuffle buffer, bounded 100,000-document deduplication window, 4 prefetched batches and up to 5 retry attempts. The corpus JSON's generic 10,000 shuffle value is not the active stream override. Filtering rejects documents below 200 characters, above 200,000 UTF-8 bytes, with null characters, excess replacement characters or an excessive repeated-line pattern. Source exhaustion fails rather than silently recycling. Deduplication is bounded; it is not global semantic or fuzzy deduplication.

## 6. Pretraining execution, optimization and validation

Pretraining runs on the recorded five-Mac Apple Silicon cluster. Mac 1's current local hardware was independently checked during report preparation as an Apple M5 Ultra with 96 GB unified memory. Other worker descriptions below are historical cluster records, not newly queried hardware inventories.

| Mac | Recorded chip | RAM | Microbatch | MLX cache cap |
| --- | --- | --- | --- | --- |
| 1 | M5 Ultra | 96 GB | 7 | 16 GiB |
| 2 | M3 Ultra | 96 GB | 3 | 16 GiB |
| 3 | M4 Max | 36 GB | 2 | 4 GiB |
| 4 | M4 Max | 36 GB | 2 | 4 GiB |
| 5 | M4 Max | 36 GB | 2 | 4 GiB |

The physical ring is 1→2→3→5→4→1. Four accumulation microsteps and 2,048-token sequences produce `(7+3+2+2+2)×4×2048=131072` positions/update. Unequal-rank gradients are weighted by token counts. Early batch transitions mean the lifetime token count is not simply `132005×131072`.

- First 125 updates: microbatches [4,4,2,2,2], 114,688 positions/update, 14,336,000 tokens.
- From 125: microbatches [6,4,2,2,2], 131,072 positions/update.
- From 963 / 124,174,336 tokens: [7,3,2,2,2], same global batch; per-Mac cache caps recorded.
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

Regular saves occur every 100M tokens, plus best-validation and clean-stop saves. Exact checkpoint metadata is authoritative. The initial October 2 technical report's early observations, 20B target and old schedule describe history; they are superseded here for this selected artifact. Contemporary qualitative logs often showed roughly 41 k tokens/s, but no reproducible end-to-end cluster performance/energy benchmark is established by this report. The separate 42.7 k profiling candidate is not claimed as the configuration that trained this checkpoint.

### Training run

The 175.6M-parameter base was trained from random initialization using next-token prediction across the educational web, Wikipedia, Cosmopedia and Python mixture. Five Apple Silicon Macs carried out distributed pretraining over Thunderbolt/RDMA. Validation was measured every 50M tokens. The curves show a steep early loss reduction followed by slower improvement; the published base was selected using held-out validation.

![Stage A pretraining training loss, EMA and held-out validation loss](pretraining-training-loss.png)

*Training loss and held-out validation during pretraining. Left: raw training loss and exponential moving average. Right: held-out validation loss.*


## 7. Continued pretraining and joint SFT

The model starts from the from-scratch 175,611,648-parameter base at update 132005 (17,300,111,360 processed positions), then inherits **100,096,883 additional tokens of structured-data continued pretraining**, followed by **one joint full-weight SFT**. It does not inherit weights from earlier extraction SFT experiments.

The continuation samples 25% web, 9% DCLM, 6.5% English wiki, 4.5% Cosmopedia, 5% educational Python, 5% JSON records, 15% schema/instance documents, 20% prose/schema/JSON documents and 10% Python working with JSON. These are configured token-mixture shares. Continuation uses next-token loss over all real document/EOS tokens, BF16 compute and imported FP32-master AdamW state. It warms from 1e-5 to 3.221225472e-5 over 5M additional tokens, then holds that rate. The selected continuation checkpoint contains 100,096,883 additional tokens; later continuation checkpoints are not inherited. Small fixed next-token validation panels are not extraction-accuracy benchmarks.

The joint SFT prepares **36,000 presentations** covering original SNIPS extraction (15,000), schema variation (3,000), grounded QA rendered as JSON (3,000), official MASSIVE TRAIN slot extraction (3,000), authored prior tasks (2,400), atomic JSON tasks (3,200) and composite JSON tasks (6,400). Atomic/composite tasks cover types, missing and optional values, lists, role assignment, corrections, nested records, references, composition and constraints. Coverage in training does not establish reliable mastery of each feature.

The main checkpoint is selected at update 500 of a 563-update experiment, inheriting **32,000 presentations from 21,392 distinct source texts**. Checkpoint metadata records **1,027,385 supervised assistant/EOS positions**; this is not a count of all serialized input positions. The exact selected-prefix component counts and source hashes are in `provenance/SELECTED_PREFIX.json`.

SFT uses a fresh full-weight AdamW optimizer, BF16 compute with FP32 master weights/moments, batch 64, microbatch 2, betas (0.9, 0.95), epsilon 1e-8, no weight decay, clipping at 1, and seed 202610180142. The learning rate warms for 25 updates to 3e-5, then follows cosine decay toward 3e-6 over 563 updates. Only assistant JSON and EOS contribute to mean per-request token loss; source, schema and instruction tokens are masked. The training sequence limit is 1,536 within the 2,048-token model context. There is no teacher distillation, LoRA or RL in this SFT, and no teacher is needed for inference. The development evaluation selected update 500 before the broader four-suite evaluation.

### Selected SFT prefix

| Component | Presentations |
| --- | ---: |
| atomic | 2,844 |
| composite | 5,689 |
| grounded_qa | 2,667 |
| massive | 2,667 |
| prior_authored | 2,133 |
| schema_variation | 2,667 |
| snips | 13,333 |

## 8. Inference protocol

The native runner accepts the displayed fixed extraction instruction, source text and schema only. It emits a complete JSON object in one autoregressive call, with greedy temperature 0 and repetition penalty 1. It performs no constrained decoding, assembly or repair. Validation happens afterward. Source, instruction, schema and generated answer share a 2,048-token context; overflow is rejected without truncation. Public-runner output budget is 1–256 tokens. The runner supports the documented flat, closed string-based schema contract, not the entire JSON Schema standard. The tokenizer is unchanged. Strict parsing rejects duplicate keys and nonfinite numbers.

## 9. Evaluation

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

### Original extraction correctness by domain

| Domain | Correct / requests | Rate |
| --- | ---: | ---: |
| AddToPlaylist | 179/206 | 86.89% |
| BookRestaurant | 173/208 | 83.17% |
| GetWeather | 186/206 | 90.29% |
| PlayMusic | 169/208 | 81.25% |
| RateBook | 199/204 | 97.55% |
| SearchCreativeWork | 200/208 | 96.15% |
| SearchScreeningEvent | 187/204 | 91.67% |

## 10. Failure audit and limitations

The intended public runner contract is short English evidence extraction into one flat closed JSON object with 1–8 required string, nullable-string or string-array fields. Dates and numbers stay as copied text. Accepted input does not guarantee correct output. The model can omit or merge entities, shorten spans, confuse roles and invent unsupported values. Unfamiliar key names, descriptions and domains require their own validation. No reliable maximum entity count has been established.

Training includes broader JSON tasks, but their measured reliability remains limited. On the dedicated diagnostic, nested objects score **13/24**, role binding **5/24**, parallel groups **4/24**, numeric types **6/24**, and the missing/optional family **4/24** completely correct requests. The last number is whole-request correctness for that challenging family, not raw missing-field accuracy. Canonical and renamed schema variants can behave differently. Deep nesting, record association and optional-property decisions remain research weaknesses. The supplied flat-schema runner rejects broader schemas rather than implying verified support.

General chat, coding, multilingual quality, long documents, calibrated confidence and prompt-injection robustness are not established. Check both structure and meaning: parse, validate the schema, then compare values and roles with source evidence. `success` reports valid structure plus natural completion, not factual correctness. Constrained-decoding scores from other checkpoints do not apply to these weights; no matched constrained/unconstrained result is claimed here.

## 11. Packaging and reproducibility

Checkpoint identifiers, original and sanitized source hashes, the exact SFT data/order and selected prefix, continued-pretraining recipe, and saved generated answers are included. Optimizer binaries, packed corpus pools and parent model weights are not bundled. Reproduction needs the exact parent weight hashes, upstream data under its own terms, compatible dependencies and fresh local preflight checks. Published sanitized receipts are audit evidence, not resumable optimizer state; absolute workspace references in archival training code need local configuration. The supported self-contained runtime is `inference.py`, not a Transformers or MLX-LM auto-loader.

The source archive contains a SHA-256 snapshot manifest. `INFERENCE_VERIFICATION.json` records the live public-runner comparison against saved main-checkpoint generations. `results/JSONLITE_RESULTS.json` separates each evaluation panel and protocol. `provenance/LINEAGE.json`, `provenance/SELECTED_PREFIX.json`, and `TRAINING_AND_REPRODUCIBILITY.md` document training identity. Evaluation correctness is checked after independent generation; references never enter the inference prompt.

## 12. Attribution and licensing

See `THIRD_PARTY_NOTICES.md`, `licenses/` and the included source manifests for pinned upstream records. Original pretraining uses FineWeb-Edu, DCLM-Edu, English FineWiki, Cosmopedia and educational Python. SFT reuses annotated official TRAIN material and authored tasks, including SNIPS and MASSIVE, plus grounded-QA-derived rows. Source, upstream datasets and trained weights have separate licensing status. Apache-2.0 applies to author-controlled source contributions; no trained-weight license is assigned by this release.
