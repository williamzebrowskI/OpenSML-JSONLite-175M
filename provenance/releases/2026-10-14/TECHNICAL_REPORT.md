# OpenSML-JSONLite-175M-Instruct: native JSON technical report

**Author:** William Zebrowski · **Parameters:** 175,611,648 · **Date:** 2026-10-14

## 1. Current artifact and behavior

The main checkpoint directly generates JSON from source text and a supplied JSON Schema. A single native autoregressive request emits the entire object. There is no fieldwise extraction-to-JSON assembler, answer repair, constrained decoder or gold-conditioned filling in the current inference path. Parsing and schema validation happen only after generation; the raw generated string is retained even when incorrect or invalid.

Current artifact: **main checkpoint**, BF16 `model.safetensors`, 351,242,342 bytes.
SHA-256: `655fb5cfba009163cd8a7f7fe7b170b998bb452c217b1b473cfe10b33de0ccd1`.
This is a custom native MLX architecture, not a Transformers/MLX-LM auto-loader export.

This report supersedes the earlier fieldwise SFT-093 release. Its historical report is retained as `HISTORICAL_SFT093_REPORT.md`; those adapter-assisted scores are not current native JSON scores. No release-readiness or arbitrary-schema reliability claim is made.

## 2. Exact inherited lineage

`Stage A → grounded QA → requested-field extraction → focused extraction → native JSON foundation → additional extraction SFT`

Only those selected checkpoints are inherited. Other trials and failed RL/JSON experiments are research history, not hidden additional ancestors. The architecture and frozen tokenizer remain unchanged. Each SFT stage resets its local counters and AdamW optimizer while continuing the inherited neural weights.

| Stage | Inherited update | Presentations through selected artifact | Serialized positions |
| --- | ---: | ---: | ---: |
| Stage A pretrained | 132005 | Document stream | 17,300,111,360 lifetime pretraining positions |
| SFT-065 grounded QA | 469 of 782 | 15,008 | 3,097,331 |
| SFT-068 requested-field extraction | 379 | 12,115 | 1,842,593 |
| SFT-093 focused extraction/replay | 160 | 5,120 | 706,399 |
| SFT-100 native JSON | 469 | 15,000 | 4,634,126 |
| Additional extraction SFT | 64 selected of 96 completed | 2,048 | 600,558 |
| Inherited post-training total | 1,541 updates | 49,291 | 10,881,007 |

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

The exact corpus specification is [`configs/corpus_20b.json`](../provenance/LINEAGE.json). Training preserves source text and code indentation. NFC/whitespace normalization is used for identity hashing and partitioning, not to rewrite model input.

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

![Stage A pretraining training loss, EMA and held-out validation loss](figures/pretraining-training-loss.png)

*Training loss and held-out validation during pretraining. Left: raw training loss and exponential moving average. Right: held-out validation loss.*


## 7. Supervised foundation and additional extraction SFT

The four inherited foundation stages remain unchanged: grounded evidence QA (15,008 presentations through its selected checkpoint), requested-field SNIPS extraction (12,115), focused single-field extraction/replay(5,120), then directly supervised native JSON (15,000). Their original source and checkpoint records remain in provenance. Each stage uses full model weights, fresh AdamW, BF16 computation with FP32 master weights/moments/gradient accumulation, and assistant-output-only supervision. No RL, LoRA or added extraction head is in this lineage.

The native JSON foundation has 12,115 distinct SNIPS TRAIN requests, with 12,115 canonical string-array schemas, 1,443 renamed-key views and 1,442 nullable-string views. Source text, field definitions and the complete schema are input; complete JSON and stop tokens are the assistant target. Original CC0 SNIPS chunk annotations provide source spans. Metadata fidelity and schema validation do not establish perfect semantic annotation. Frozen tokenizer and architecture are unchanged.

### Additional extraction SFT

This completed continuation starts from the native JSON foundation. It adds matched zero/one/two/three-value supervision for artists and cities, including zero-primary-value cases where an album or country is explicitly present. It retains original JSON TRAIN replay and adds frozen-parent distribution distillation on replay assistant tokens. Every model batch has 24 replay and 8 new views. Public checkpoint name: **main**; the neural tensor is `model.safetensors`.

| Setting | Value |
|---|---|
| Prepared corpus |3,072 presentations: 2,304 original TRAIN replay +768 authored schema views|
| Distinct authored source requests in prepared corpus |256|
| Prepared replay variants |1,344 canonical, 480 renamed, 480 nullable|
| Completed experiment |96 optimizer updates, one pass, 900,453 serialized positions|
| Inherited selected prefix |2,048 presentations, 600,558 positions, 64 updates|
| Inherited prefix mixture |1,536 replay views from 1,482 source texts +512 authored views from 250 source texts|
| Model adaptation |Full weights; Mac 1 only; fresh AdamW|
| Batch / microbatch |32 /2|
| Peak / final schedule LR |2e-6 /2e-7 over 96 updates|
| Warmup |6 updates; cosine afterward|
| Betas / epsilon |(0.9,0.95) /1e-8|
| Weight decay / clipping |0.01 /1.0|
| Seed |20261119|

The main checkpoint inherits only the selected prefix; updates after it are not part of these weights. The endpoint's actual learning rate is recorded by the cosine schedule, not the planned final LR. Presentation counts include multiple schemas for the same text and do not mean distinct user queries.

For every row, CE is averaged over supervised assistant tokens within each sequence, then averaged across sequences. On original TRAIN replay add 1.0×KL(frozen foundation || student), using the full 32,000-token distribution, temperature 1 and teacher-forced ground-truth histories. Only assistant JSON and turn/document-stop positions participate. The reference network is frozen/stop-gradient; new authored rows use CE alone. This regularizer is training-only: no teacher is needed for inference, and it does not guarantee retention or repair parent errors.

The new views use literal original SNIPS TRAIN values with split-disjoint authored value pools/phrases, excluding observed evaluation source texts and holding new authored DEV/TEST values out of replay. Source text, exact offsets, schema/gold agreement, masks, token budgets and hashes were checked. Twenty-eight tests, GPU accumulation checks, frozen-reference checks and a temporary optimizer/checkpoint-resume smoke passed. Authored examples are programmatic controlled supervision, not independently collected user traffic. SGD, MIT Movie and other newly researched datasets are **not** in this checkpoint.

## 8. Prompt, schema and native inference

Supply source text and a flat closed JSON Schema with 1–8 required fields. Supported properties are strings, nullable strings and arrays of strings. Every required key is returned; absent arrays target [] and absent nullable strings target null. Copy case/spelling and complete values; preserve source order. Nested schemas, numbers/booleans, enums and advanced constraints are outside the verified runner contract. The supplied runner rejects overlong inputs without truncation.

The fixed instruction is followed by compact `{"source_text":...,"schema":...}` and native SFT role tokens. It issues exactly one autoregressive whole-object generation with temperature 0, repetition penalty 1 and at most 256 output tokens. No fieldwise assembly, repair, schema-constrained decoder, normalization, reference answer, frozen teacher or secondary model is used at inference. Parsing/schema checks only annotate the raw generated text afterward.

## 9. Evaluation protocol and selection

The continuation checks 404 fixed development requests plus 54 local playground fixtures at its scheduled checkpoints. DEV contains 252 original native JSON controls, 64 previously observed combined-request controls, 8 earlier single-request probes, 32 earlier authored list controls and 48 new cardinality controls. Metrics include exact requests/fields, precision/recall/F1, present/missing fields, false/missed abstentions, unsupported values, strict parsing, schema validity, stopping/budget failures, paired requests and cardinality-group correctness. These panels have prior research exposure; no new blind/general extraction benchmark claim is made.

The main checkpoint is an owner-selected research tradeoff. Strict automatic retention gates did not pass: some domain/variant unsupported values and nullable false abstentions increased. The run's automatic best pointer remains its starting baseline; the published main is explicitly the selected trained checkpoint, not a claim that the automatic gate accepted it. Larger original TEST comparison was performed after the candidate was nominated from development/playground results. Subsequent owner release choice observed that comparison. The original 1,444 panel also has earlier research exposure.

### Main checkpoint versus previous native JSON release

Identical original 1,444 source/schema requests, tokenizer, native prompt, raw greedy decoding, 256-token budget and scorer were used. Previous release outputs were identity-checked and rescored from their saved cache. The main checkpoint generated every request, with no dropped cases or token limits. Complete correctness requires natural stopping, strict JSON/schema validity and exact field/value/order agreement. Value scores use(field, value) multisets; literal unsupported-value checks do not detect every wrong association.

|Metric|Previous native JSON release|Main checkpoint|
|---|---:|---:|
|Entire request correct|82.4792%|83.0332%|
|Field accuracy|91.3068%|91.3640%|
|Extracted-value F1|90.4798%|90.3963%|
|Extracted-value precision|90.5488%|90.5344%|
|Extracted-value recall|90.4110%|90.2588%|
|Missing-field accuracy|92.4640%|92.7916%|
|Schema-valid JSON|99.7922%|99.7922%|

Complete requests increase 1,191→1,199/1,444 (+8).27 previously wrong requests become correct and 19 previously correct become wrong; those 19 schema views cover 18 unique source texts. Paired source correctness increases 501→512/694. Unsupported values 30→27; false abstentions 111→107; missed missing fields 115→110. Strict parsing remains 1,442/1,444 and schema validity 1,441/1,444. These modest aggregate gains coexist with small F1/precision/recall decreases.

|Domain|Previous correct|Main correct|Requests|
|---|---:|---:|---:|
|AddToPlaylist|168|165|206|
|BookRestaurant|156|162|208|
|GetWeather|168|164|206|
|PlayMusic|149|153|208|
|RateBook|188|193|204|
|SearchCreativeWork|197|197|208|
|SearchScreeningEvent|165|165|204|

Canonical schemas: 1,163/1,388 (83.79%), 100% schema validity. Renamed-key stress 21/28; nullable stress 15/28. Playlist and weather exactness decrease, while restaurants, music and ratings increase. Do not describe this as uniform improvement or guaranteed nonregression. Full per-field counts, NLL and per-domain/variant metrics are in the results JSON.

### Development and local examples

|Panel|Previous|Main|
|---|---:|---:|
|Original 252 DEV complete|200/252|206/252|
|New 48 cardinality DEV complete|15/48|32/48|
|New two/three-value requests|0/24|14/24|
|Earlier 8 multi-value probes|0/8|7/8|
|54 local playground complete|42/54|44/54|
|54 local playground schema-valid|52/54|54/54|

Local field accuracy 124/135(91.85%); value precision 89.53%, recall 91.67%, F190.59%; missing 49/53(92.45%); unsupported values 3; all 54 stop naturally. Local precision declines from 91.03% despite higher recall/correctness. The new 48-pattern panel has only 36/48(75%) schema-valid outputs. All panels remain separate; do not pool them or claim an external leaderboard result.

## 10. Failure audit and limitations

All 19 newly regressed original TEST requests are schema-valid JSON. Manual primary categories: 4 changed/invented values, 3 shortened spans, 8 wrong field assignments, 2 omissions and 2 especially ambiguous source/schema contracts. Examples include Pete Murray becoming Peanut, a complete movie title losing its final phrase, a venue assigned as a movie, and an explicit city becoming null. A bare music title may not specify track versus album; broad entity_name definitions also conflict with exclusive annotation roles. Those ambiguity flags do not change benchmark golds or excuse clearer errors. Full-case evidence is retained in the report archive.

Multiple-value coverage improves, but errors remain: Bergen and Oslo can generate Bergen, Orange and Lake. Correct JSON structure cannot establish correct extraction. The model is primarily SNIPS-domain research software; unfamiliar fields, long documents, corrections/associations, nested schemas, arbitrary-schema reliability, general chat/coding, multilingual behavior, confidence calibration and prompt-injection robustness remain unestablished. No 95% reliability or universal release-readiness claim is made.

## 11. Packaging, verification and reproduction

The main package retains BF16 weights, frozen tokenizer and native MLX architecture. Stock-MLX standalone verification compares all 54 raw outputs and stop reasons against the selected saved playground protocol; the receipt specifies achieved coverage/parity and correctness. This is packaging parity, not a fresh blind accuracy estimate or complete 1,444-case cross-runtime parity claim.

Hugging Face supplies canonical model.safetensors, config, tokenizer, launcher, native_bundle.zip and model/report/results/provenance records. GitHub ships runnable source, curated JSONL records, preparation/training/evaluation code and audit evidence. Optimizer/packed-array binaries, private workspace paths, credentials and host network logs are excluded. Original workspace manifests record excluded files; provenance/SNAPSHOT_MANIFEST.json lists the actual shipped source inventory. Sanitized metadata hashes are separate from original metadata hashes.

The additional stage has frozen paths to historical baseline receipts and prepared-data inputs. Reproduction requires the specified parent weights, original source dependencies/layout, re-packing omitted arrays and regenerating local preflight receipts; published audit receipts are not ready-to-resume optimizer state. Native inference needs none of those training dependencies. The teacher is only used during training.

See provenance/LINEAGE.json, results/JSONLITE_RESULTS.json, pretraining/native_json_105/, examples/cases.json and reports/model-design/SFT-105_cardinality_distillation_20261014/ inside the source archive. Prior native JSON release hashes/revisions remain in provenance/RELEASE_HISTORY.json; old Git/Hub revisions remain accessible.

## 12. Attribution and licensing

Source code retains Apache-2.0; upstream data and dependencies retain their notices. Original SNIPS annotations are CC0; the matched cardinality examples are programmatically authored from TRAIN literals. The earlier SQuAD/Dolly/pretraining source notices remain in THIRD_PARTY_NOTICES.md. Source licensing does not automatically assign a trained-weight license; none is assigned by this update.
