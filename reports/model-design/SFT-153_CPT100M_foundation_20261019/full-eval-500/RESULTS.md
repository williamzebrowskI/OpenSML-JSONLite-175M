# Main checkpoint: four-suite evaluation

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

![Complete-request correctness on separate observed panels](complete-request-exact.svg)
