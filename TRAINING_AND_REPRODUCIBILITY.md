# Training lineage and reproducibility

The main model inherits pretrained Stage A, grounded QA, requested-field extraction,
focused extraction, native JSON foundation, and the additional cardinality/replay
SFT continuation. Exact internal stage/update identities remain in provenance/LINEAGE.json.
The selected final prefix contains2,048presentations and600,558serialized positions;
the complete experiment ran96 updates over3,072presentations, but later weights are
not inherited. No unrelated trial is an ancestor.

The add-on trains full weights with fresh AdamW, batch32/micro2,75% original TRAIN
replay and25% controlled cardinality views per batch. Frozen-foundation output KL
regularizes replay; new views use CE only. No teacher is used at inference.
Source, JSONL examples, original manifests and receipts are supplied for audit.
Packed arrays and optimizer binaries are omitted. Reproduction needs the specified
parent, original dependencies/layout, array preparation and a new local preflight;
published sanitized receipts are not ready-to-resume training state.

The main checkpoint is owner-selected after observing development/playground and
original1,444TEST tradeoffs; strict automatic retention gates did not pass. These
are previously observed panels, not blind benchmarks. Expected values do not
enter generation. See docs/TECHNICAL_REPORT.md and results/JSONLITE_RESULTS.json.
