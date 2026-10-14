# Training lineage and reproducibility

Stage A/132005 → SFT-065/469 → SFT-068/379 → SFT-093/160 → SFT-100/469.
The current final stage directly supervises source + schema → JSON output.
Its 15,000 presentations comprise 12,115 canonical, 1,443 alias and 1,442 nullable
views of 12,115 TRAIN sources. Full-weight Mac1 training: batch32/micro2, one pass,
469 updates, 5e-6 → 1e-6 LR, warmup20, cosine decay, output-only loss, fresh AdamW.
See docs/TECHNICAL_REPORT.md for the complete pretrained and supervised pipeline.

pretraining/native_json_100 contains frozen preparation, masking, evaluator,
training and verification source. Curated train/development/test JSONL and a
manifest are included. Original annotation inputs/revisions are pinned there;
packed NumPy arrays and optimizer binaries are not Git artifacts. Reproducing
training requires the corresponding preserved SFT-093 parent and original
workspace source inputs/path layout, then a newly generated preflight receipt.
Published sanitized receipts are audit evidence, not ready-to-resume state.

Development selected update469 before final testing. Research coverage:252 DEV,
1,444 TEST,54 local diagnostics. Expected answers do not enter generation. Raw
native generation is scored without assembler, repair or constrained decoding.
No discarded research run is an extra ancestor. Historical SFT-093 documents
remain explicitly historical. Do not attribute their software-assisted scores
to the current neural JSON model.
