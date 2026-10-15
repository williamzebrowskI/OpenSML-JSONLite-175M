# Native JSON inference

The main checkpoint is loaded by its weight SHA256 in config.json. Source text
and a supplied JSON Schema are serialized into one native model request. Raw
JSON text is returned, with strict parsing/schema checks afterward. No JSON
assembler, repair, constrained decoder or secondary model is used. Default:
temperature0, repetition penalty 1, max 256tokens. See RUN_GUIDE.md and the technical
report for supported flat schemas, results and limitations. Canonical weights
are distributed by the two Hugging Face JSONLite repositories.
