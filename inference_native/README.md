# Native JSON generation

`NativeJSON` receives source text and the full schema together, using the frozen
native SFT prompt. The model generates the entire JSON object in one request.
The runner preserves its raw output and only parses it afterward for validation.
No fieldwise extraction, canonical-field mapping, answer assembly, repair,
constrained decoder or reference answer enters generation.

Use flat closed schemas with 1–8 required properties: strings, nullable strings
and arrays of strings. The neural weights were trained primarily on 39 SNIPS
field definitions; accepting an unfamiliar description does not establish
reliable extraction. Default decoding: greedy BF16 MLX, repetition penalty 1,
max 256 output tokens, 2,048-token shared context. Overflow rejects without
truncation. See the technical report and inference verification for results.

`adapter.py` is retained as historical SFT-093 source only; the current entry
point imports `native_json.py` and does not use that adapter.
