# Native MLX inference

The root `inference.py` loads only the preserved recreated SFT-093/160 hash. It
uses the shipped model, tokenizer, prompt renderer, strict parser and adapter.
The supported schema is a closed object with 1–8 required properties containing
arrays of strings and known field definitions. Missing values become `[]`.
Decoding is fixed to temperature 0, repetition penalty 1 and at most 160 tokens
per field. No reference answer is passed to model generation. One neural call
is made for each requested field. See the technical report for the glossary.

Weights are not in this Git source repository. Supply a matching local weight
file via `--weights`; a public weight download has not yet been published.
Recorded MLX runtime and portable-runner checks are in `../provenance/`.
