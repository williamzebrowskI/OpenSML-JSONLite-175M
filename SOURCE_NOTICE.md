# Source and artifact scope

The native inference package contains the model architecture, tokenizer, fixed prompt, raw decoder, validation and 60 source/schema/reference fixtures. Frozen training recipes, prepared SFT data/order, selected checkpoint metadata and evaluation records are included for inspection. No optimizer binaries, parent weights or packed pretraining pools are distributed. Training code is archival and uses the original workspace conventions; create local configuration and readiness checks before attempting reproduction. The public runner is a native MLX implementation.
