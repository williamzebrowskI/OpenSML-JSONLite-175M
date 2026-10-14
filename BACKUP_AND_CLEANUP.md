# Preservation and repository scope

The current selected artifact is native JSON SFT-100/update469. Preserved parents
and optimizer state remain in the owner workspace. This repository update does
not delete or retrain them. GitHub contains source, documentation, tokenizer,
curated SNIPS source/schema/JSON records and observed research evidence. Hugging
Face contains canonical model.safetensors and the runnable inference package.
No optimizer binaries, packed arrays, credentials or private cluster logs are
included. Git source is not a complete backup of training state.

provenance/SNAPSHOT_MANIFEST.json enumerates the shipped source artifacts;
scripts/verify_snapshot.py checks their hashes. The selected weight identity is
656f45015aab548465d93bed0b57e687497163741885da7364872851fa17e9a4.
Historical SFT-093 evidence is retained as history, not current JSON performance.
