# Preservation and repository scope

The preserved selected checkpoint is recreated SFT-093/update 160. Its ancestor checkpoints, tokenizer, local model and optimizer state remain in the owner's workspace. Creating this repository does not delete, resume, retrain or replace them.

This Git repository preserves source, documentation, tokenizer and selected evidence. It excludes raw/packed training data, all model and optimizer weights, credentials and private cluster receipts. It is therefore not a complete recovery backup for model weights or training state.

`provenance/SNAPSHOT_MANIFEST.json` enumerates the shipped files. `scripts/verify_snapshot.py` verifies them. Public verification of a downloaded weight artifact will additionally require its pinned hash and a separate published distribution location; that artifact is not yet uploaded.
