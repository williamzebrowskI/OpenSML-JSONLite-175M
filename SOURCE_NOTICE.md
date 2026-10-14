# Source and data notice

OpenSML-JSONLite-175M is extracted from William Zebrowski's OpenSML research workspace. Its inherited weights follow Stage A/132005 → SFT-065/469 → SFT-068/379 → recreated SFT-093/160. Supporting code dependencies and unused representations are not additional weight ancestors.

Author-controlled source contributions use the root Apache-2.0 license. Upstream dependencies, datasets and their underlying content retain their own notices. No model-weight license is assigned here; no weights are committed to this repository.

Pretraining uses pinned FineWeb-Edu, DCLM-Edu, English FineWiki, Cosmopedia v2 and the Python-Edu text mirror. Grounded QA uses original selected SQuAD 2.0/Dolly answers plus programmatically checked fictional examples. Extraction uses original SNIPS chunk labels. See the technical report for exact revisions, sampling, exclusions and selected exposure.

Raw training pools, packed arrays, optimizer states, credentials, SSH keys, private launch/network receipts and unrelated model benchmark scores are excluded. Public checkpoint JSON replaces private workspace paths with `[workspace]`; provenance records both the original and published hashes. Source and tokenizer identifiers remain unchanged for compatibility.

Actual SNIPS predictions and the documented 40 fixtures are included as evaluation evidence under their source notices. The recorded corpus declarations do not establish blanket rights to redistribute every underlying pretraining document or public weight licensing clearance.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [technical report](docs/TECHNICAL_REPORT.md), [training records](TRAINING_AND_REPRODUCIBILITY.md) and [CITATION.cff](CITATION.cff).

The adapted 1,388-request SNIPS held-out evaluation panel is included under its CC0 notice in `evaluation/snips-heldout.jsonl`. It remains outside training; requested-field transformations and prior evaluation exposure are disclosed.
