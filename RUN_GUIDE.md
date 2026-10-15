# Run OpenSML-JSONLite-175M-Instruct: native JSON

The current weights generate the entire JSON object from source text and the
supplied schema. No fieldwise assembler, constrained decoder or repair is used.
The runnable package includes model.safetensors, tokenizer/config, native MLX
model source, a source archive, 54 examples, results and verification records.
Verified platform: Apple Silicon macOS, Python 3.11, stock MLX 0.32.3.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
# Authenticate if access requires it: hf auth login
hf download wzebrowski/OpenSML-JSONLite-175M --local-dir OpenSML-JSONLite-175M
cd OpenSML-JSONLite-175M
python -m pip install -r requirements.txt
python inference.py --example 29
python inference.py --example 29 --json
python inference.py --text-file request.txt --schema-file schema.json
```

Default stdout is the unmodified generated text. `--json` prints a result record
with raw `text`, parsed `value`, structural success, stop reason, token count and
example exact correctness when a reference exists. The reference is compared
only after generation and never enters the model input. Schema validity does
not certify extracted values. Some included examples intentionally expose
remaining errors. The current local diagnostic is 44/54, not 100% correctness.

Example source: `Play the track Blue Lantern by Mira Sol.`

```json
{"type":"object","properties":{"track":{"type":"array","items":{"type":"string"},"description":"track name"},"artist":{"type":"array","items":{"type":"string"},"description":"performer or musical artist"},"album":{"type":"array","items":{"type":"string"},"description":"album name"}},"required":["track","artist","album"],"additionalProperties":false}
```

Actual native output:

```json
{"track":["Blue Lantern"],"artist":["Mira Sol"],"album":[]}
```

Schemas must be flat closed objects with 1–8 required properties: strings,
nullable strings or arrays of strings. The training scope is primarily SNIPS
fields; unfamiliar field descriptions are accepted but are not reliable by
construction. Missing arrays use []; missing nullable strings use null. Clear
descriptions are important. Default decoding is greedy, repetition penalty 1,
max 256 output tokens. `--max-tokens` supports 1..256. Source + full schema +
fixed instruction + output budget must fit 2,048 tokens; overflow rejects.

The runner supplies the exact fixed instruction and native SFT serialization.
It checks model and source-archive SHA-256 before loading. Root tokenizer files
mirror the tokenizer inside the archive, including its integrity companions.
After downloading and installing dependencies, generation runs offline.
This custom architecture is not a Transformers or MLX-LM auto-loader export.

```bash
shasum -a 256 -c SHA256SUMS
python -m zipfile -e native_bundle.zip native_bundle
python native_bundle/scripts/verify_snapshot.py
python inference.py --weights "$PWD/model.safetensors" --example 29 --json
```

Selected weights:
`655fb5cfba009163cd8a7f7fe7b170b998bb452c217b1b473cfe10b33de0ccd1`.
The current source module is inference_native/native_json.py. The old labeled
adapter, if present in the historical source archive, is not used by this
entry point. Optimizer state and the original training workspace are not needed
for inference. See the technical report for lineage, metrics and limitations.


The main checkpoint includes an additional cardinality/replay extraction SFT stage. The fixed inference instruction/schema/one-call native JSON protocol is unchanged; no frozen teacher is needed after download. Use example35 to try two artists. Exact identities/history are in checkpoint_provenance.json.
