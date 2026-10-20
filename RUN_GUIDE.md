# Run OpenSML-JSONLite-175M-Instruct: native JSON

The current weights generate the entire JSON object from source text and the
supplied schema. No fieldwise assembler, constrained decoder or repair is used.
The runnable package includes model.safetensors, tokenizer/config, native MLX
model source, a source archive, 60 examples, results and verification records.
Verified platform: Apple Silicon macOS, Python 3.11, stock MLX 0.32.3.

## Intended use and limitations

This is a research model for **extracting information that is explicitly stated in short English text** and placing it into one flat JSON object. Supply a schema with clear field descriptions. The verified runner accepts 1–8 required fields containing strings, nullable strings or arrays of strings. Dates and numbers are copied as text rather than normalized or converted.

**Expect extraction mistakes.** The model can shorten a name or title, miss or merge list items, confuse who or what a value belongs to, or supply a value when the right answer is empty. Unfamiliar field names and domains need their own testing. There is no established number of entities it will always extract correctly; only 3 of the 6 observed three/four/five-item probes were completely correct.

**Outside the verified scope:** nested objects, arrays of objects, numeric or Boolean outputs, optional-property logic, enums, references and advanced schema constraints. General chat, multilingual input and long-document extraction are also unverified. This describes the released runner's tested contract, not a claim that the weights can never produce these formats.

**Check both structure and meaning.** Parse the output, validate it against the schema, and check the values against the source. The reported 100% schema validity applies to the observed 1,444-request evaluation; complete correctness there was 88.02%. It is not a promise for new inputs. The instruction, source, schema and generated answer must share the 2,048-token context; the supplied runner allows up to 256 output tokens.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
# Authenticate if access requires it: hf auth login
hf download OpenSML/OpenSML-JSONLite-175M --local-dir OpenSML-JSONLite-175M
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
remaining errors. The current local diagnostic is 49/60, not 100% correctness.

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
python native_bundle/inference.py --weights "$PWD/model.safetensors" --example 29 --json
```

Selected weights:
`781597baeb9b51d535fc27cd6f941dbc202e01d62c8d963587a3f3d565128560`.
The current source module is inference_native/native_json.py. The old labeled
adapter, if present in the historical source archive, is not used by this
entry point. Optimizer state and the original training workspace are not needed
for inference. See the technical report for lineage, metrics and limitations.


The main checkpoint adds joint source/schema/JSON SFT with replay, cardinality and authored prose. No teacher is needed at inference. Exact lineage and settings are in checkpoint_provenance.json.
