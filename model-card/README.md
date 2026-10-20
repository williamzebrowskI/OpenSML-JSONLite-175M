# OpenSML-JSONLite-175M-Instruct

[Current model card](../README.md) · [Technical report](../docs/TECHNICAL_REPORT.md) · [Current metrics](../results/JSONLITE_RESULTS.json)

The main checkpoint uses the latest joint JSON extraction SFT and directly generates the complete JSON object from source text and schema. Current complete-request correctness is 1,271/1,444 (88.02%) on the repeatedly observed SNIPS-derived evaluation. Full scope and remaining failures are in the report.

## Intended use and limitations

This is a research model for **extracting information that is explicitly stated in short English text** and placing it into one flat JSON object. Supply a schema with clear field descriptions. The verified runner accepts 1–8 required fields containing strings, nullable strings or arrays of strings. Dates and numbers are copied as text rather than normalized or converted.

**Expect extraction mistakes.** The model can shorten a name or title, miss or merge list items, confuse who or what a value belongs to, or supply a value when the right answer is empty. Unfamiliar field names and domains need their own testing. There is no established number of entities it will always extract correctly; only 3 of the 6 observed three/four/five-item probes were completely correct.

**Outside the verified scope:** nested objects, arrays of objects, numeric or Boolean outputs, optional-property logic, enums, references and advanced schema constraints. General chat, multilingual input and long-document extraction are also unverified. This describes the released runner's tested contract, not a claim that the weights can never produce these formats.

**Check both structure and meaning.** Parse the output, validate it against the schema, and check the values against the source. The reported 100% schema validity applies to the observed 1,444-request evaluation; complete correctness there was 88.02%. It is not a promise for new inputs. The instruction, source, schema and generated answer must share the 2,048-token context; the supplied runner allows up to 256 output tokens.
