# Third-party notices

The root Apache-2.0 license covers author-controlled source contributions. Installed dependencies and training materials retain their upstream terms. Source, datasets and trained weights have separate licensing status.

| Source | Role | Recorded declaration / attribution |
| --- | --- | --- |
| [SmolLM corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | FineWeb-Edu and Cosmopedia pretraining | ODC-BY source card; underlying material retains its own notices. Pinned revisions are in the corpus file. |
| [DCLM-Edu](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu) | Pretraining | CC-BY-4.0 source declaration. |
| [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | English pretraining | CC-BY-SA-4.0/GFDL source declaration. |
| [Python-Edu mirror](https://huggingface.co/datasets/Avelina/python-edu-cleaned) | Educational Python pretraining | Local metadata records ODC-BY; underlying code repositories and The Stack v2/SWH provenance retain their conditions. The pinned mirror card was not accessible during the report's external audit. |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Grounded QA SFT | Pranav Rajpurkar, Robin Jia and Percy Liang; CC-BY-SA-4.0. Original answer offsets were checked and a curated subset retained. Raw training corpus is not distributed. |
| [Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | Selected closed QA SFT | Databricks dataset contributors; CC-BY-SA-3.0. Raw training corpus is not distributed. |
| [SNIPS NLU benchmark](https://github.com/sonos/nlu-benchmark/tree/b86ac7f1577868c42158d0dec77db50956046696) | Exact field extraction, native JSON SFT and adapted held-out evaluation | SNIPS/Sonos contributors; CC0-1.0. Original chunk annotations transformed into requested-field views; [included license](licenses/SNIPS-CC0.txt). |
| OpenSML authored fictional facts | Controlled QA examples | Programmatically authored and checked; not independent human certification. |

MLX, NumPy, Tokenizers and jsonschema are installed dependencies, not vendored distributions in this archive. Their upstream distributions supply their license files. This repository includes author-controlled MLX model/kernel and orchestration source from the OpenSML workspace; the original first release's public source repository is [opensml-150m](https://github.com/williamzebrowskI/opensml-150m).

No IFEval, Paraloq, SOB, ARC, PIQA, HellaSwag or other experiment's scores are advertised as evaluation of these weights. Older research helpers present in dependency packages do not change the inherited training lineage.

For actual source-file pins and identity-preserving adaptations, see the technical report and `provenance/`. These declarations are attribution records, not a blanket commercial-use or model-weight redistribution determination. The public weight license remains unassigned.

## Joint SFT and structured continuation

The SFT uses official SNIPS TRAIN and MASSIVE en-US TRAIN annotations plus authored/schema-transformed tasks and inherited grounded-QA-derived text rows. MASSIVE: Amazon Science MASSIVE contributors, CC-BY-4.0; https://huggingface.co/datasets/AmazonScience/massive . The included prepared rows and their provenance identify original sources and transformations. SQuAD and Dolly declarations above apply to their derived prepared rows where present; prepared derived text is distributed here with its recorded upstream attribution. TEST annotations are used only for evaluation. Authored facts and schema frames are correlated synthetic research material.

Structured continuation adds authored JSON records, schema/instance documents, prose/schema/JSON documents and Python/JSON code alongside the original corpus mixture. It is next-token training, not benchmark-answer supervision. Dataset use does not assign a blanket trained-weight license.
