# Source and data notice

Current inherited weights: Stage A/132005 → SFT-065/469 → SFT-068/379 →
SFT-093/160 → SFT-100/469 → additional extraction SFT (main checkpoint). The final stage directly supervises native JSON.
Other research trials and unused helper code are not weight ancestors.
Author-controlled source uses Apache-2.0. Upstream datasets/dependencies retain
their notices; no trained-weight license is assigned by this source update.
GitHub excludes weight and optimizer binaries; the HF companion has the weights.

Pretraining uses the pinned educational web/wiki/Cosmopedia/Python mixture.
Grounded QA uses selected SQuAD2/Dolly and checked authored fictional facts.
SNIPS original CC0 chunk annotations supply exact extraction and native JSON
labels. The 15,000 curated JSON records are source/schema/target views of 12,115
TRAIN requests, not 15,000 distinct documents. Development/test are source-disjoint
from TRAIN within this curated dataset, but have previous research exposure.
Pretraining decontamination and perfectly correct semantic golds are not claimed.

Raw pretraining pools, packed arrays, optimizer state, credentials, SSH keys and
private host/network receipts are excluded. Published metadata replaces private
workspace paths with [workspace]; original and published hashes are separated.
54 actual local fixtures, raw predictions and held-out evidence are retained.
The canonical held-out 1,388 and schema-stress 56 requests are included as JSONL in
evaluation/native-json-heldout.jsonl and the prepared native data directory.
See THIRD_PARTY_NOTICES.md, docs/TECHNICAL_REPORT.md and provenance/LINEAGE.json.

The additional stage uses original SNIPS TRAIN replay plus authored cardinality
views, with frozen-parent training-only KL. Main inherits 2,048 presentations, not
the entire 3,072-presentation completed experiment. See the report for exact counts.
