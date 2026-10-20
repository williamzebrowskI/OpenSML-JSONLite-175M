from pathlib import Path
from jsonlite_eval.common import ROOT, TOKENIZER, LOCK, INSTRUCTION, sha, read_rows, content
from json_restart_130.common import BASE, BASE_SHA
from jsonlite_eval.common import MODELS
from sml_v1.common import atomic_json

def dumps(value):
    return __import__('json').dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


DATA = ROOT/'pretraining/data/sft_141_learnability_20261018'
REPORT = ROOT/'reports/model-design/SFT-141_learnability_20261018'
RUN = ROOT/'pretraining/runs/sft_141_learnability_20261018'
PARENT = ROOT/MODELS['140'][0]
PARENT_SHA = MODELS['140'][1]
PLAN = dict(seed=202610180141, batch_size=32, microbatch=2, epochs=2,
    presentations=4096, unique_training_rows=2048, updates=128,
    peak_lr=1e-5, final_lr=2e-6, warmup=8, betas=[.9,.95], eps=1e-8,
    clip=1., weight_decay=0., context=2048, max_tokens=256,
    train_max_sequence=1536, eval_every=32, eval_steps=[32,64,96,128],
    adaptation='full weights', precision='BF16 compute / FP32 master AdamW',
    objective='Per-request assistant JSON and EOS loss; source/schema masked.',
    retention='All checkpoints retained; no automatic promotion. Strict retention pointer separately labelled.')


def protected():
    for p,h in [(BASE,BASE_SHA)]+[(ROOT/p,h) for p,h in MODELS.values()]:
        assert sha(p)==h, 'Protected checkpoint changed: '+str(p)


def write_rows(path, rows):
    Path(path).write_text(''.join(dumps(r)+'\n' for r in rows))


def normalized(s):
    return ' '.join(s.casefold().split())
