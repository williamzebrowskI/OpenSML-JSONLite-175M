from pathlib import Path
from jsonlite_eval.common import ROOT,TOKENIZER,LOCK,INSTRUCTION,sha,read_rows,MODELS
from json_restart_130.common import BASE,BASE_SHA
from sml_v1.common import atomic_json
import json

DATA=ROOT/'pretraining/data/sft_142_fresh_foundation_20261018'
REPORT=ROOT/'reports/model-design/SFT-142_fresh_foundation_20261018'
RUN=ROOT/'pretraining/runs/sft_142_fresh_foundation_20261018'
PARENT=BASE
PARENT_SHA=BASE_SHA
PLAN=dict(seed=202610180142,batch_size=64,microbatch=2,epochs=1,presentations=36000,updates=563,
    peak_lr=3e-5,final_lr=3e-6,warmup=25,betas=[.9,.95],eps=1e-8,clip=1.,weight_decay=0.,
    context=2048,max_tokens=384,train_max_sequence=1536,eval_every=100,eval_steps=[100,200,300,400,500,563],
    adaptation='full weights',precision='BF16 compute / FP32 master AdamW',
    objective='Per-request assistant JSON and EOS loss; source/schema tokens masked.',
    retention='Keep every checkpoint. Compare against130/500 and140/563; no automatic promotion.')
QUOTAS=dict(snips=15000,schema_variation=3000,grounded_qa=3000,massive=3000,prior_authored=2400,atomic=3200,composite=6400)


def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def normalized(s):return ' '.join(s.casefold().split())
def write_rows(p,rows):Path(p).write_text(''.join(dumps(r)+'\n' for r in rows))

def protected():
    protected_models=[(BASE,BASE_SHA)]+[(ROOT/p,h) for p,h in MODELS.values()]
    protected_models += [(ROOT/'pretraining/runs/sft_141_learnability_20261018'/folder/'model.safetensors',h) for folder,h in [
        ('step_0000096_e05ef05f4d85','2fdd54db62147a8a5cc5a93fd9acf7f3ff9e2268bfd86c2615638eb469ee9913'),
        ('step_0000128_a5ec5258d137','fd9f45c29f2a7a2859672a19ecfa68f0a7c49714539eb1e25e9570dab7e42551')]]
    for p,h in protected_models:assert sha(p)==h,'Protected weight changed: '+str(p)
