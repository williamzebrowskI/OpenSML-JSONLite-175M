import hashlib,json,math,os
from pathlib import Path
from sml_v1.common import atomic_json,file_sha256 as sha

ROOT=Path(os.environ.get('CPT151_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
DATA=ROOT/'pretraining/data/structured_cpt_151_20261018'
REPORT=ROOT/'reports/pretraining/CPT-151_structured_20261018'
RUN=ROOT/'pretraining/runs/cpt_151_structured_20261018'
BASE=ROOT/'pretraining/runs/research_30b_176m_stage_a_continue_20261009/step_0132005_63f79113c3d2/model.safetensors'
BASE_SHA='5151567e266244c9e32fd6e01775192633abe227320269ecdefe28ce43e9ccc3'
BASE_OPT=Path(str(BASE)+'.optimizer.safetensors')
BASE_OPT_SHA='af3f9d2b5ac4275effd44b892dbadd3393a066931468fb157006b8f72e216f2d'
TOKENIZER=ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
LOCK=ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'
CORPUS=ROOT/'configs/corpus_20b.json'
OLD_WEIGHTS=dict(web=50,dclm=18,wiki=13,cosmopedia=9,python=10)
NEW_WEIGHTS=dict(json_structures=10,schema_instances=30,prose_schema_json=40,json_code=20)
MIX=dict(OLD_WEIGHTS,**NEW_WEIGHTS) # denominator200; each old source is halved.
PLAN=dict(format='cpt151-open-ended-v1',seed=202610180151,additional_tokens=None,context=2048,
    batches=[7,3,2,2,2],grad_accum=4,cache_gib_by_mac=[16,16,4,4,4],
    validation_rows_per_source=16,save_every_tokens=25000000,eval_every_tokens=25000000,log_every=10,
    warmup_tokens=5000000,start_lr=1e-5,peak_lr=3.221225472000002e-5,
    betas=[.9,.95],eps=1e-8,weight_decay=.1,clip=1.,
    objective='Token-mean next-token CE over every real document/EOS token; padding only is masked.',
    new_documents='Whole structured documents fit one2048-token window; multiple documents may share a row. Long original-corpus documents may span windows.',
    precision='BF16 compute; imported FP32 master weights, first/second moments and optimizer step.',
    weights=MIX,
    optimizer_origin='Import exact132005 optimizer and original-source consumed cursors. New structured cursors and explicit warmup/hold LR; not an exact historical-run resume.',
    retention='Keep all saved checkpoints, original base and142/145/150. No automatic promotion, reserved predictions or public changes.',
    scope='Open-ended five-Mac continuation. No50M/1B token stop, no timed stop. User requests a clean save with Ctrl+C. Source exhaustion/errors stop rather than silently recycle.')
KEPT={
 '142/563':('sft_142_fresh_foundation_20261018/step_0000563_aa4933fe911c','e53ee7d9a74ad74cda62d0344d7099c990f7e162700bb649936a6eae41eaecb7'),
 '145/500':('sft_145_paired_foundation_20261018/step_0000500_53c799fd0001','20f3398475b28dfe47e5402ef2cf07a9a772b0c8fdc0d85f452ad453d978998c'),
 '150/563':('sft_150_gated_distillation_20261018/step_0000563_d91dd3df1e20','291e481276b019ff9946bb3ba46323d1489141ff97b836cac8305e53822bf135')}

def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def load(p):return json.loads(Path(p).read_text())
def text_key(t):return hashlib.sha256(' '.join(t.casefold().split()).encode()).hexdigest()
def protected():
    assert sha(BASE)==BASE_SHA and sha(BASE_OPT)==BASE_OPT_SHA,'Pretrained origin changed'
    for label,(folder,h) in KEPT.items():assert sha(ROOT/'pretraining/runs'/folder/'model.safetensors')==h,'Kept model changed: '+label

def lr(tokens):
    if tokens<=PLAN['warmup_tokens']:
        return PLAN['start_lr']+(PLAN['peak_lr']-PLAN['start_lr'])*tokens/PLAN['warmup_tokens']
    return PLAN['peak_lr']
