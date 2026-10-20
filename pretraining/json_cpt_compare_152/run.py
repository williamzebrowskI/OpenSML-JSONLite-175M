"""Same SFT recipe and observed DEV as142; parent weights are the only treatment."""
import argparse,builtins,hashlib,math
from pathlib import Path
from unittest.mock import patch
from .common import *
from json_foundation_142 import runtime as R,scoring as S
from json_foundation_142.run import aggregate,selection
from sml_v1.common import lock

def engine_print(*values,**kwargs):
    builtins.print(*[v.replace('joint130','SFT152-CPT').replace('fresh foundation','matched CPT foundation') if isinstance(v,str) else v for v in values],**kwargs)

def evaluate(model,requests,path,h,step):
    import mlx.core as mx
    path=Path(path);model.eval();enc=R.Encoding()
    identity=dict(model_sha256=h,rows_sha256=hashlib.sha256(dumps(requests).encode()).hexdigest(),
        runtime_sha256=sha(Path(R.__file__)),scorer_sha256=sha(Path(S.__file__)),runner_sha256=sha(Path(__file__)),
        max_output_tokens=PLAN['max_tokens'],raw_greedy=True,repair=False,assembly=False,reference_in_input=False)
    prior=load(path) if path.exists() else None
    if prior:assert prior['identity']==identity
    outputs=prior['outputs'] if prior else []
    for r,o in zip(requests,outputs):assert r['id']==o['id'] and o['checks']==S.score(r['document'],r['schema'],r['expected'],o['text'],o['stop_reason'])
    try:
        for r in requests[len(outputs):]:
            raw=R.generate_one(model,enc,r['document'],r['schema'],r['instruction'])
            # Gold is used only after independent generation, for scoring/NLL.
            x,y=R.batch([enc.row(r)],[0]);nll=float(R.loss(model,mx.array(x),mx.array(y)).item());assert math.isfinite(nll)
            outputs.append(dict(id=r['id'],**raw,output_nll=nll,checks=S.score(r['document'],r['schema'],r['expected'],raw['text'],raw['stop_reason'])))
            if len(outputs)%32==0 or len(outputs)==len(requests):
                atomic_json(path,dict(status='evaluating',identity=identity,step=step,outputs=outputs))
                print(f'[SFT152-CPT eval {step}] {path.stem} {len(outputs)}/{len(requests)}',flush=True)
        metrics=aggregate(requests,outputs);metrics['all']['output_nll']=sum(o['output_nll'] for o in outputs)/len(outputs)
        result=dict(status='complete',identity=identity,step=step,outputs=outputs,metrics=metrics)
        atomic_json(path,result);return result
    finally:model.train()

def compare(before,after):
    a={o['id']:o['checks']['exact'] for o in before['outputs']};b={o['id']:o['checks']['exact'] for o in after['outputs']}
    assert set(a)==set(b)
    fixed=sorted(i for i in a if not a[i] and b[i]);lost=sorted(i for i in a if a[i] and not b[i])
    keys=['complete_request_accuracy','top_level_field_accuracy','structural_path_f1','schema_valid_rate','missing_value_accuracy','optional_omission_accuracy']
    delta={k:after['metrics']['all'][k]-before['metrics']['all'][k] for k in keys if before['metrics']['all'][k]is not None and after['metrics']['all'][k]is not None}
    compat={r['id'] for r in rows(DATA/'development.jsonl') if r['component']=='compatibility'}
    return dict(fixed=fixed,lost=lost,compatibility_lost=sorted(compat&set(lost)),metric_deltas=delta,
        no_loss_on_observed_requests=not lost,scope='Repeatedly observed448-case development; not a universal retention guarantee.')

def assess(arm,model,step,bundle):
    out=ARMS[arm]['run'];h=sha(bundle/'model.safetensors');panels={};dev=None
    for panel in ['fit_probe','development']:
        result=evaluate(model,rows(DATA/(panel+'.jsonl')),out/'evaluations'/f'{panel}_{step:07d}.json',h,step)
        panels[panel]=result['metrics']
        if panel=='development':dev=result
    control=load(REPORT/'control'/f'development_{step:07d}.json')
    endpoint=load(REPORT/'control'/'development_0000563.json')
    key=selection(panels)
    record=dict(arm=arm,step=step,bundle=bundle.name,model_sha256=h,selection=key,panels=panels,
        versus_same_update=compare(control,dev),versus_control563=compare(endpoint,dev),automatic_promotion=False)
    eligible=not record['versus_same_update']['lost'] and all(v>=-1e-12 for v in record['versus_same_update']['metric_deltas'].values())
    for name,allowed in [('best-research',True),('best-no-loss',eligible)]:
        old=load(out/(name+'-metrics.json')) if (out/(name+'-metrics.json')).exists() else None
        if allowed and (old is None or key>old['selection']):
            atomic_json(out/(name+'.json'),dict(format='sml-pretrain-bundle-v1',bundle=bundle.name));atomic_json(out/(name+'-metrics.json'),record)
    atomic_json(out/'evaluations'/f'summary_{step:07d}.json',record);atomic_json(out/'last_eval.json',record)
    nested={k:v['counts']['exact'] for k,v in panels['development'].items() if k.startswith('family/') and any(n in k for n in ['nested','record_allocations','mixed_inventory','local_ref','parallel_teams'])}
    print('[SFT152-CPT summary] '+dumps(dict(arm=arm,step=step,correct=panels['development']['all']['counts']['exact'],
        total=448,control_same_update_correct=control['metrics']['all']['counts']['exact'],
        fixed=len(record['versus_same_update']['fixed']),lost=len(record['versus_same_update']['lost']),
        nested_related_dev_correct=nested,selection=key)),flush=True)

def run_arm(arm):
    import importlib.metadata
    import mlx.core as mx
    from json_restart_130 import train as ENGINE
    from json_sampling_119.run import active_workers
    c=contract(arm);ready=load(REPORT/'ready.json')
    assert ready['status']=='passed' and ready['contracts'][arm]==c,'Missing or stale readiness'
    assert importlib.metadata.version('mlx')=='0.32.3'
    for p,h in ready['receipts'].items():assert sha(ROOT/p)==h,'Changed readiness evidence'
    assert not active_workers(),'Another model worker is active'
    protected();out=ARMS[arm]['run'];state=load(out/'result.json') if (out/'result.json').exists() else {}
    if state.get('status')=='complete':print('[already complete] '+arm,flush=True);return True
    out.mkdir(parents=True,exist_ok=True)
    session=out/'session.json'
    if session.exists():assert load(session)==c,'Changed experiment; do not mix progress'
    else:atomic_json(session,c)
    if (out/'STOP').exists():raise ValueError('Remove only this SFT run\'s STOP marker before intentionally resuming it')
    mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
    model=R.model_at(ARMS[arm]['parent'])
    try:
        for k in ['betas','eps','clip','weight_decay']:assert ENGINE.PLAN[k]==PLAN[k],k
        with patch.object(ENGINE,'Encoding',R.Encoding),patch.object(ENGINE,'print',engine_print,create=True):
            ENGINE.execute(model,rows(DATA/'train.jsonl'),out,c,custom_plan=PLAN,custom_batches=load(DATA/'batches.json'),
                assessor=lambda m,s,b:assess(arm,m,s,b),resume=(out/'latest.json').exists())
        state=load(out/'result.json')
        if state['status']=='sft_complete':
            pick=load(out/'best-research-metrics.json')
            atomic_json(out/'best.json',dict(format='sml-pretrain-bundle-v1',bundle=pick['bundle']))
            atomic_json(out/'best-metrics.json',pick)
            state.update(status='complete',training_complete=True,broader_evaluation_complete=False,arm=arm,selected=pick,
                         inherited_sft_weights=False,fresh_sft_optimizer=True,reserved_predictions=0,automatic_promotion=False)
            atomic_json(out/'result.json',state)
        protected();assert contract(arm)==c
        return state['status']=='complete'
    finally:del model;mx.clear_cache()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--arm',choices=['25m','100m','all'],default='all');a=p.parse_args()
    print(dumps(dict(training_started=False,arms=a.arm,recipe='Exact142 recipe/data/order',control='Verified saved142 evaluations; no redundant base retraining',
        plan=PLAN,optimizer='Fresh AdamW per arm; no CPT optimizer or earlier SFT weights inherited',
        selection='Original142 DEVELOPMENT rule; report fixed563 endpoint separately; no benchmark-guided selection')),flush=True)
    if not a.run:return
    with lock(LOCK):
        for arm in (['25m','100m'] if a.arm=='all' else [a.arm]):
            if not run_arm(arm):break # Ctrl+C must not start the next arm.

if __name__=='__main__':main()
