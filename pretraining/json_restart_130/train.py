"""User-launched foundation, optional development-guided refinement; never auto-promote."""
import argparse,copy,json,math,signal,time
from dataclasses import asdict
from pathlib import Path
from importlib.metadata import version
import numpy as np
from .common import *
from .runtime import model_at,Encoding,order_for,gradients,lr
from .evaluate import evaluate
from .scoring import selection,comparison,original_score,original_aggregate
from sml_v1.common import lock,fingerprint


def contract():
    check();files=list(Path(__file__).parent.glob('*.py'))+list(TOK.glob('*'))
    files += [BASE.with_suffix('.safetensors.json'),RELEASE.with_suffix('.safetensors.json'),DATA/'manifest.json']
    files += [ROOT/'pretraining'/p for p in ['tests/test_json_restart_130.py','scripts/run_sft_130_joint_foundation.sh','scripts/run_sft_131_json_refinement.sh','sft/data.py','sft/evaluate.py','sft/quality.py','sml_v1/model.py','sml_v1/precision.py','sml_v1/checkpoint_bundle.py','sml_v1/common.py','sml_v1/tokenization.py','sml_v1/fast_loss.py','sml_v1/fast_swiglu.py','json_sampling_119/common.py','json_sampling_119/data.py','json_sampling_119/scoring.py','native_json_128/scoring.py']]
    if any(not p.exists() for p in files):raise ValueError('A required source/tokenizer file is missing')
    return dict(plan=PLAN,refinement=REFINEMENT,files={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()},
        versions={n:version(n) for n in ['mlx','numpy','tokenizers','jsonschema']},base_sha256=BASE_SHA,release_sha256=RELEASE_SHA)


def development_key(m):return selection(m)


def published_cache(panel):
    rows=load(PUBLISHED/(panel+'.jsonl'));before=json.loads((PUBLISHED/f'replay128_300-{panel}.json').read_text())
    if before['identity']['model_sha256']!=RELEASE_SHA:raise ValueError('Released cache identity mismatch')
    assert [r['id'] for r in rows]==[o['id'] for o in before['outputs']]
    for r,o in zip(rows,before['outputs']):assert original_score(r,o['text'],o['stop_reason'])==o['checks']
    metrics=original_aggregate(rows,before['outputs'],[o['output_nll'] for o in before['outputs']])
    assert all(v==before['metrics'][k] for k,v in metrics.items())
    return rows,before

def refine_rows(train,bank,metrics):
    rates={k.split('/',1)[1]:v['exact'] for k,v in metrics.items() if k.startswith('family/') and k!='family/snips'}
    weak=sorted(rates,key=lambda k:(rates[k],k))[:REFINEMENT['max_focus_families']]
    rng=np.random.default_rng(REFINEMENT['seed']);replay=rng.choice(len(train),size=3072,replace=False)
    focus=[]
    for j in range(1024):
        family=weak[j%len(weak)];eligible=[i for i,r in enumerate(bank) if r['family']==family and i not in focus]
        if not eligible:raise ValueError('Insufficient independent TRAIN focus examples')
        focus.append(int(rng.choice(eligible)))
    rows=[dict(copy.deepcopy(train[int(i)]),component='replay131') for i in replay]+[dict(copy.deepcopy(bank[i]),component='focus131') for i in focus]
    # Each update16focus? Require exactly24 replay+8 focus (75% replay).
    a=rng.permutation(3072);b=rng.permutation(1024)+3072
    batches=[np.concatenate([a[i*24:(i+1)*24],b[i*8:(i+1)*8]]).tolist() for i in range(128)]
    return rows,batches,weak,rates


def prepare_refinement(c):
    path=DATA/'refinement.json';rows_path=DATA/'refinement_train.jsonl'
    if path.exists():
        p=json.loads(path.read_text());assert p['contract']==c and sha(rows_path)==p['rows_sha256']
        assert sha(ROOT/p['parent'])==p['parent_sha256'];return p
    result=json.loads((RUN/'result.json').read_text())
    if result['status']!='complete':raise ValueError('Finish foundation training/comparison before optional refinement')
    from sml_v1.checkpoint_bundle import resolve_bundle
    parent=Path(resolve_bundle(RUN/'best.json'));meta=json.loads((RUN/'best-metrics.json').read_text())
    # This policy reads DEVELOPMENT only. Original test/playground/reserved outcomes do not choose examples.
    dev=json.loads((RUN/'evaluations'/f'development_{meta["step"]:07d}.json').read_text())
    rows,batches,weak,rates=refine_rows(load(DATA/'train.jsonl'),load(DATA/'refinement_bank.jsonl'),dev['metrics'])
    write_rows(rows_path,rows)
    p=dict(contract=c,parent=str(parent.relative_to(ROOT)),parent_sha256=sha(parent),rows_sha256=sha(rows_path),batches=batches,
        weak_families=weak,development_rates=rates,development_sha256=sha(RUN/'evaluations'/f'development_{meta["step"]:07d}.json'),
        baseline_path=str((RUN/'evaluations'/f'development_{meta["step"]:07d}.json').relative_to(ROOT)),
        selection_inputs='Foundation development metrics only; no test/reserved predictions, no generated training targets.')
    atomic_json(path,p);return p


def execute(model,rows,out,c,*,refine=False,resume=False,assessor=None,stop_after=None,custom_plan=None,custom_batches=None,configure_trainable=None,on_restore=None):
    import mlx.core as mx
    from mlx.utils import tree_map,tree_flatten,tree_unflatten
    from sml_v1.precision import MasterAdamW
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    enc=Encoding();encoded=[enc.row(r,training=True) for r in rows]
    p=custom_plan or (dict(PLAN,**REFINEMENT) if refine else PLAN)
    batches=custom_batches or order_for(encoded,p['seed'],p['batch_size']);total=len(batches);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    order_hash=fingerprint(batches);step=records=tokens=0
    model.unfreeze()
    if configure_trainable is not None:configure_trainable(model)
    model.train();opt=MasterAdamW(learning_rate=p['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=PLAN['weight_decay']);opt.init(model.trainable_parameters());mx.eval(opt.state)
    if resume:
        weight=resolve_bundle(out/'latest.json');meta=json.loads(Path(weight+'.json').read_text());state=meta['restart130']
        if state['contract']!=c or state['order_hash']!=order_hash or state['refine']!=refine:raise ValueError('Resume contract/order changed')
        model.load_weights(weight);opt.state=tree_unflatten(list(mx.load(weight+'.optimizer.safetensors').items()));step=meta['step'];records=state['records'];tokens=meta['tokens'];mx.eval(model.parameters(),opt.state)
    if int(opt.step.item())!=step or records!=sum(len(b) for b in batches[:step]):raise ValueError('Optimizer/cursor mismatch')
    if resume and on_restore is not None:on_restore(model,opt,step)
    steps=p.get('eval_steps') if refine else sorted(set(range(p['eval_every'],total+1,p['eval_every']))|{total})
    if custom_plan is not None:steps=p.get('eval_steps',[total])
    stop=[False];handlers={};saved_step=step if resume else -1
    for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def save():
        nonlocal saved_step
        meta=dict(step=step,tokens=tokens,recipe=dict(model=asdict(model.cfg)),restart130=dict(contract=c,order_hash=order_hash,records=records,total=total,refine=refine))
        bundle=Path(save_bundle(out,model,opt,meta,dict(next_batch=step),keep=10000,reserve_gib=5));saved_step=step;return bundle
    def assess(bundle):
        if assessor:return assessor(model,step,bundle)
        result=evaluate(model,load(DATA/'development.jsonl'),out/'evaluations'/f'development_{step:07d}.json',sha(bundle/'model.safetensors'),step)
        baseline=json.loads((REPORT/'released-development.json').read_text());result['vs_release']=comparison(load(DATA/'development.jsonl'),baseline,result)
        if refine:
            info=json.loads((DATA/'refinement.json').read_text());parent=json.loads((ROOT/info['baseline_path']).read_text());result['vs_foundation']=comparison(load(DATA/'development.jsonl'),parent,result)
        atomic_json(out/'evaluations'/f'development_{step:07d}.json',result)
        key=development_key(result['metrics']);prior=json.loads((out/'best-metrics.json').read_text()) if (out/'best-metrics.json').exists() else None
        if prior is None or key>prior['selection']:
            atomic_json(out/'best.json',dict(format='sml-pretrain-bundle-v1',bundle=bundle.name));atomic_json(out/'best-metrics.json',dict(step=step,selection=key,bundle=bundle.name,scope='Best DEVELOPMENT research candidate; not a release qualification.'))
        atomic_json(out/'last_eval.json',{k:v for k,v in result.items() if k!='outputs'})
        print('[joint130 retention] '+dumps(dict(step=step,vs_release=result['vs_release'],vs_foundation=result.get('vs_foundation'),automatic_promotion=False)),flush=True)
    try:
        if resume and step in steps:assess(Path(resolve_bundle(out/'latest.json')).parent)
        start=time.monotonic();print(f'[ready-joint130] {"refinement" if refine else "fresh foundation"} {step}/{total}; batch={p["batch_size"]}; micro=2; eval={steps}; output-only; all checkpoints retained',flush=True)
        while step<total:
            if stop[0] or (out/'STOP').exists() or (stop_after is not None and step>=stop_after):
                if step!=saved_step:save()
                atomic_json(out/'result.json',dict(status='stopped',step=step,total=total,records=records,assistant_tokens=tokens));return
            ids=batches[step];v,g,n=gradients(model,encoded,ids,p['microbatch']);norm=mx.sqrt(sum(mx.sum(a*a) for _,a in tree_flatten(g)));mx.eval(norm)
            if not math.isfinite(v) or not math.isfinite(norm.item()):raise ValueError('Nonfinite training loss/gradient')
            g=tree_map(lambda a:a*mx.minimum(1.,PLAN['clip']/(norm+1e-6)),g);opt.learning_rate=lr(step+1,total,p);opt.update(model,g);mx.eval(model.parameters(),opt.state)
            step+=1;records+=len(ids);tokens+=n
            atomic_json(out/'progress.json',dict(step=step,total=total,records=records,assistant_tokens=tokens,loss=v,gradient_norm=norm.item(),learning_rate=lr(step,total,p),elapsed_seconds=time.monotonic()-start))
            if step==1 or step%10==0:print(f'[joint130 {step}/{total}] output_loss={v:.5f} lr={lr(step,total,p):.3e} grad_norm={norm.item():.3f} records={records} elapsed={time.monotonic()-start:.1f}s',flush=True)
            if step in steps:assess(save())
        atomic_json(out/'result.json',dict(status='sft_complete',step=step,total=total,records=records,assistant_tokens=tokens,automatic_promotion=False))
    finally:
        for sig,h in handlers.items():signal.signal(sig,h)


def finish(out,c):
    import mlx.core as mx
    from sml_v1.checkpoint_bundle import resolve_bundle
    out=Path(out);best=json.loads((out/'best-metrics.json').read_text());weight=Path(resolve_bundle(out/'best.json'));model=model_at(weight);comparisons={}
    # Exact public protocol/scorer and frozen parent outputs. Does not change selection.
    for panel in ['original1444','playground60']:
        rows,before=published_cache(panel)
        candidate=evaluate(model,rows,out/(panel+'.json'),sha(weight),best['step'],original=True)
        a={o['id']:o['checks']['exact'] for o in before['outputs']};b={o['id']:o['checks']['exact'] for o in candidate['outputs']}
        comparisons[panel]=dict(parent_metrics=before['metrics'],candidate_metrics=candidate['metrics'],fixed=[i for i in a if not a[i] and b[i]],lost=[i for i in a if a[i] and not b[i]])
    del model;mx.clear_cache()
    state=json.loads((out/'result.json').read_text());state.update(status='complete',selected=best,selected_sha256=sha(weight),comparison=compare_summary(comparisons),reserved_predictions=0,automatic_promotion=False,
        next='Optional refinement uses development only. Reserved evaluation is a separate --final-eval after deciding the last stage.')
    atomic_json(out/'comparison.json',dict(contract=c,selected=best,panels=comparisons,automatic_promotion=False))
    atomic_json(out/'result.json',state);print('[joint130 complete] '+dumps(state),flush=True)


def compare_summary(comp):
    return {n:dict(parent_correct=v['parent_metrics']['all']['counts']['exact'],candidate_correct=v['candidate_metrics']['all']['counts']['exact'],fixed=len(v['fixed']),lost=len(v['lost'])) for n,v in comp.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--refine',action='store_true');p.add_argument('--final-eval',action='store_true');a=p.parse_args();c=contract()
    if not(a.run or a.final_eval):print(dumps(dict(training_started=False,base=str(BASE),foundation=PLAN,refinement=REFINEMENT,stats=check()['stats'])));return
    ready=json.loads((REPORT/'ready.json').read_text())
    if ready['status']!='passed' or ready['contract']!=c:raise ValueError('Preflight missing/stale')
    for n,h in ready['receipts'].items():
        if sha(ROOT/n)!=h:raise ValueError('Readiness receipt changed')
    with lock(LOCK):
        from json_sampling_119.run import active_workers
        if active_workers():raise RuntimeError('Another training worker is active')
        import mlx.core as mx
        mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
        out=REFINE if a.refine else RUN
        if a.final_eval:
            from sml_v1.checkpoint_bundle import resolve_bundle
            state=json.loads((out/'result.json').read_text())
            if state['status']!='complete':raise ValueError('Finish selected stage first')
            previously_observed=any((p/'reserved-final.json').exists() for p in [RUN,REFINE])
            weight=Path(resolve_bundle(out/'best.json'));m=model_at(weight)
            result=evaluate(m,load(DATA/'reserved.jsonl'),out/'reserved-final.json',sha(weight))
            state.update(reserved_predictions=len(result['outputs']),reserved_sha256=sha(out/'reserved-final.json'),reserved_previously_observed=previously_observed)
            atomic_json(out/'result.json',state);return
        info=prepare_refinement(c) if a.refine else None
        run_contract=dict(contract=c,refinement=info) if a.refine else c
        out.mkdir(parents=True,exist_ok=True);session=out/'session.json'
        if session.exists() and json.loads(session.read_text())!=run_contract:raise ValueError('Session changed')
        atomic_json(session,run_contract)
        has=(out/'latest.json').exists()
        if (out/'result.json').exists() and json.loads((out/'result.json').read_text())['status']=='complete':print((out/'result.json').read_text());return
        # --run resumes an intact bundle automatically; --resume is accepted for explicit use.
        rows=load(DATA/('refinement_train.jsonl' if a.refine else 'train.jsonl'))
        model=model_at(ROOT/info['parent'] if a.refine else BASE)
        if (out/'result.json').exists() and json.loads((out/'result.json').read_text())['status']=='sft_complete':del model;mx.clear_cache();finish(out,run_contract);return
        execute(model,rows,out,run_contract,refine=a.refine,resume=has,custom_batches=info['batches'] if info else None)
        state=json.loads((out/'result.json').read_text());del model;mx.clear_cache()
        if state['status']=='sft_complete':finish(out,run_contract)
        if sha(BASE)!=BASE_SHA or sha(RELEASE)!=RELEASE_SHA:raise ValueError('Protected weights changed')
if __name__=='__main__':main()
