"""Fresh pretrained foundation with cached130/140 compatibility comparisons."""
import argparse
import builtins
import importlib.metadata
import json
import math
import platform
from pathlib import Path
from unittest.mock import patch
from .common import *
from . import runtime as R
from . import scoring as S
from sml_v1.common import lock


def engine_print(*values,**kwargs):
    builtins.print(*[v.replace('joint130','SFT142').replace('fresh foundation','fresh pretrained132005') if isinstance(v,str) else v for v in values],**kwargs)


def contract():
    m=json.loads((DATA/'manifest.json').read_text());assert m['plan']==PLAN and m['base_sha256']==BASE_SHA
    for name,h in m['files'].items():assert sha(DATA/name)==h,'Changed dataset: '+name
    for name,h in m['inputs'].items():assert sha(ROOT/name)==h,'Changed source: '+name
    paths=list(Path(__file__).parent.glob('*.py'))+list(TOKENIZER.glob('*'))
    paths += [DATA/'manifest.json',ROOT/'pretraining/scripts/run_sft_142_fresh_foundation.sh',ROOT/'pretraining/tests/test_json_foundation_142.py',Path(str(PARENT)+'.json')]
    paths += [ROOT/'pretraining'/p for p in ['json_restart_130/train.py','json_restart_130/runtime.py','json_restart_130/common.py','json_foundation_142/scoring.py','json_learnability_141/data.py','jsonlite_eval/common.py','sft/data.py','sml_v1/common.py','sml_v1/model.py','sml_v1/precision.py','sml_v1/checkpoint_bundle.py','sml_v1/tokenization.py','sml_v1/fast_loss.py','sml_v1/fast_swiglu.py']]
    assert all(p.exists() for p in paths)
    return dict(plan=PLAN,base_sha256=BASE_SHA,files={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ['mlx','numpy','tokenizers','jsonschema']})


def aggregate(rows,outs):
    metrics=S.aggregate(rows,outs)
    # Keep source/domain/cardinality and TRAIN-fit separate from unseen transfer.
    for tag in sorted({t for r in rows for t in r.get('tags',[]) if t.startswith(('domain_','cardinality_','nonempty_'))}):
        ids=[i for i,r in enumerate(rows) if tag in r.get('tags',[])]
        metrics['tag/'+tag]=S.aggregate([rows[i] for i in ids],[outs[i] for i in ids])['all']
    return metrics


def evaluate(model,rows,path,h,step):
    import mlx.core as mx
    path=Path(path);enc=R.Encoding();model.eval()
    identity=dict(model_sha256=h,rows_sha256=__import__('hashlib').sha256(dumps(rows).encode()).hexdigest(),runner_sha256=sha(Path(__file__)),scorer_sha256=sha(Path(S.__file__)),max_output_tokens=PLAN['max_tokens'],runtime_sha256=sha(Path(R.__file__)),shared_decoder_sha256=sha(ROOT/'pretraining/json_restart_130/runtime.py'),mlx=importlib.metadata.version('mlx'),temperature=0,repetition_penalty=1,raw_greedy=True,repair=False,assembly=False,reference_in_input=False)
    prior=json.loads(path.read_text()) if path.exists() else None
    if prior:assert prior['identity']==identity
    outs=prior['outputs'] if prior else [];assert len(outs)<=len(rows)
    for r,o in zip(rows,outs):assert r['id']==o['id'] and o['checks']==S.score(r['document'],r['schema'],r['expected'],o['text'],o['stop_reason'])
    try:
        for r in rows[len(outs):]:
            raw=R.generate_one(model,enc,r['document'],r['schema'],r['instruction'])
            # Gold enters scoring/teacher-forced NLL only AFTER free generation.
            x,y=R.batch([enc.row(r)],[0]);nll=R.loss(model,mx.array(x),mx.array(y)).item();assert math.isfinite(nll)
            outs.append(dict(id=r['id'],**raw,output_nll=nll,checks=S.score(r['document'],r['schema'],r['expected'],raw['text'],raw['stop_reason'])))
            if len(outs)%32==0 or len(outs)==len(rows):
                atomic_json(path,dict(status='evaluating',step=step,identity=identity,outputs=outs));print(f'[SFT142 eval {step}] {path.stem} {len(outs)}/{len(rows)}',flush=True)
        metrics=aggregate(rows,outs);metrics['all']['output_nll']=sum(o['output_nll'] for o in outs)/len(outs)
        result=dict(status='complete',identity=identity,step=step,outputs=outs,metrics=metrics)
        atomic_json(path,result)
        print('[SFT142 metrics] '+dumps(dict(step=step,panel=path.stem,metrics=metrics)),flush=True)
        return result
    finally:model.train()


def evaluate_baselines():
    import mlx.core as mx
    rows=read_rows(DATA/'development.jsonl')
    for name in ['130','140']:
        path=REPORT/('baseline-'+name+'.json')
        model=None
        try:
            if path.exists() and json.loads(path.read_text())['status']=='complete':
                cached=json.loads(path.read_text());assert cached['identity']['model_sha256']==MODELS[name][1]
                assert cached['identity']['runner_sha256']==sha(Path(__file__)) and cached['identity']['scorer_sha256']==sha(Path(S.__file__))
                assert cached['identity']['runtime_sha256']==sha(Path(R.__file__)) and cached['identity']['shared_decoder_sha256']==sha(ROOT/'pretraining/json_restart_130/runtime.py')
                assert cached['identity']['max_output_tokens']==PLAN['max_tokens']
                assert cached['identity']['rows_sha256']==__import__('hashlib').sha256(dumps(rows).encode()).hexdigest()
                for r,o in zip(rows,cached['outputs']):assert r['id']==o['id'] and o['checks']==S.score(r['document'],r['schema'],r['expected'],o['text'],o['stop_reason'])
                assert len(cached['outputs'])==len(rows)
                computed=aggregate(rows,cached['outputs']);stored=json.loads(json.dumps(cached['metrics']));stored['all'].pop('output_nll',None);assert computed==stored
                continue
            model=R.model_at(ROOT/MODELS[name][0]);evaluate(model,rows,path,MODELS[name][1],name)
        finally:
            del model;mx.clear_cache()


def selection(panels):
    m=panels['development'];means={}
    for label in ['atomic','composite']:
        rates=[v['complete_request_accuracy'] for k,v in m.items() if k.startswith('family/'+label+'/')]
        assert rates;means[label]=sum(rates)/len(rates)
    legacy=m['tier/compatibility']['complete_request_accuracy']
    return [.5*legacy+.25*means['atomic']+.25*means['composite'],legacy,m['all']['structural_path_f1'] or 0]


def comparison(before,after,rows):
    ids={r['id'] for r in rows if r['component']=='compatibility'}
    a={o['id']:o for o in before['outputs']};b={o['id']:o for o in after['outputs']}
    lost=sorted(i for i in ids if a[i]['checks']['exact'] and not b[i]['checks']['exact'])
    fixed=sorted(i for i in ids if not a[i]['checks']['exact'] and b[i]['checks']['exact'])
    x=before['metrics']['tier/compatibility'];y=after['metrics']['tier/compatibility']
    keys=['complete_request_accuracy','top_level_field_accuracy','structural_path_f1','schema_valid_rate','natural_stop_rate','missing_value_accuracy']
    delta={k:y[k]-x[k] for k in keys if x[k] is not None and y[k] is not None}
    return dict(passed=not lost and all(v>=-1e-12 for v in delta.values()),fixed=fixed,lost=lost,metric_deltas=delta,
        scope='Observed64 SNIPS/MASSIVE compatibility subset only. Fresh foundation does not inherit these SFT weights; this comparison is not a universal retention guarantee.')


def assess(model,step,bundle):
    h=sha(bundle/'model.safetensors');panels={};dev_result=None
    for panel in ['fit_probe','development']:
        rows=read_rows(DATA/(panel+'.jsonl'));result=evaluate(model,rows,RUN/'evaluations'/f'{panel}_{step:07d}.json',h,step);panels[panel]=result['metrics']
        if panel=='development':dev_result=result
    rows=read_rows(DATA/'development.jsonl')
    record=dict(step=step,model_sha256=h,bundle=bundle.name,panels=panels,automatic_promotion=False,
        comparisons={n:comparison(json.loads((REPORT/('baseline-'+n+'.json')).read_text()),dev_result,rows) for n in ['130','140']})
    key=selection(panels);record['selection']=key
    for name,eligible in [('best',True),('best-compatible',all(v['passed'] for v in record['comparisons'].values()))]:
        p=RUN/(name+'-metrics.json');old=json.loads(p.read_text()) if p.exists() else None
        if eligible and (old is None or key>old['selection']):
            atomic_json(RUN/(name+'.json'),dict(format='sml-pretrain-bundle-v1',bundle=bundle.name));atomic_json(p,record)
    atomic_json(RUN/'evaluations'/f'summary_{step:07d}.json',record);atomic_json(RUN/'last_eval.json',record)
    print('[SFT142 summary] '+dumps(record),flush=True)


def interpretation():
    records=[json.loads((RUN/'evaluations'/f'summary_{s:07d}.json').read_text()) for s in PLAN['eval_steps']]
    lines=['# SFT-142 foundation results','','Fresh pretrained132005 -> one full-weight joint SFT. No earlier SFT weights inherited.','',
           '| Update | TRAIN fit /64 | Development /448 | Compatibility /64 | Lost vs130 | Lost vs140 |','|---|---:|---:|---:|---:|---:|']
    for r in records:
        m=r['panels']['development'];lines.append(f"| {r['step']} | {r['panels']['fit_probe']['all']['counts']['exact']} | {m['all']['counts']['exact']} | {m['tier/compatibility']['counts']['exact']} | {len(r['comparisons']['130']['lost'])} | {len(r['comparisons']['140']['lost'])} |")
    lines+=['','This development set uses controlled, related fact grammar. Read per-family and compatibility regressions before any release decision. Reserved and larger external suites are not generated automatically.']
    (REPORT/'results-summary.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--reserved-eval',action='store_true');a=p.parse_args()
    c=contract()
    if not(a.run or a.reserved_eval):
        print(dumps(dict(training_started=False,parent='Stage-A pretrained132005 (no inherited SFT weights)',plan=PLAN,coverage=str(REPORT/'coverage.md'),reserved_evaluation=False)));return
    ready=json.loads((REPORT/'ready.json').read_text());assert ready['status']=='passed' and ready['contract']==c,'Readiness missing or stale'
    for path,h in ready['receipts'].items():assert sha(ROOT/path)==h
    assert importlib.metadata.version('mlx')=='0.32.3'
    from json_sampling_119.run import active_workers
    from json_restart_130 import train as ENGINE
    from sml_v1.checkpoint_bundle import resolve_bundle
    import mlx.core as mx
    with lock(LOCK):
        if active_workers():raise RuntimeError('Another training worker is active')
        protected();mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
        if a.reserved_eval:
            state=json.loads((RUN/'result.json').read_text());assert state['status']=='complete'
            weight=Path(resolve_bundle(RUN/'best.json'));model=R.model_at(weight)
            evaluate(model,read_rows(DATA/'reserved.jsonl'),RUN/'reserved-final.json',sha(weight),state['selected']['step']);protected();return
        state=json.loads((RUN/'result.json').read_text()) if (RUN/'result.json').exists() else {}
        if state.get('status')=='complete':print(dumps(state));return
        RUN.mkdir(parents=True,exist_ok=True);session=RUN/'session.json'
        if session.exists():assert json.loads(session.read_text())==c,'Changed experiment; refusing to mix results'
        else:atomic_json(session,c)
        for name in ['130','140']:
            baseline=json.loads((REPORT/('baseline-'+name+'.json')).read_text());assert baseline['status']=='complete'
        model=R.model_at(BASE)
        rows=read_rows(DATA/'train.jsonl');batches=json.loads((DATA/'batches.json').read_text())
        with patch.object(ENGINE,'Encoding',R.Encoding),patch.object(ENGINE,'print',engine_print,create=True):
            ENGINE.execute(model,rows,RUN,c,custom_plan=PLAN,custom_batches=batches,assessor=assess,resume=(RUN/'latest.json').exists())
        state=json.loads((RUN/'result.json').read_text())
        if state['status']=='sft_complete':
            best=json.loads((RUN/'best-metrics.json').read_text());state.update(status='complete',selected=best,automatic_promotion=False,reserved_predictions=0)
            atomic_json(RUN/'result.json',state);interpretation()
        protected();assert contract()==c
        print('[SFT142 state] '+dumps(state),flush=True)


if __name__=='__main__':main()
