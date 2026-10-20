"""CPU data checks, five-Mac idle receipt, real-model SFT optimizer/resume smoke."""
import datetime,inspect,json,math,sys,tempfile
from pathlib import Path
from unittest.mock import patch
from .common import *
from . import run as T
from json_foundation_142 import runtime as R,scoring as S
from sml_v1.common import lock

def workers_stopped():
    sys.path.insert(0,str(ROOT/'pretraining/scripts'))
    from five_mac_cluster import Cluster,PYTHON
    cluster=Cluster(LOCK.parent);reports=[]
    for physical in range(5):
        code="import subprocess,json; rows=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines(); hits=[r.strip() for r in rows if ' -c ' not in r and any(s in r for s in ['-m structured_cpt_151.worker','-m sml_v1.pretrain'])]; print(json.dumps(dict(mac="+str(physical+1)+",active_pretraining_workers=hits)))"
        report=json.loads(cluster.remote(physical,[PYTHON,'-c',code]));assert not report['active_pretraining_workers'];reports.append(report)
    progress=load(CPT/'progress.json');assert progress['status']=='stopped'
    jobs=sorted((CPT/'jobs').glob('*'));result=load(jobs[-1]/'result.json');assert result['status']=='stopped'
    from sml_v1.checkpoint_bundle import resolve_bundle
    latest=Path(resolve_bundle(CPT/'latest.json'));metadata=load(Path(str(latest)+'.json'))
    assert metadata['cpt151']['added_tokens']==result['added_tokens']==progress['added_tokens']
    receipt=dict(status='passed',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),macs=reports,
        training_lock_released=True,final_checkpoint=str(latest.relative_to(ROOT)),final_weight_sha256=sha(latest),
        final_added_tokens=result['added_tokens'],job_result=result)
    atomic_json(REPORT/'pretraining-stop-verification.json',receipt)

def nll(model,requests):
    import mlx.core as mx
    enc=R.Encoding();encoded=[enc.row(r,training=True) for r in requests];values=[]
    for i in range(0,len(requests),2):
        ids=list(range(i,min(i+2,len(requests))));x,y=R.batch(encoded,ids)
        v=float(R.loss(model,mx.array(x),mx.array(y)).item());assert math.isfinite(v);values.append((v,len(ids)))
    return sum(v*n for v,n in values)/sum(n for _,n in values)

def checks():
    requests=rows(DATA/'development.jsonl');r=next(r for r in requests if r['family']=='composite/nested_delivery')
    enc=R.Encoding();prefix=enc.prefix(r['document'],r['schema'],r['instruction'])
    assert 'expected' not in inspect.signature(R.generate_one).parameters
    x1,_=enc.row(r);x2,_=enc.row(dict(r,expected={'secret':'PRIVATE_TARGET_MARKER'}))
    assert x1[:len(prefix)].tolist()==x2[:len(prefix)].tolist()
    assert 'PRIVATE_TARGET_MARKER' not in enc.tokenizer.decode(prefix)
    schema=dict(type='object',properties=dict(flag=dict(type='boolean')),required=['flag'],additionalProperties=False)
    assert S.score('false',schema,dict(flag=False),'{"flag":false}','turn_end')['exact']
    for text,stop in [('{"flag":0}','turn_end'),('{"flag":false,"flag":true}','turn_end'),('{"flag":false}','token_limit'),('```json\n{"flag":false}\n```','turn_end')]:
        assert not S.score('false',schema,dict(flag=False),text,stop)['exact']
    r=next(r for r in rows(DATA/'train.jsonl') if r['family']=='composite/record_allocations' and len(next(iter(r['expected'].values())))>=2)
    value=dumps(r['expected']);wrong=json.loads(value);k=next(iter(wrong));wrong[k][0],wrong[k][1]=wrong[k][1],wrong[k][0]
    assert not S.score(r['document'],r['schema'],r['expected'],dumps(wrong),'turn_end')['exact']
    assert S.score(r['document'],r['schema'],r['expected'],value,'turn_end')['exact']
    atomic_json(REPORT/'scorer-input-checks.json',dict(status='passed',gold_not_a_generation_argument=True,
        strict_types=True,duplicate_keys_rejected=True,markdown_not_repaired=True,token_limits_not_exact=True,array_associations_and_order_checked=True))

def verify():
    import mlx.core as mx
    import importlib.metadata
    from mlx.utils import tree_flatten
    from sml_v1.checkpoint_bundle import resolve_bundle
    from sml_v1.model import TransformerConfig
    from dataclasses import asdict
    from json_restart_130 import train as ENGINE
    from json_sampling_119.run import active_workers
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not any(a['run'].exists() for a in ARMS.values()),'Do not prepare over initialized SFT'
    assert importlib.metadata.version('mlx')=='0.32.3'
    checks()
    with lock(LOCK):
        assert not active_workers(),'Another model worker is active';workers_stopped();protected()
        for a in ARMS.values():
            resolved=Path(resolve_bundle(a['parent']));assert sha(resolved)==a['sha256']
            meta=load(Path(str(resolved)+'.json'));assert meta['cpt151']['added_tokens']==a['added_tokens']
            assert asdict(TransformerConfig(**meta['recipe']['model']))==asdict(TransformerConfig(**load(Path(str(BASE)+'.json'))['recipe']['model']))
        mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3)
        # Predetermined DEV probes check cached control reproduction. No new test set is run.
        dev=rows(DATA/'development.jsonl');probe=[]
        for family in ['atomic/nested_object','composite/nested_delivery','composite/record_allocations']:
            probe.extend([r for r in dev if r['family']==family][:2])
        probe.extend([r for r in dev if r['component']=='compatibility'][:2]);assert len(probe)==8
        control_model=ROOT/'pretraining/runs'/KEPT['142/563'][0]/'model.safetensors'
        cached={o['id']:o for o in load(REPORT/'control'/'development_0000563.json')['outputs']}
        model=R.model_at(control_model);model.eval();enc=R.Encoding()
        for r in probe:
            o=R.generate_one(model,enc,r['document'],r['schema'],r['instruction'])
            assert o['text']==cached[r['id']]['text'] and o['stop_reason']==cached[r['id']]['stop_reason'],'Cached control no longer reproduces'
        del model;mx.clear_cache()
        atomic_json(REPORT/'control-replay-probe.json',dict(status='passed',model_sha256=sha(control_model),
            ids=[r['id'] for r in probe],raw_generated_text_and_stop_match_cached=True,scope='Eight predetermined observed DEV probes; all cached panels separately rescored.'))
        train=rows(DATA/'train.jsonl');chosen=[]
        for component in dict.fromkeys(r['component'] for r in train):chosen.extend([r for r in train if r['component']==component][:2])
        for family in ['composite/nested_delivery','composite/mixed_inventory']:chosen.append(next(r for r in train if r['family']==family))
        assert len(chosen)==16
        evidence={}
        for arm,a in ARMS.items():
            mx.random.seed(PLAN['seed']);model=R.model_at(a['parent']);before=nll(model,chosen)
            p=dict(PLAN,batch_size=8,eval_steps=[2]);batches=[list(range(8)),list(range(8,16))]
            for k in ['betas','eps','clip','weight_decay']:assert ENGINE.PLAN[k]==PLAN[k],k
            with tempfile.TemporaryDirectory(prefix='sft152-matched-smoke-') as tmp,patch.object(ENGINE,'Encoding',R.Encoding),patch.object(ENGINE,'print',T.engine_print,create=True):
                out=Path(tmp);identity=dict(preflight=arm,parent_sha256=a['sha256'])
                ENGINE.execute(model,chosen,out,identity,custom_plan=p,custom_batches=batches,stop_after=1,assessor=lambda *args:None)
                state=load(out/'result.json');assert state['status']=='stopped' and state['step']==1 and state['records']==8
                weight=resolve_bundle(out/'latest.json');opt=mx.load(weight+'.optimizer.safetensors')
                assert int(opt['step'].item())==1,'SFT inherited the pretraining optimizer step'
                assert all(v.dtype==mx.float32 for _,v in opt.items() if mx.issubdtype(v.dtype,mx.floating))
                del opt,model;mx.clear_cache();model=R.model_at(a['parent'])
                ENGINE.execute(model,chosen,out,identity,custom_plan=p,custom_batches=batches,resume=True,assessor=lambda *args:None)
                state=load(out/'result.json');assert state['status']=='sft_complete' and state['step']==2 and state['records']==16
                after=nll(model,chosen);assert after<before,(arm,before,after)
            del model;mx.clear_cache()
            evidence[arm]=dict(status='passed',parent_sha256=a['sha256'],updates=2,rows=16,fresh_sft_optimizer=True,
                optimizer_save_resume=True,train_nll_before=before,train_nll_after=after,scope='Disposable fit/runtime check, not model-quality evidence.')
            print('[SFT152 smoke passed] '+dumps(dict(arm=arm,**evidence[arm])),flush=True)
        atomic_json(REPORT/'engine-smoke.json',evidence);protected()
        receipts={}
        for p in [REPORT/'data-audit.json',REPORT/'control-verification.json',REPORT/'scorer-input-checks.json',
                  REPORT/'control-replay-probe.json',REPORT/'engine-smoke.json',REPORT/'pretraining-stop-verification.json']:
            receipts[str(p.relative_to(ROOT))]=sha(p)
        receipts.update(load(DATA/'manifest.json')['control_receipts'])
        assert not any(a['run'].exists() for a in ARMS.values())
        atomic_json(REPORT/'ready.json',dict(status='passed',contracts={k:contract(k) for k in ARMS},receipts=receipts,
            live_training_started=False,control_reused=True,reserved_predictions=0,benchmarks_used_for_training=False))
        print('[SFT152 READY] Both matched arms verified; full training not started',flush=True)

if __name__=='__main__':verify()
