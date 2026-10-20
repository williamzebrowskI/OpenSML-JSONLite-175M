"""Prepare pinned datasets, test invariants and fresh-base optimizer; cache comparators."""
import json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from .common import *
from . import runtime as R,data as D,run as T
from sml_v1.common import lock


def nll(model,rows):
    import mlx.core as mx
    model.eval();enc=R.Encoding();encoded=[enc.row(r,training=True) for r in rows];values=[]
    for i in range(0,len(rows),2):
        ids=list(range(i,min(i+2,len(rows))));x,y=R.batch(encoded,ids);v=R.loss(model,mx.array(x),mx.array(y)).item();assert math.isfinite(v);values.append((v,len(ids)))
    return sum(v*n for v,n in values)/sum(n for v,n in values)


def verify():
    import mlx.core as mx
    from json_restart_130 import train as ENGINE
    from json_sampling_119.run import active_workers
    assert not RUN.exists(),'Refuse to prepare over an initialized live experiment'
    D.prepare();protected()
    suite=unittest.defaultTestLoader.loadTestsFromName('pretraining.tests.test_json_foundation_142')
    with (REPORT/'tests.log').open('w') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    assert result.wasSuccessful(),'Tests failed; see tests.log'
    print('[SFT142 preflight] counts, valid gold, source splits, reference views, direct JSON, masks and strict scorer passed',flush=True)
    with lock(LOCK):
        if active_workers():raise RuntimeError('Another worker active')
        mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
        rows=read_rows(DATA/'train.jsonl');chosen=[]
        for component in QUOTAS:chosen += [r for r in rows if r['component']==component][:2]
        chosen += [next(r for r in rows if r['family']=='composite/mixed_inventory'),next(r for r in rows if r['family']=='composite/numeric_bundle')]
        assert len(chosen)==16
        model=R.model_at(BASE);before=nll(model,chosen);p=dict(PLAN,batch_size=8,eval_steps=[2]);batches=[list(range(8)),list(range(8,16))]
        with tempfile.TemporaryDirectory(prefix='sft142-pretrained-smoke-') as tmp,patch.object(ENGINE,'Encoding',R.Encoding),patch.object(ENGINE,'print',T.engine_print,create=True):
            tmp=Path(tmp);c=dict(preflight='142-disposable',base_sha256=BASE_SHA)
            ENGINE.execute(model,chosen,tmp,c,custom_plan=p,custom_batches=batches,stop_after=1,assessor=lambda *a:None)
            state=json.loads((tmp/'result.json').read_text());assert state['status']=='stopped' and state['step']==1 and state['records']==8
            del model;mx.clear_cache();model=R.model_at(BASE)
            ENGINE.execute(model,chosen,tmp,c,custom_plan=p,custom_batches=batches,resume=True,assessor=lambda *a:None)
            state=json.loads((tmp/'result.json').read_text());assert state['status']=='sft_complete' and state['step']==2 and state['records']==16
            after=nll(model,chosen);assert after<before
        del model;mx.clear_cache()
        atomic_json(REPORT/'engine-smoke.json',dict(status='passed',starting_weight_sha256=BASE_SHA,starting_stage='pretrained132005',inherited_sft_weights=False,
            real_model=True,updates=2,rows=16,optimizer_save_resume=True,train_nll_before=before,train_nll_after=after,live_training_started=False,scope='Disposable runtime/fit smoke, not model quality evidence.'))
        print('[SFT142 preflight] real PRETRAINED model optimizer/save/resume passed; caching same448-case130 and140 baselines',flush=True)
        T.evaluate_baselines();protected();assert not RUN.exists()
        receipts={str(p.relative_to(ROOT)):sha(p) for p in [REPORT/'tests.log',REPORT/'engine-smoke.json',REPORT/'baseline-130.json',REPORT/'baseline-140.json']}
        atomic_json(REPORT/'ready.json',dict(status='passed',contract=T.contract(),receipts=receipts,tests=result.testsRun,live_training_started=False,inherited_sft_weights=False))
        print('[SFT142 READY] '+dumps(dict(status='passed',tests=result.testsRun,starting_checkpoint='pretrained132005',live_training_started=False,command='bash pretraining/scripts/run_sft_142_fresh_foundation.sh --run')),flush=True)


if __name__=='__main__':verify()
