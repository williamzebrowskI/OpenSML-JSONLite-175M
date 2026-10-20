import json,time,math
from .common import *
from .runtime import Encoding,batch,loss,generate
from .scoring import score,aggregate,comparison,original_score,original_aggregate

def evaluate(model,rows,path,weight_sha,step=0,original=False):
    import mlx.core as mx
    path=Path(path);enc=Encoding();model.eval();outs=[];start=time.monotonic()
    identity=dict(weight_sha256=weight_sha,requests_sha256=__import__('hashlib').sha256(dumps(rows).encode()).hexdigest(),
        protocol='raw-greedy-256-native-sft-chat-v1; original instruction; no truncation/assembly/repair',
        files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('scoring.py')]},original=original)
    if path.exists():
        saved=json.loads(path.read_text());assert saved['identity']==identity;outs=saved['outputs']
        for r,o in zip(rows,outs):
            assert r['id']==o['id'] and o['checks']==(original_score(r,o['text'],o['stop_reason']) if original else score(r,o['text'],o['stop_reason']))
    for i in range(len(outs),len(rows)):
        r=rows[i];raw=generate(model,enc.tokenizer,enc.prefix(r['document'],r['schema']))
        # Reference is read only for teacher-forced NLL AFTER raw generation.
        x,y=batch([enc.row(r)],[0]);nll=loss(model,mx.array(x),mx.array(y)).item()
        if not math.isfinite(nll):raise ValueError('Nonfinite evaluation NLL')
        checks=original_score(r,raw['text'],raw['stop_reason']) if original else score(r,raw['text'],raw['stop_reason'])
        outs.append(dict(id=r['id'],**raw,checks=checks,output_nll=nll))
        if (i+1)%40==0 or i+1==len(rows):
            atomic_json(path,dict(status='evaluating',identity=identity,outputs=outs));print(f'[joint130 eval {step}] {path.stem} {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    metrics=original_aggregate(rows,outs,[o['output_nll'] for o in outs]) if original else aggregate(rows,outs)
    result=dict(status='complete',identity=identity,step=step,outputs=outs,metrics=metrics)
    atomic_json(path,result);print('[joint130 metrics] '+dumps(dict(step=step,split=path.stem,scorer='published original extracted-value' if original else 'typed path/value diagnostic; not published extracted-value F1',metrics=metrics)),flush=True)
    model.train();return result
