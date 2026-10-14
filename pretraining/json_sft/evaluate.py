"""Strict JSON scoring and raw, unconstrained extraction generation."""
from collections import defaultdict
import math
from pathlib import Path
from jsonschema import FormatChecker, validators
from json_specialist.data import strict_json
from .ordered_serialization import prompt
from .corpus import rank

def equal_json(a,b):
    if isinstance(a,bool) or isinstance(b,bool): return type(a) is type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)): return a==b
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_json(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(equal_json(x,y) for x,y in zip(a,b))
    return a==b

def score(text,row,reason='document_end'):
    result={'valid_json':False,'schema_valid':False,'exact_json':False,'natural_stop':reason=='document_end'}
    try: value=strict_json(text)
    except (ValueError,TypeError):return result
    result['valid_json']=True
    try:
        validators.validator_for(row['schema'])(row['schema'],format_checker=FormatChecker()).validate(value)
        result['schema_valid']=True
    except Exception as e:
        from jsonschema import ValidationError, SchemaError
        if not isinstance(e,(ValidationError,SchemaError)):raise
    result['exact_json']=bool(result['schema_valid'] and result['natural_stop'] and equal_json(value,row['target']))
    return result

def panel(rows,per_source):
    sources=defaultdict(list)
    for i,row in enumerate(rows):sources[row['source']].append(i)
    return {s:sorted(indices,key=lambda i:rank('panel:'+rows[i]['id']))[:per_source] for s,indices in sorted(sources.items())}

def generate(model,tokenizer,row,limit=512):
    import mlx.core as mx
    ids=tokenizer.encode(prompt(row),add_special_tokens=False).ids
    if len(ids)>1536 or any(i<4 for i in ids):raise ValueError('Unsupported input length/tokens; no truncation')
    limit=min(limit,model.cfg.max_seq_len-len(ids))
    if limit<1:raise ValueError('No generation space')
    inputs=mx.array([ids],dtype=mx.int32);cache=None;out=[];reason='token_limit'
    for _ in range(limit):
        logits,cache=model.logits(inputs,cache)
        token=int(mx.argmax(logits[0,-1]).item())
        if token==1:reason='document_end';break
        if token<4:reason='unexpected_structure';break
        out.append(token);inputs=mx.array([[token]],dtype=mx.int32)
    return {'text':tokenizer.decode(out,skip_special_tokens=False),'output_tokens':len(out),'stop_reason':reason}

def loss_sum(model,x,y):
    import mlx.core as mx
    mask=y!=-100
    losses=model(x,targets=y)['token_loss']
    return ((losses*mask).sum(axis=1)/mx.maximum(mask.sum(axis=1),1)).sum()

def evaluate(model,tokenizer,prepared,step,output,*,split='development',per_source=12):
    import mlx.core as mx
    from sft.data import Corpus
    from sml_v1.common import atomic_json
    if split not in ('development','test'):raise ValueError('Evaluation requires a held-out split')
    root=Path(prepared);rows=[strict_json(s) for s in (root/(split+'.jsonl')).open()]
    corpus=Corpus(root,split);components={};outputs=[]
    model.eval()
    try:
        for source,indices in panel(rows,32).items():
            values=[]
            for i in indices:
                x,y=corpus.batch([i]);value=float(loss_sum(model,mx.array(x),mx.array(y)).item())
                if not math.isfinite(value):raise RuntimeError('Nonfinite output validation loss')
                values.append(value)
            generated=[]
            for i in panel(rows,per_source)[source]:
                result=generate(model,tokenizer,rows[i]);checks=score(result['text'],rows[i],result['stop_reason']);generated.append(checks)
                outputs.append(dict(id=rows[i]['id'],source=source,document=rows[i]['document'],schema=rows[i]['schema'],instruction=rows[i]['instruction'],target=rows[i]['target'],**result,checks=checks))
            components[source]={'output_nll':sum(values)/len(values),'nll_records':len(values),'generated_records':len(generated),
                **{k:sum(v[k] for v in generated)/len(generated) for k in generated[0]}}
        smoke=[]
        for row in strict_json((root/'extraction-smoke-10.json').read_text())['cases']:
            row=dict(row,target=row['expected'],_prompt_style=strict_json((root/'manifest.json').read_text())['serialization'])
            result=generate(model,tokenizer,row);smoke.append(dict(id=row['id'],**result,checks=score(result['text'],row,result['stop_reason'])))
        summary={'step':step,'split':split,'components':components,'macro_exact_json':sum(c['exact_json'] for c in components.values())/len(components),
            'macro_output_nll':sum(c['output_nll'] for c in components.values())/len(components),
            'smoke_exact':sum(r['checks']['exact_json'] for r in smoke),'smoke_count':10,
            'selection':'Development source-macro exact generated JSON; tie: output NLL. Smoke/test never select.',
            'scope':'Unconstrained greedy generation; exact JSON is annotation agreement, not independently established semantic correctness.'}
        atomic_json(Path(output)/f'{split}_step_{step:07d}.json',dict(**summary,generated_outputs=outputs,extraction_smoke=smoke))
        print(f'[eval-json {step}] dev_exact={summary["macro_exact_json"]:.3f} output_nll={summary["macro_output_nll"]:.4f} smoke={summary["smoke_exact"]}/10',flush=True)
        return summary
    finally:model.train()
