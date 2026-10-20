"""Typed diagnostic scores and separately labelled unchanged original SNIPS scores."""
import collections,json,math
from decimal import Decimal
from jsonschema import Draft202012Validator
from native_json_128.scoring import strict_json,score as original_score,aggregate as original_aggregate
from sft.quality import repeated

def typed(v):
    if v is None:return ('null',)
    if isinstance(v,bool):return ('boolean',v)
    if isinstance(v,(int,float)):return ('number',str(Decimal(str(v)).normalize()))
    if isinstance(v,str):return ('string',v)
    if isinstance(v,list):return ('array',tuple(typed(x) for x in v))
    return ('object',tuple(sorted((k,typed(x)) for k,x in v.items())))

def leaves(v,path=()):
    if isinstance(v,dict) and v:return [a for k,x in v.items() for a in leaves(x,path+(k,))]
    if isinstance(v,list) and v:return [a for i,x in enumerate(v) for a in leaves(x,path+(i,))]
    return [(path,typed(v))]

def strings(v):
    if isinstance(v,str):return [v]
    if isinstance(v,list):return [s for x in v for s in strings(x)]
    if isinstance(v,dict):return [s for x in v.values() for s in strings(x)]
    return []

def score(r,text,stop):
    parse=False;valid=False;v={};natural=stop in ('turn_end','document_end')
    try:v=strict_json(text);parse=True;valid=Draft202012Validator(r['schema']).is_valid(v)
    except (ValueError,TypeError):pass
    obj=v if isinstance(v,dict) else {};want=r['expected'];a=collections.Counter(leaves(want));b=collections.Counter(leaves(v)) if parse else collections.Counter()
    tp=sum((a&b).values());good=lambda k:k in obj and typed(obj[k])==typed(want[k]) and Draft202012Validator(r['schema']['properties'][k]).is_valid(obj[k])
    empty=lambda x:x is None or x==[];missing=[k for k,x in want.items() if empty(x)];present=[k for k in want if k not in missing]
    return dict(exact=int(natural and valid and typed(v)==typed(want)),json_parse_ok=int(parse),schema_ok=int(valid),
        field_correct=sum(natural and good(k) for k in want),fields=len(set(want)|set(obj)),tp=tp,fp=sum(b.values())-tp,fn=sum(a.values())-tp,
        present_correct=sum(natural and good(k) for k in present),present_total=len(present),absent_correct=sum(natural and good(k) for k in missing),absent_total=len(missing),
        false_abstention=sum(k in obj and empty(obj[k]) for k in present),missed_abstention=sum(not good(k) for k in missing),
        unsupported_values=sum(s not in r['document'] for s in strings(v)),predicted_values=len(strings(v)),natural_stop=int(natural),token_limited=int(stop=='token_limit'),
        nonrepetitive_nonempty=int(bool(text.strip()) and not repeated(text)))

def aggregate(rows,outs):
    assert [r['id'] for r in rows]==[o['id'] for o in outs]
    sets=[('all',list(range(len(rows))))]
    sets += [('family/'+f,[i for i,r in enumerate(rows) if r['family']==f]) for f in sorted({r['family'] for r in rows})]
    sets += [('panel/'+f,[i for i,r in enumerate(rows) if r.get('panel','all')==f]) for f in sorted({r.get('panel','all') for r in rows})]
    sets += [('domain/'+f,[i for i,r in enumerate(rows) if r['source']==f]) for f in sorted({r['source'] for r in rows if r['family']=='snips'})]
    metrics={}
    for name,idx in sets:
        c={k:sum(outs[i]['checks'][k] for i in idx) for k in outs[0]['checks']};ratio=lambda a,b:a/b if b else None
        metrics[name]=dict(count=len(idx),counts=c,exact=c['exact']/len(idx),field_accuracy=ratio(c['field_correct'],c['fields']),
            leaf_precision=ratio(c['tp'],c['tp']+c['fp']),leaf_recall=ratio(c['tp'],c['tp']+c['fn']),leaf_f1=ratio(2*c['tp'],2*c['tp']+c['fp']+c['fn']),
            present_accuracy=ratio(c['present_correct'],c['present_total']),missing_accuracy=ratio(c['absent_correct'],c['absent_total']),
            false_abstention_rate=ratio(c['false_abstention'],c['present_total']),missed_abstention_rate=ratio(c['missed_abstention'],c['absent_total']),
            unsupported_value_rate=ratio(c['unsupported_values'],c['predicted_values']),json_parse_rate=c['json_parse_ok']/len(idx),schema_valid_rate=c['schema_ok']/len(idx),
            natural_stop_rate=c['natural_stop']/len(idx),token_limited_rate=c['token_limited']/len(idx),nonrepetitive_nonempty_rate=c['nonrepetitive_nonempty']/len(idx),
            output_nll=sum(outs[i]['output_nll'] for i in idx)/len(idx))
    pairs=collections.defaultdict(list)
    for r,o in zip(rows,outs):
        if r.get('pair_id'):pairs[r['pair_id']].append(o)
    assert all(len(v)==2 for v in pairs.values())
    metrics['paired_requests']=dict(count=len(pairs),both_correct=sum(all(o['checks']['exact'] for o in v) for v in pairs.values()))
    metrics['macro_exact']=sum(metrics['family/'+f]['exact'] for f in {r['family'] for r in rows})/len({r['family'] for r in rows})
    return metrics

def comparison(rows,baseline,candidate):
    assert [r['id'] for r in rows]==[o['id'] for o in baseline['outputs']]==[o['id'] for o in candidate['outputs']]
    before={o['id']:o['checks']['exact'] for o in baseline['outputs']};after={o['id']:o['checks']['exact'] for o in candidate['outputs']}
    fixed=[i for i in before if not before[i] and after[i]];lost=[i for i in before if before[i] and not after[i]]
    return dict(fixed=fixed,lost=lost,net=len(fixed)-len(lost),strict_case_retention_passed=not lost,
        metric_deltas={k:candidate['metrics']['all'][k]-baseline['metrics']['all'][k] for k in ['exact','field_accuracy','leaf_f1','schema_valid_rate']})

def selection(metrics):
    # New skill family and original compatibility exact receive equal weight.
    old=metrics['family/snips']['exact'];new=[v['exact'] for k,v in metrics.items() if k.startswith('family/') and k!='family/snips']
    return [.5*old+.5*sum(new)/len(new),old,metrics['all']['field_accuracy'],metrics['all']['leaf_f1']]
