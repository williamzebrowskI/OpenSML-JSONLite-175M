#!/usr/bin/env python3
"""Run the observed adapted SNIPS extraction panel with native JSONLite."""
import argparse,collections,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'pretraining'),str(ROOT/'inference_native')]
import jsonschema
from adapter import Extractor,strict_json
def schema(row):return row['schema']
def json_score(r,text,stop):
 parsed={};valid=False;parse=False
 try:
  obj=strict_json(text);parse=True
  if isinstance(obj,dict):parsed={k:v for k,v in obj.items() if isinstance(v,list) and all(isinstance(s,str) for s in v)}
  valid=jsonschema.Draft202012Validator(schema(r)).is_valid(obj)
 except (ValueError,TypeError,RecursionError):pass
 want=collections.Counter((f,v) for f,vs in r['expected'].items() for v in vs);got=collections.Counter((f,v) for f,vs in parsed.items() for v in vs)
 present=[f for f in r['fields'] if r['expected'][f]];absent=[f for f in r['fields'] if not r['expected'][f]];natural=stop in ('turn_end','document_end')
 from sft.quality import repeated
 return dict(exact=int(natural and valid and parsed==r['expected']),format_ok=int(valid),json_parse=int(parse),schema_valid=int(valid),field_correct=sum(parsed.get(f,None)==v for f,v in r['expected'].items()) if natural else 0,fields=len(r['fields']),tp=sum((want&got).values()),fp=sum((got-want).values()),fn=sum((want-got).values()),present_correct=sum(parsed.get(f,None)==r['expected'][f] for f in present) if natural else 0,present_total=len(present),absent_correct=sum(parsed.get(f,None)==[] for f in absent) if natural else 0,absent_total=len(absent),false_abstention=sum(parsed.get(f,None)==[] for f in present),missed_abstention=sum(parsed.get(f,None)!=[] for f in absent),unsupported_values=sum(v not in r['document'] for _,v in got.elements()),predicted_values=sum(got.values()),natural_stop=int(natural),token_limited=int(stop=='token_limit'),nonrepetitive_nonempty=int(bool(text.strip()) and not repeated(text)),requested_order=int(list(parsed)==r['fields']))

def aggregate(rows,outs):
    metrics={}
    for category in ['all']+sorted({r['source'] for r in rows}):
        group=outs if category=='all' else [o for o in outs if o['source']==category]
        sums={k:sum(o['checks'][k] for o in group) for k in group[0]['checks']}
        precision=sums['tp']/max(1,sums['tp']+sums['fp']);recall=sums['tp']/max(1,sums['tp']+sums['fn'])
        metrics[category]=dict(count=len(group),counts=sums,exact=sums['exact']/len(group),
          field_accuracy=sums['field_correct']/sums['fields'],precision=precision,recall=recall,
          f1=2*precision*recall/max(1e-12,precision+recall),
          present_accuracy=sums['present_correct']/max(1,sums['present_total']),
          absent_accuracy=sums['absent_correct']/max(1,sums['absent_total']))
    pairs=collections.defaultdict(list)
    for r,o in zip(rows,outs):pairs[r['pair_id']].append(o)
    metrics['paired_requests']=dict(count=len(pairs),both_correct=sum(all(o['checks']['exact'] for o in group) for group in pairs.values()))
    return metrics

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite previous results')
    source=ROOT/'evaluation/snips-heldout.jsonl';rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows)==1388
    model=Extractor(a.weights,ROOT);outs=[];start=time.monotonic()
    for i,row in enumerate(rows):
        # The reference never enters the extractor call.
        result=model.generate(row['document'],row['schema'])
        checks=json_score(row,result['text'],result['stop_reason'])
        outs.append(dict(id=row['id'],source=row['source'],result=result,checks=checks))
        if (i+1)%100==0 or i+1==len(rows):print('[JSONLite eval]',i+1,'/',len(rows),flush=True)
    data=dict(complete=True,evaluated=len(outs),expected=len(rows),weight_sha256=model.config['weight_sha256'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),metrics=aggregate(rows,outs),outputs=outs,elapsed_seconds=time.monotonic()-start,protocol=dict(temperature=0,repetition_penalty=1,max_tokens_per_field=160,prompt_truncation=False,gold_sent_to_model=False),scope='Previously observed adapted SNIPS annotation agreement; 694 sources, two views each. Not an official intent-classification benchmark or fresh blind test.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(data['metrics'],indent=2))
if __name__=='__main__':main()
