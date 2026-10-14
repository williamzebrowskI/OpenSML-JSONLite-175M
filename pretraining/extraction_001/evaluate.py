"""Strict extraction, source-grounding, abstention and stopping metrics."""
from collections import Counter
import json,re,string,time
from json_sft.evaluate import generate,score,loss_sum
from json_sft.constrained import generate as constrained_generate
from json_specialist.data import strict_json
from json_sft.corpus import load_tokenizer
from sml_v1.common import atomic_json
from .data import CORPUS,load

def normalized(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',str(s).lower().translate(str.maketrans('','',string.punctuation))).split())
def f1(a,b):
 x=normalized(a).split();y=normalized(b).split()
 if not x or not y:return float(x==y)
 hits=sum((Counter(x)&Counter(y)).values())
 return 2*hits/(len(x)+len(y))
def metrics(row,out):
 checks=score(out['text'],row,out['stop_reason']);valid=checks['schema_valid'];value='__invalid__'
 if valid:value=strict_json(out['text'])['answer']
 supported=row['target']['answer'] is not None;natural=checks['natural_stop']
 exact=valid and natural and value in row['accepted_answers']
 return dict(exact=int(exact),valid_json=int(checks['valid_json']),schema_valid=int(valid),natural_stop=int(natural),token_limited=int(out['stop_reason']=='token_limit'),answerable=int(supported),answerable_correct=int(exact and supported),false_null=int(valid and value is None and supported),predicted_null=int(valid and value is None),correct_null=int(valid and value is None and not supported and natural),missing=int(not supported),unsupported_string=int(valid and isinstance(value,str) and value not in row['document']),span_f1=max(f1(value,a) for a in row['accepted_answers']) if supported and valid and isinstance(value,str) else float(exact and not supported),normalized_exact=int(valid and natural and (value is None and not supported or supported and isinstance(value,str) and any(normalized(value)==normalized(a) for a in row['accepted_answers']))))
def aggregate(items):
 n=len(items);totals={k:sum(i[k] for i in items) for k in items[0]}
 return dict(count=n,counts={k:v for k,v in totals.items() if k!='span_f1'},exact_rate=totals['exact']/n,normalized_exact=totals['normalized_exact']/n,token_f1=totals['span_f1']/n,answerable_accuracy=totals['answerable_correct']/totals['answerable'] if totals['answerable'] else None,null_precision=totals['correct_null']/totals['predicted_null'] if totals['predicted_null'] else None,null_recall=totals['correct_null']/totals['missing'] if totals['missing'] else None)
def selection(r):
 # Equal source weights; absent-only improvement cannot dominate positive extraction.
 return (r['macro_raw_exact'],r['macro_constrained_exact'],-r['output_nll'])
def evaluate(model,step,path,split='development'):
 import mlx.core as mx
 from sft.data import Corpus
 rows=load(CORPUS/(split+'.jsonl'));tok=load_tokenizer();c=Corpus(CORPUS,split);outputs=[];start=time.monotonic();model.eval()
 try:
  losses=[]
  for i in range(len(rows)):
   x,y=c.batch([i]);losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
  summary={}
  for mode,generator in [('raw',generate),('constrained',constrained_generate)]:
   scored=[]
   for i,r in enumerate(rows):
    request={k:r[k] for k in ('document','instruction','schema','_prompt_style')}
    out=generator(model,tok,request,limit=128);m=metrics(r,out);scored.append(m);outputs.append(dict(id=r['id'],mode=mode,source=r['source'],**out,metrics=m))
    if (i+1)%32==0:print(f'[extract-eval {step}] {split} {mode} {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
   summary[mode]={'all':aggregate(scored),'sources':{s:aggregate([m for r,m in zip(rows,scored) if r['source']==s]) for s in sorted({r['source'] for r in rows})}}
  result=dict(step=step,split=split,output_nll=sum(losses)/len(losses),metrics=summary,macro_raw_exact=sum(v['exact_rate'] for v in summary['raw']['sources'].values())/3,macro_constrained_exact=sum(v['exact_rate'] for v in summary['constrained']['sources'].values())/3,outputs=outputs,scope='Annotation agreement on filtered article-disjoint source subsets; no claim of arbitrary-schema capability or release readiness.')
  atomic_json(path,result);print('[extract-metrics]',json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True);return result
 finally:model.train()
