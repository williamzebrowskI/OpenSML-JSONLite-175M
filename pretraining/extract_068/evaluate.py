"""Exact requested-slot extraction; no paraphrase grader and no JSON repair."""
import json,time,collections,re
from .data import CORPUS,TOK,load
from sml_v1.common import atomic_json

def score(row,text,stop):
 natural=stop in ('turn_end','document_end');parsed={};order=[];valid=True
 for line in text.strip().splitlines():
  m=re.fullmatch(r'([A-Za-z_][A-Za-z_0-9]*): ?(.+)',line)
  if not m:valid=False;continue
  field,value=m.groups();order.append(field)
  if value=='NOT_STATED':
   if field in parsed:valid=False
   parsed[field]=[]
  else:
   if field in parsed and not parsed[field]:valid=False
   parsed.setdefault(field,[]).append(value)
 if set(parsed)!=set(row['fields']):valid=False
 want=collections.Counter((f,v) for f,vs in row['expected'].items() for v in vs)
 got=collections.Counter((f,v) for f,vs in parsed.items() for v in vs)
 tp=sum((want&got).values());fp=sum((got-want).values());fn=sum((want-got).values())
 present=[f for f,v in row['expected'].items() if v];absent=[f for f,v in row['expected'].items() if not v]
 field_correct=sum(parsed.get(f,None)==v for f,v in row['expected'].items()) if natural else 0
 unsupported=sum(v not in row['document'] for _,v in got.elements())
 from sft.quality import repeated
 return dict(exact=int(natural and valid and parsed==row['expected']),format_ok=int(valid),field_correct=field_correct,fields=len(row['fields']),tp=tp,fp=fp,fn=fn,present_correct=sum(parsed.get(f,None)==row['expected'][f] for f in present) if natural else 0,present_total=len(present),absent_correct=sum(parsed.get(f,None)==[] for f in absent) if natural else 0,absent_total=len(absent),false_abstention=sum(parsed.get(f,None)==[] for f in present),missed_abstention=sum(parsed.get(f,None)!=[] for f in absent),unsupported_values=unsupported,predicted_values=sum(got.values()),natural_stop=int(natural),token_limited=int(stop=='token_limit'),nonrepetitive_nonempty=int(bool(text.strip()) and not repeated(text)),requested_order=int(list(dict.fromkeys(order))==row['fields']))

def aggregate(rows,outputs,losses):
 assert [r['id'] for r in rows]==[o['id'] for o in outputs] and len(rows)==len(losses)
 result={}
 for key,indices in [('all',list(range(len(rows))))]+[(s,[i for i,r in enumerate(rows) if r['source']==s]) for s in sorted({r['source'] for r in rows})]:
  sums={k:sum(outputs[i]['checks'][k] for i in indices) for k in outputs[0]['checks']}
  precision=sums['tp']/max(1,sums['tp']+sums['fp']);recall=sums['tp']/max(1,sums['tp']+sums['fn'])
  result[key]=dict(count=len(indices),counts=sums,exact=sums['exact']/len(indices),field_accuracy=sums['field_correct']/max(1,sums['fields']),precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall),present_accuracy=sums['present_correct']/max(1,sums['present_total']),absent_accuracy=sums['absent_correct']/max(1,sums['absent_total']),output_nll=sum(losses[i] for i in indices)/len(indices))
 groups=collections.defaultdict(list)
 for r,o in zip(rows,outputs):
  if r.get('pair_id'):groups[r['pair_id']].append(o)
 assert all(len(g)==2 for g in groups.values())
 result['paired_requests']=dict(count=len(groups),both_correct=sum(all(o['checks']['exact'] for o in g) for g in groups.values()))
 result['macro_exact']=sum(result[s]['exact'] for s in sorted({r['source'] for r in rows}))/len({r['source'] for r in rows})
 return result

def evaluate(model,step,path,split='development'):
 import mlx.core as mx
 from sft.data import Corpus,serialize
 from sft.evaluate import generate,loss_sum
 from sml_v1.tokenization import Tokenizer
 tok=Tokenizer(TOK);rows=load(CORPUS/f'{split}.jsonl');c=Corpus(CORPUS,split);outputs=[];losses=[];start=time.monotonic();model.eval()
 try:
  for i,r in enumerate(rows):
   x,y=c.batch([i]);losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
   prompt,_=serialize(tok,r['messages'][:-1],generation=True);assert len(prompt)+160<=model.cfg.max_seq_len
   out=generate(model,tok,r['messages'][:-1],limit=160);outputs.append(dict(id=r['id'],source=r['source'],**out,checks=score(r,out['text'],out['stop_reason'])))
   if (i+1)%50==0:print(f'[extract068 eval {step}] {split} {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
  m=aggregate(rows,outputs,losses);result=dict(step=step,split=split,metrics=m,output_nll=m['all']['output_nll'],outputs=outputs,protocol=dict(temperature=0,repetition_penalty=1,max_tokens=160,serialization='native-sft-chat-v1',prompt_truncation=False),scope='Original SNIPS chunk annotation agreement, requested-field exact spans; no intent classification, JSON, paraphrase scoring, normalization, or general relation-extraction claim.')
  atomic_json(path,result);print('[extract068 metrics]',json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True);return result
 finally:model.train()

def selection(r):
 m=r['metrics'];a=m['all'];pairs=m['paired_requests'];return (m['macro_exact'],a['field_accuracy'],a['f1'],pairs['both_correct']/max(1,pairs['count']),-a['counts']['unsupported_values'],-r['output_nll'])
