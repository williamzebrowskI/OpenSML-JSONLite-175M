import collections,json,math,time
from pathlib import Path
import numpy as np
import jsonschema
from .data import ROOT,REPORT,SNIPS,QA,TOK,PARENT_SHA,schema,messages,singleton,labeled_messages,dumps,load
from extract_068.evaluate import score as label_score,aggregate as extract_aggregate
from qa_066.evaluate import score as qa_score,aggregate as qa_aggregate,retention as qa_retention
from sml_v1.common import atomic_json

def strict_json(text):
 def pairs(xs):
  d={}
  for k,v in xs:
   if k in d:raise ValueError('duplicate key')
   d[k]=v
  return d
 def constant(x):raise ValueError(x)
 return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)

def decode(r,n,text):
 if n in (90,91,98,99):result=strict_json(text)
 elif n in (92,93):
  result={}
  for line in text.strip().splitlines():
   if ': ' not in line:raise ValueError('invalid field line')
   f,v=line.split(': ',1)
   if f not in r['fields'] or not v:raise ValueError('invalid key/value')
   if v=='NOT_STATED':
    if f in result:raise ValueError('duplicate empty field')
    result[f]=[]
   else:
    if f in result and not result[f]:raise ValueError('empty plus value')
    result.setdefault(f,[]).append(v)
 elif n==94:
  values=strict_json(text)
  if not isinstance(values,list) or len(values)!=len(r['fields']):raise ValueError('wrong field count')
  result=dict(zip(r['fields'],values))
 elif n==95:
  if len(r['fields'])!=1 or not text.strip():raise ValueError('empty output')
  result={r['fields'][0]:[] if text.strip()=='NOT_STATED' else text.strip().splitlines()}
 elif n in (96,97):
  indexes=strict_json(text)
  if not isinstance(indexes,dict) or set(indexes)!=set(r['fields']):raise ValueError('wrong keys')
  result={}
  for f,xs in indexes.items():
   if not isinstance(xs,list):raise ValueError('wrong span list')
   result[f]=[];end=-1
   for pair in xs:
    if not isinstance(pair,list) or len(pair)!=2 or any(type(x)!=int for x in pair):raise ValueError('bad span')
    a,b=pair
    if not (0<=a<b<=len(r['document'])) or a<end:raise ValueError('bad span bounds/order')
    result[f].append(r['document'][a:b]);end=b
 else:raise ValueError(n)
 if not isinstance(result,dict) or set(result)!=set(r['fields']):raise ValueError('wrong final object keys')
 if any(not isinstance(v,list) or any(not isinstance(s,str) for s in v) for v in result.values()):raise ValueError('bad final values')
 return {f:result[f] for f in r['fields']}

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

def nll(model,tok,ms):
 import mlx.core as mx
 from sft.data import serialize,shifted
 from sft.evaluate import loss_sum
 a,b=serialize(tok,ms)
 # Demonstration is prompt context, not a supervised target.
 if len(ms)==4 and ms[0]['content'].startswith('Extract exact source spans into JSON.'):
  prefix,_=serialize(tok,ms[:-1],generation=True);b[:len(prefix)]=False
 x,y=shifted(a,b)
 return float(loss_sum(model,mx.array(x[None]),mx.array(y[None])).item())

def generate_request(model,tok,r,n,demos):
 from sft.evaluate import generate
 from sft.data import serialize
 pieces=[];losses=[];begin=time.monotonic()
 requests=[singleton(r,f) for f in r['fields']] if n in (93,95) else [r]
 final={};stop='turn_end'
 for rr in requests:
  ms=labeled_messages(rr)[:-1] if n==98 else messages(rr,n,demos,training=False)
  prefix,_=serialize(tok,ms,generation=True)
  if len(prefix)+160>model.cfg.max_seq_len:raise ValueError('would truncate input')
  out=generate(model,tok,ms,limit=160);pieces.append(dict(field=rr['fields'] if len(requests)>1 else None,**out))
  losses.append(nll(model,tok,messages(rr,n,demos)))
  if n==98:
   # The second turn receives actual generated extraction, never gold.
   convert=messages(rr,n,demos,training=False)[-1]
   ms=[dict(role='user',content=ms[0]['content']),dict(role='assistant',content=out['text'] or '<EMPTY EXTRACTION>'),convert]
   prefix,_=serialize(tok,ms,generation=True)
   if len(prefix)+160>model.cfg.max_seq_len:raise ValueError('second stage would truncate')
   out=generate(model,tok,ms,limit=160);pieces.append(dict(stage='json_conversion',**out))
  if any(p['stop_reason'] not in ('turn_end','document_end') for p in pieces):stop=next(p['stop_reason'] for p in pieces if p['stop_reason'] not in ('turn_end','document_end'))
  try:final.update(decode(rr,n,out['text']))
  except (ValueError,TypeError,KeyError,IndexError,RecursionError):
   return dict(text='<INVALID REPRESENTATION>',stop_reason=stop,tokens=sum(p['tokens'] for p in pieces),pieces=pieces,output_nll=sum(losses)/len(losses),seconds=time.monotonic()-begin)
 return dict(text=dumps(final),stop_reason=stop,tokens=sum(p['tokens'] for p in pieces),pieces=pieces,output_nll=sum(losses)/len(losses),seconds=time.monotonic()-begin)

def aggregate_extraction(rows,outs):
 m=extract_aggregate(rows,outs,[o['output_nll'] for o in outs])
 for v in m.values():
  if isinstance(v,dict) and 'counts' in v and 'schema_valid' in v['counts']:
   v['schema_valid_rate']=v['counts']['schema_valid']/v['count'];v['json_parse_rate']=v['counts']['json_parse']/v['count']
 return m

def retention(current,base):
 lower={'fp','fn','false_abstention','missed_abstention','unsupported_values','token_limited'}
 keys=('exact','field_correct','present_correct','absent_correct','tp','fp','fn','false_abstention','missed_abstention','unsupported_values','natural_stop','nonrepetitive_nonempty','format_ok')
 ds={s:{k:current[s]['counts'][k]-v['counts'][k] for k in keys} for s,v in base.items() if isinstance(v,dict) and 'counts' in v}
 pd=current['paired_requests']['both_correct']-base['paired_requests']['both_correct']
 return dict(passed=all(d<=0 if k in lower else d>=0 for vs in ds.values() for k,d in vs.items()) and pd>=0,per_source_deltas=ds,paired_delta=pd)

def qa_gate(m,b):
 out=qa_retention(m,b);ds={}
 lower={'false_abstention','missed_abstention','wrong_value_mentioned','token_limited'}
 keys=('exact','false_abstention','missed_abstention','wrong_value_mentioned','all_requested_values_present','natural_stop','nonrepetitive_nonempty')
 for s,v in b.items():
  if isinstance(v,dict) and 'counts' in v:ds[s]={k:m[s]['counts'][k]-v['counts'][k] for k in keys}
 out['per_source_deltas']=ds
 out['passed'] &= all(d<=0 if k in lower else d>=0 for vs in ds.values() for k,d in vs.items())
 return out

def readiness(m,controls,initial,fit=True):
 j=m['json']['all'];gate=retention(m['json'],controls['labeled'])
 gate['schema_valid']=j['counts']['schema_valid']==j['count'];gate['passed'] &= gate['schema_valid']
 baseline=retention(m['json'],initial['json'])
 parts=dict(json_content_floor=gate,json_vs_start=baseline,labeled_retention=retention(m['labeled'],controls['labeled']),qa_retention=qa_gate(m['qa'],controls['qa']))
 if fit:
  a=m['fit']['all'];parts['train_fit']=dict(passed=a['counts']['exact']>=29 and a['counts']['schema_valid']==32,correct=a['counts']['exact'],required=29,count=32)
 return dict(passed=all(v['passed'] for v in parts.values()),failed_gates=[k for k,v in parts.items() if not v['passed']],**parts)

def controls(model,tok,split):
 from sft.evaluate import generate
 from json_sft.corpus import file_sha
 path=REPORT/f'parent-controls-{split}.json'
 rows=load(SNIPS/f'{split}.jsonl');qa=load(QA/f'{split}.jsonl')
 identity=dict(parent_sha256=PARENT_SHA,rows_sha256=[file_sha(SNIPS/f'{split}.jsonl'),file_sha(QA/f'{split}.jsonl')],generator_sha256=file_sha(ROOT/'pretraining/sft/evaluate.py'),scorer_sha256=file_sha(Path(__file__)),qa_scorer_sha256=file_sha(ROOT/'pretraining/qa_066/evaluate.py'))
 if path.exists():
  d=json.loads(path.read_text());assert d['identity']==identity
  assert [o['id'] for o in d['labeled_outputs']]==[r['id'] for r in rows]
  assert [o['id'] for o in d['qa_outputs']]==[r['id'] for r in qa]
  for r,o in zip(rows,d['labeled_outputs']):assert o['checks']==label_score(r,o['text'],o['stop_reason'])
  for r,o in zip(qa,d['qa_outputs']):assert o['checks']==qa_score(r,o['text'],o['stop_reason'])
  return d
 model.eval();lo=[];qo=[]
 for name,rs,outs,scorer in [('labeled',rows,lo,label_score),('qa',qa,qo,qa_score)]:
  for i,r in enumerate(rs):
   out=generate(model,tok,r['messages'][:-1],limit=160)
   outs.append(dict(id=r['id'],source=r['source'],**out,output_nll=nll(model,tok,r['messages']),checks=scorer(r,out['text'],out['stop_reason'])))
   if (i+1)%100==0:print(f'[parent controls {split}] {name} {i+1}/{len(rs)}',flush=True)
 m=dict(labeled=aggregate_extraction(rows,lo),qa=qa_aggregate(qa,qo,[o['output_nll'] for o in qo]))
 d=dict(identity=identity,metrics=m,labeled_outputs=lo,qa_outputs=qo);atomic_json(path,d);return d

def evaluate(model,tok,n,step,demos,train,base_controls,initial,path,split='development'):
 from sft.evaluate import generate
 model.eval();start=time.monotonic();rows=load(SNIPS/f'{split}.jsonl');qa=load(QA/f'{split}.jsonl');panels={};outputs={}
 fit=sorted(train,key=lambda r:r['id'])[:32] if split=='development' else []
 for name,rs in [('json',rows),('fit',fit)]:
  if not rs:continue
  outs=[]
  for i,r in enumerate(rs):
   out=generate_request(model,tok,r,n,demos)
   outs.append(dict(id=r['id'],source=r['source'],**out,checks=json_score(r,out['text'],out['stop_reason'])))
   if (i+1)%100==0:print(f'[approach {n} eval {step}] {name} {i+1}/{len(rs)} elapsed={time.monotonic()-start:.1f}s',flush=True)
  panels[name]=aggregate_extraction(rs,outs);outputs[name]=outs
 for name,rs,scorer in [('labeled',rows,label_score),('qa',qa,qa_score)]:
  if step==0:
   outs=base_controls[name+'_outputs']
  else:
   outs=[]
   for r in rs:
    out=generate(model,tok,r['messages'][:-1],limit=160);outs.append(dict(id=r['id'],source=r['source'],**out,output_nll=nll(model,tok,r['messages']),checks=scorer(r,out['text'],out['stop_reason'])))
  panels[name]=aggregate_extraction(rs,outs) if name=='labeled' else qa_aggregate(rs,outs,[o['output_nll'] for o in outs]);outputs[name]=outs
 checks=readiness(panels,base_controls['metrics'],initial or panels,fit=bool(fit))
 d=dict(approach=n,step=step,split=split,metrics=panels,outputs=outputs,readiness=checks,seconds=time.monotonic()-start,protocol=dict(temperature=0,repetition_penalty=1,max_tokens_per_call=160,truncation=False,serialization='native-sft-chat-v1',external_serialization=n in (92,93,94,95,96,97),fieldwise=n in (93,95),model_calls=sum(len(o['pieces']) for o in outputs['json'])),scope='SNIPS annotation agreement on observed panels. External assemblers only transform representations; no gold access, field filling, source filtering or JSON repair. Representation-dependent NLL is not comparable across approaches; final content scores are comparable. No general JSON Schema or unseen paragraph reliability claim.')
 atomic_json(path,d);print('[approach metrics]',dumps({k:v for k,v in d.items() if k!='outputs'}),flush=True);model.train();return d
