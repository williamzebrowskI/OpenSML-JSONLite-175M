"""Frozen exact metrics plus tested conservative semantic equivalence."""
import copy,json,time
from .data import ROOT,CORPUS,TOK,PARENT_SHA,load
from qa_065.evaluate import score as exact_score,aggregate as exact_aggregate
from qa_semantics.score import equivalent,VERSION
from sml_v1.common import atomic_json
PARENT_RUN=ROOT/'pretraining/runs/sft_065_targeted_qa_20261013'

def score(row,text,stop):
 c=exact_score(row,text,stop)
 c['relation_correct']=int(equivalent(row,text,stop)['correct']) if 'oracle' in row else c['exact']
 return c

def aggregate(rows,outputs,losses):
 m=exact_aggregate(rows,outputs,losses)
 targeted=[]
 for s in ('calibration','multipart','corrections'):
  if s in m:
   m[s]['relation_accuracy']=m[s]['counts']['relation_correct']/m[s]['count'];targeted.append(m[s]['relation_accuracy'])
 m['selection_score']=.5*m['balanced_squad_exact']+.5*(sum(targeted)/len(targeted) if targeted else m['balanced_squad_exact'])
 return m

def cached_parent(split):
 path=PARENT_RUN/('final-test.json' if split=='test' else 'evaluations/development_0000469.json')
 d=json.loads(path.read_text());rows=load(CORPUS/f'{split}.jsonl');assert [r['id'] for r in rows]==[o['id'] for o in d['outputs']]
 outs=[dict(o,checks=score(r,o['text'],o['stop_reason'])) for r,o in zip(rows,d['outputs'])]
 metrics=aggregate(rows,outs,[0.]*len(rows))
 # Per-row losses weren't saved. Restore unchanged source/overall means from
 # the original evaluation; do not invent per-row values.
 for k,v in metrics.items():
  if isinstance(v,dict) and 'output_nll' in v:v['output_nll']=d['metrics'][k]['output_nll']
 return rows,outs,metrics,d

def retention(m,baseline):
 sq=m['squad2'];old=baseline['squad2']
 r=dict(legacy_exact_delta=m['legacy_summary']['exact']-baseline['legacy_summary']['exact'],answerable_correct_delta=sq['present']['correct']-old['present']['correct'],absent_correct_delta=sq['absent']['correct']-old['absent']['correct'],paired_correct_delta=m['pairs_by_source']['squad2']['both_exact']-baseline['pairs_by_source']['squad2']['both_exact'],false_abstention_delta=m['legacy_summary']['false_abstention']-baseline['legacy_summary']['false_abstention'])
 for s in ('calibration','multipart','corrections'):r[s+'_relation_delta']=m[s]['relation_accuracy']-baseline[s]['relation_accuracy']
 r['passed']=all(v>=-1e-12 for k,v in r.items() if k!='false_abstention_delta') and r['false_abstention_delta']<=0
 return r

def evaluate(model,step,path,split='development'):
 rows,parent_out,parent_metrics,parent=cached_parent(split)
 if step==0:
  identity=json.loads((PARENT_RUN/('final-test-identity.json' if split=='test' else 'evaluations/development_0000469.identity.json')).read_text())
  assert identity['weight_sha256' if split=='test' else 'model_sha256']==PARENT_SHA
  outputs=parent_out;metrics=parent_metrics;nll=parent['output_nll'];mode='verified cached parent predictions; no generation'
 else:
  import mlx.core as mx
  from sft.data import Corpus,serialize
  from sft.evaluate import generate,loss_sum
  from sml_v1.tokenization import Tokenizer
  corpus=Corpus(CORPUS,split);tok=Tokenizer(TOK);outputs=[];losses=[];model.eval();start=time.monotonic()
  try:
   for i,r in enumerate(rows):
    x,y=corpus.batch([i]);losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
    prompt,_=serialize(tok,r['messages'][:-1],generation=True);assert len(prompt)+160<=model.cfg.max_seq_len
    result=generate(model,tok,r['messages'][:-1],limit=160)
    outputs.append(dict(id=r['id'],source=r['source'],**result,checks=score(r,result['text'],result['stop_reason'])))
    if (i+1)%40==0:print(f'[qa066 eval {step}] {split} {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
   metrics=aggregate(rows,outputs,losses);nll=sum(losses)/len(losses);mode='generated'
  finally:model.train()
 metrics['retention_vs_parent065']=retention(metrics,parent_metrics)
 ref=json.loads((ROOT/'pretraining/runs/sft_064_grounded_qa_20261013'/('final-test.json' if split=='test' else 'evaluations/development_0000469.json')).read_text())['metrics']
 metrics['legacy_vs064']=dict(exact_delta=metrics['legacy_summary']['exact']-ref['all']['exact'],absent_correct_delta=metrics['squad2']['absent']['correct']-ref['squad2']['absent']['correct'])
 result=dict(step=step,split=split,metrics=metrics,outputs=outputs,output_nll=nll,mode=mode,protocol=dict(temperature=0,repetition_penalty=1,max_tokens=160,serialization='native-sft-chat-v1',prompt_truncation=False,relation_evaluator=VERSION),scope='Conservative relation-equivalence lower bound on controlled examples; unrecognized does not prove incorrect. Exact natural-QA metrics unchanged. Previously observed held-out panels; no fresh blind benchmark claim.')
 atomic_json(path,result);print('[qa066 metrics]',json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True);return result

def selection(r):
 m=r['metrics'];p=m['question_pairs']
 return (int(m.get('retention_vs_parent065',{}).get('passed',False)),m['selection_score'],m['balanced_squad_exact'],p['both_exact']/max(1,p['count']),m['squad2']['token_f1'],-r['output_nll'])
