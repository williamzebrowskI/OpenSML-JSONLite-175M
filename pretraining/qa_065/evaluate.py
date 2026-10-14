"""Fixed single-turn generated QA metrics, not schema validity or chat proxies."""
import json,re,string,time
from collections import Counter,defaultdict
from .data import CORPUS,TOK,ROOT,load
from sft.data import Corpus
from sft.quality import repeated
from sml_v1.tokenization import Tokenizer
from sml_v1.common import atomic_json

def norm(s):
 # SQuAD-style normalization: punctuation deletion, articles, whitespace, lowercase.
 s=''.join(c for c in s.lower() if c not in string.punctuation)
 return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def f1(pred,gold):
 p,g=norm(pred).split(),norm(gold).split()
 if not p or not g:return float(p==g)
 common=sum((Counter(p)&Counter(g)).values())
 return 2*common/(len(p)+len(g))
def score(row,text,stop):
 natural=stop in ('turn_end','document_end');abstain=norm(text)==norm('Not stated.');exact=natural and norm(text) in {norm(a) for a in row['answers']}
 checks=dict(exact=int(exact),token_f1=max(f1(text,a) for a in row['answers']),natural_stop=int(natural),token_limited=int(stop=='token_limit'),nonrepetitive_nonempty=int(bool(text.strip()) and not repeated(text)),abstained=int(abstain),false_abstention=int(abstain and not row['answer_absent']),missed_abstention=int(not abstain and row['answer_absent']))
 checks['all_requested_values_present']=int(bool(row.get('required_values')) and all((' '+norm(v)+' ') in (' '+norm(text)+' ') for v in row.get('required_values',[])))
 checks['wrong_value_mentioned']=int(any((' '+norm(v)+' ') in (' '+norm(text)+' ') for v in row.get('forbidden_values',[])))
 return checks

def aggregate(rows,outputs,losses):
 assert len(rows)==len(outputs)==len(losses) and [r['id'] for r in rows]==[o['id'] for o in outputs]
 metrics={}
 for key,indices in [('all',list(range(len(rows))))]+[(s,[i for i,r in enumerate(rows) if r['source']==s]) for s in sorted({r['source'] for r in rows})]:
  if not indices:continue
  checks=[outputs[i]['checks'] for i in indices]
  totals={k:sum(x[k] for x in checks) for k in checks[0]}
  m=dict(count=len(indices),counts=totals,exact=totals['exact']/len(indices),token_f1=totals['token_f1']/len(indices),output_nll=sum(losses[i] for i in indices)/len(indices));metrics[key]=m
  for absent in (False,True):
   ix=[i for i in indices if rows[i]['answer_absent']==absent]
   m['absent' if absent else 'present']=dict(correct=sum(outputs[i]['checks']['exact'] for i in ix),total=len(ix))
 paired=defaultdict(list)
 for r,o in zip(rows,outputs):
  if r.get('pair_id'):paired[r['pair_id']].append(o)
 assert all(len(a)==2 for a in paired.values())
 metrics['question_pairs']=dict(count=len(paired),both_exact=sum(all(x['checks']['exact'] for x in a) for a in paired.values()),same_answer=sum(norm(a[0]['text'])==norm(a[1]['text']) for a in paired.values()))
 metrics['pairs_by_source']={}
 for source in sorted({r['source'] for r in rows if r.get('pair_id')}):
  ids={r['pair_id'] for r in rows if r['source']==source and r.get('pair_id')};groups=[paired[k] for k in ids]
  metrics['pairs_by_source'][source]=dict(count=len(groups),both_exact=sum(all(x['checks']['exact'] for x in g) for g in groups),same_answer=sum(norm(g[0]['text'])==norm(g[1]['text']) for g in groups))
 sq=metrics['squad2'];metrics['balanced_squad_exact']=sum(sq[k]['correct']/sq[k]['total'] for k in ('present','absent'))/2
 legacy=[i for i,r in enumerate(rows) if r.get('panel')=='legacy']
 if legacy:
  metrics['legacy_summary']=dict(count=len(legacy),exact=sum(outputs[i]['checks']['exact'] for i in legacy)/len(legacy),false_abstention=sum(outputs[i]['checks']['false_abstention'] for i in legacy))
 targeted=[metrics[s]['exact'] for s in ('calibration','multipart','corrections') if s in metrics]
 metrics['selection_score']=.5*metrics['balanced_squad_exact']+.5*(sum(targeted)/len(targeted) if targeted else metrics['balanced_squad_exact'])
 return metrics

def evaluate(model,step,path,split='development'):
 import mlx.core as mx
 from sft.evaluate import generate,loss_sum
 rows=load(CORPUS/f'{split}.jsonl');c=Corpus(CORPUS,split);tok=Tokenizer(TOK);model.eval();outputs=[];losses=[];started=time.monotonic()
 try:
  for i,row in enumerate(rows):
   x,y=c.batch([i]);losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
   prompt,_=__import__('sft.data',fromlist=['serialize']).serialize(tok,row['messages'][:-1],generation=True)
   assert len(prompt)+160<=model.cfg.max_seq_len
   result=generate(model,tok,row['messages'][:-1],limit=160)
   outputs.append(dict(id=row['id'],source=row['source'],**result,checks=score(row,result['text'],result['stop_reason'])))
   if (i+1)%20==0:print(f'[qa065 eval {step}] {split} {i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
  metrics=aggregate(rows,outputs,losses)
  ref=ROOT/'pretraining/runs/sft_064_grounded_qa_20261013'/('final-test.json' if split=='test' else 'evaluations/development_0000469.json')
  baseline=json.loads(ref.read_text())['metrics'];sq=metrics['squad2'];old=baseline['squad2']
  retention=dict(legacy_exact_delta=metrics['legacy_summary']['exact']-baseline['all']['exact'],answerable_correct_delta=sq['present']['correct']-old['present']['correct'],absent_correct_delta=sq['absent']['correct']-old['absent']['correct'],paired_correct_delta=metrics['pairs_by_source']['squad2']['both_exact']-baseline['question_pairs']['both_exact'],false_abstention_delta=metrics['legacy_summary']['false_abstention']-baseline['all']['counts']['false_abstention'])
  retention['passed']=all(retention[k]>=0 for k in ('legacy_exact_delta','answerable_correct_delta','absent_correct_delta','paired_correct_delta')) and retention['false_abstention_delta']<=0
  metrics['retention_vs_sft064_selected469']=retention
  result=dict(step=step,split=split,metrics=metrics,outputs=outputs,output_nll=sum(losses)/len(losses),protocol=dict(temperature=0,repetition_penalty=1,max_tokens=160,serialization='native-sft-chat-v1',prompt_truncation=False),scope='Original-target agreement, not certified semantic truth. Dolly EM/F1 are lexical diagnostics and do not judge paraphrase quality. Requested-value coverage is lexical and does not prove correct relationships. Controlled synthetic panel reported separately from legacy QA. Manually review generated answers before promotion. Former experiments may have observed source articles.')
  atomic_json(path,result);print('[qa065 metrics]',json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True);return result
 finally:model.train()
def selection(r):
 m=r['metrics'];p=m['question_pairs'];sq=m['squad2']
 return (int(m.get('retention_vs_sft064_selected469',{}).get('passed',False)),m['selection_score'],m['balanced_squad_exact'],p['both_exact']/max(1,p['count']),sq['token_f1'],-r['output_nll'])
