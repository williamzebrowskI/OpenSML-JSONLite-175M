"""One model call per source/schema request. Score the exact unmodified output."""
import collections,json,time
from jsonschema import Draft202012Validator,ValidationError
from sml_v1.common import atomic_json
from .data import CORPUS,TOK,PLAN,load

def strict_json(text):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 def nonfinite(value):raise ValueError('Nonfinite JSON number')
 return json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)

def values(value):
 if isinstance(value,str):return [value]
 if isinstance(value,list):return [v for v in value if isinstance(v,str)]
 return []

def score(row,text,stop):
 natural=stop in ('turn_end','document_end');parsed=None;parse_ok=False;schema_ok=False
 try:
  parsed=strict_json(text);parse_ok=True
  Draft202012Validator(row['schema']).validate(parsed);schema_ok=True
 except (ValueError,TypeError,ValidationError):pass
 obj=parsed if isinstance(parsed,dict) else {};expected=row['expected'];props=row['schema']['properties']
 def valid_field(k):
  return k in obj and not list(Draft202012Validator(props[k]).iter_errors(obj[k]))
 correct={k:natural and valid_field(k) and obj[k]==v for k,v in expected.items()}
 want=collections.Counter((k,v) for k,x in expected.items() for v in values(x))
 got=collections.Counter((k,v) for k,x in obj.items() for v in values(x))
 tp=sum((want&got).values());fp=sum((got-want).values());fn=sum((want-got).values())
 present=[k for k,v in expected.items() if values(v)];absent=[k for k in expected if k not in present]
 empty=lambda k: k in obj and valid_field(k) and obj[k] in ([],None)
 from sft.quality import repeated
 return dict(exact=int(natural and schema_ok and obj==expected),json_parse_ok=int(parse_ok),schema_ok=int(schema_ok),
  field_correct=sum(correct.values()),fields=len(expected),tp=tp,fp=fp,fn=fn,
  present_correct=sum(correct[k] for k in present),present_total=len(present),absent_correct=sum(correct[k] for k in absent),absent_total=len(absent),
  false_abstention=sum(empty(k) for k in present),missed_abstention=sum(not empty(k) for k in absent),
  unsupported_values=sum(v not in row['document'] for _,v in got.elements()),predicted_values=sum(got.values()),
  natural_stop=int(natural),token_limited=int(stop=='token_limit'),nonrepetitive_nonempty=int(bool(text.strip()) and not repeated(text)))

def aggregate(rows,outputs,losses):
 if [r['id'] for r in rows]!=[o['id'] for o in outputs] or len(rows)!=len(losses):raise ValueError('Prediction coverage mismatch')
 metrics={};panels=[('all',list(range(len(rows))))]
 panels += [('domain/'+s,[i for i,r in enumerate(rows) if r['source']==s]) for s in sorted({r['source'] for r in rows})]
 panels += [('variant/'+s,[i for i,r in enumerate(rows) if r['variant']==s]) for s in sorted({r['variant'] for r in rows})]
 for name,idx in panels:
  c={k:sum(outputs[i]['checks'][k] for i in idx) for k in outputs[0]['checks']}
  precision=c['tp']/max(1,c['tp']+c['fp']);recall=c['tp']/max(1,c['tp']+c['fn'])
  metrics[name]=dict(count=len(idx),counts=c,exact=c['exact']/len(idx),json_parse_rate=c['json_parse_ok']/len(idx),schema_valid_rate=c['schema_ok']/len(idx),
    field_accuracy=c['field_correct']/max(1,c['fields']),precision=precision,recall=recall,f1=2*precision*recall/max(1e-12,precision+recall),
    present_accuracy=c['present_correct']/max(1,c['present_total']),absent_accuracy=c['absent_correct']/max(1,c['absent_total']),
    false_abstention_rate=c['false_abstention']/max(1,c['present_total']),missed_abstention_rate=c['missed_abstention']/max(1,c['absent_total']),
    unsupported_value_rate=c['unsupported_values']/max(1,c['predicted_values']),natural_stop_rate=c['natural_stop']/len(idx),
    token_limited_rate=c['token_limited']/len(idx),nonrepetitive_nonempty_rate=c['nonrepetitive_nonempty']/len(idx),output_nll=sum(losses[i] for i in idx)/len(idx))
 groups=collections.defaultdict(list)
 for r,o in zip(rows,outputs):
  if r['pair_id']:groups[r['pair_id']].append(o)
 if any(len(g)!=2 for g in groups.values()):raise ValueError('Incomplete canonical question pair')
 both=sum(all(o['checks']['exact'] for o in g) for g in groups.values())
 metrics['paired_requests']=dict(count=len(groups),both_correct=both,accuracy=both/max(1,len(groups)))
 domains=[v['exact'] for k,v in metrics.items() if k.startswith('domain/')]
 metrics['macro_exact']=sum(domains)/len(domains)
 return metrics

def selection(r):
 m=r['metrics'];a=m['all']
 return (m['macro_exact'],a['exact'],a['field_accuracy'],a['f1'],a['schema_valid_rate'],m['paired_requests']['accuracy'],-a['counts']['unsupported_values'],-a['output_nll'])

def evaluate(model,step,path,split='development'):
 import mlx.core as mx
 from sft.data import Corpus,serialize
 from sft.evaluate import generate,loss_sum
 from sml_v1.tokenization import Tokenizer
 tok=Tokenizer(TOK);rows=load(CORPUS/f'{split}.jsonl');c=Corpus(CORPUS,split);outputs=[];losses=[];began=time.monotonic();model.eval()
 try:
  for i,r in enumerate(rows):
   x,y=c.batch([i]);losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
   prompt,_=serialize(tok,r['messages'][:-1],generation=True)
   if len(prompt)+PLAN['max_tokens']>model.cfg.max_seq_len:raise ValueError('Generation would truncate')
   # This is the sole model invocation for the request. No decode/repair/assembler follows it.
   out=generate(model,tok,r['messages'][:-1],limit=PLAN['max_tokens'],temperature=0.)
   outputs.append(dict(id=r['id'],source=r['source'],variant=r['variant'],**out,checks=score(r,out['text'],out['stop_reason'])))
   if (i+1)%50==0 or i+1==len(rows):print(f'[native-json100 eval {step}] {split} {i+1}/{len(rows)} elapsed={time.monotonic()-began:.1f}s',flush=True)
  m=aggregate(rows,outputs,losses);result=dict(step=step,split=split,metrics=m,output_nll=m['all']['output_nll'],outputs=outputs,
    protocol=dict(temperature=0,repetition_penalty=1,max_tokens=PLAN['max_tokens'],serialization='native-sft-chat-v1',prompt_truncation=False,model_calls_per_request=1,constrained_decoding=False,output_repair=False,json_assembly=False,scorer='strict-raw-json100-v1'),
    scope='Raw model-generated JSON, source-span annotation agreement. Duplicate keys, nonfinite JSON, fences, trailing prose and schema violations fail exact correctness. Flat string arrays/nullable strings in seven SNIPS domains; observed research panels, not a new blind or external benchmark.')
  atomic_json(path,result);print('[native-json100 metrics]',json.dumps({k:v for k,v in result.items() if k!='outputs'}),flush=True);return result
 finally:model.train()

def retention(result,baseline):
 """Same frozen raw protocol only; never compare assembler scores as a raw baseline."""
 changes={}
 for name,old in baseline['metrics'].items():
  if name=='all' or name.startswith(('domain/','variant/')):
   now=result['metrics'][name]
   changes[name]=dict(exact_delta=now['exact']-old['exact'],field_accuracy_delta=now['field_accuracy']-old['field_accuracy'],
    false_abstention_delta=now['counts']['false_abstention']-old['counts']['false_abstention'],
    unsupported_values_delta=now['counts']['unsupported_values']-old['counts']['unsupported_values'])
 passed=all(v['exact_delta']>=-1e-12 and v['field_accuracy_delta']>=-1e-12 for v in changes.values())
 return dict(passed=passed,changes=changes,gate='No loss of complete-request or field correctness on any fixed raw-parent domain/schema panel; other error counts reported separately.')
