"""Strict single raw generation. Cached parent outputs require exact input and scorer identity."""
import copy,json,time
from .common import *
from .scoring import score,aggregate
from sml_v1.common import atomic_json

FLOORS=['exact','field_accuracy','precision','recall','f1','present_accuracy','absent_accuracy','json_parse_rate','schema_valid_rate','natural_stop_rate','nonrepetitive_nonempty_rate']
ERRORS=['false_abstention','missed_abstention','unsupported_values','token_limited']

def summary(rows,outputs,losses):
 m=aggregate(rows,outputs,losses)
 for label in sorted({r.get('eval_panel','all') for r in rows}):
  ids=[i for i,r in enumerate(rows) if r.get('eval_panel','all')==label]
  rs=[dict(rows[i],pair_id=None) for i in ids]
  m['panel/'+label]=aggregate(rs,[outputs[i] for i in ids],[losses[i] for i in ids])['all']
 return m

def cached(rows):
 # Freeze and validate the old cache before any reuse. No target/reference enters generation.
 ip=LEGACY.parent/'parent-development.identity.json';identity=json.loads(ip.read_text())
 if identity['parent_sha256']!=PARENT_SHA or identity['development_sha256']!=sha(DATA/'legacy_full.jsonl') or identity['scorer_sha256']!=sha(Path(__file__).with_name('scoring.py')) or identity['generation_sha256']!=sha(ROOT/'pretraining/sft/evaluate.py'):
  # JSONL formatting differs from the archived bytes, so also verify the original full input file below.
  if identity['parent_sha256']!=PARENT_SHA or identity['development_sha256']!=sha(LEGACY/'development.jsonl') or identity['scorer_sha256']!=sha(Path(__file__).with_name('scoring.py')) or identity['generation_sha256']!=sha(ROOT/'pretraining/sft/evaluate.py'):raise ValueError('Parent cache identity changed')
 old_rows={r['id']:r for r in load(LEGACY/'development.jsonl')};saved=json.loads((LEGACY.parent/'parent-development.json').read_text());outputs={o['id']:o for o in saved['outputs']}
 if set(outputs)!=set(old_rows):raise ValueError('Incomplete parent cache')
 expected_protocol=dict(temperature=0,repetition_penalty=1,max_tokens=256,serialization='native-sft-chat-v1',prompt_truncation=False,model_calls_per_request=1,constrained_decoding=False,output_repair=False,json_assembly=False,scorer='strict-raw-json100-v1')
 if saved['protocol']!=expected_protocol:raise ValueError('Parent decoding mismatch')
 # Original 1,444 outputs are also reusable, but only under their frozen receipt.
 if any(r['id'] not in outputs for r in rows):
  path=ROOT/'reports/model-design/SFT-105_cardinality_distillation_20261014/original-test-1444-step64.json'
  receipt=json.loads(path.with_suffix('.identity.json').read_text())
  if receipt['candidate_weight_sha256']!=PARENT_SHA or receipt['test_rows_sha256']!=sha(OLD/'test.jsonl') or receipt['protocol']!=expected_protocol:raise ValueError('Original-test cache changed')
  test=json.loads(path.read_text());original=load(OLD/'test.jsonl')
  if {r['id'] for r in original}!={o['id'] for o in test['outputs']}:raise ValueError('Incomplete original test')
  old_rows.update({r['id']:r for r in original});outputs.update({o['id']:o for o in test['outputs']})
 result={}
 for r in rows:
  if r['id'] not in outputs:continue
  if any(r[k]!=old_rows[r['id']][k] for k in ['document','schema','messages','expected']):raise ValueError('Cached request changed')
  o=copy.deepcopy(outputs[r['id']])
  if score(r,o['text'],o['stop_reason'])!=o['checks']:raise ValueError('Scoring mismatch')
  result[r['id']]=o
 return result

def evaluate(model,rows,path,*,parent=False,step=0,weight_sha256=None):
 import mlx.core as mx
 from .runtime import Encoding,batch,loss
 from sft.evaluate import generate
 weight_sha256=PARENT_SHA if parent else weight_sha256
 if not weight_sha256:raise ValueError('Evaluation requires a model weight identity')
 identity=dict(model_sha256=weight_sha256,requests_sha256=__import__('hashlib').sha256(dumps(rows).encode()).hexdigest(),files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('scoring.py'),ROOT/'pretraining/sft/evaluate.py',DATA/'manifest.json']})
 enc=Encoding();outputs=[];losses=[];cache=cached(rows) if parent else {};began=time.monotonic();model.eval();path=Path(path)
 if path.exists():
  saved=json.loads(path.read_text())
  if saved['identity']!=identity:raise ValueError('Saved evaluation identity changed')
  outputs=saved['outputs']
  if len(outputs)>len(rows):raise ValueError('Extra saved predictions')
  for r,o in zip(rows,outputs):
   if r['id']!=o['id'] or score(r,o['text'],o['stop_reason'])!=o['checks'] or not __import__('math').isfinite(o['output_nll']):raise ValueError('Saved output changed')
  losses=[o['output_nll'] for o in outputs]
 for i in range(len(outputs),len(rows)):
  r=rows[i];x,y=batch([enc.row(r)],[0]);nll=loss(model,mx.array(x),mx.array(y)).item();losses.append(nll)
  if r['id'] in cache:o=cache[r['id']]
  else:
   raw=generate(model,enc.tokenizer,r['messages'][:-1],limit=256,temperature=0.)
   o=dict(id=r['id'],**raw,checks=score(r,raw['text'],raw['stop_reason']))
  outputs.append(dict(o,output_nll=nll))
  if (i+1)%50==0 or i+1==len(rows):
   atomic_json(path,dict(status='evaluating',identity=identity,outputs=outputs));print(f'[json128 eval {step}] {path.stem} {i+1}/{len(rows)} elapsed={time.monotonic()-began:.1f}s',flush=True)
 result=dict(status='complete',identity=identity,step=step,metrics=summary(rows,outputs,losses),outputs=outputs,cached_requests=len(cache),parent_sha256=PARENT_SHA if parent else None,requests_sha256=identity['requests_sha256'],protocol=dict(temperature=0,repetition_penalty=1,max_tokens=256,one_generation_per_request=True,prompt_truncation=False,json_assembly=False,output_repair=False,constrained_decoding=False))
 atomic_json(path,result);print('[json128 metrics]',dumps(dict(step=step,split=path.stem,metrics=result['metrics'])),flush=True);model.train();return result

def compare(candidate,parent,*,compact=False,rows=None):
 before={o['id']:o for o in parent['outputs']};after={o['id']:o for o in candidate['outputs']}
 if before.keys()!=after.keys() or parent['requests_sha256']!=candidate['requests_sha256']:raise ValueError('Comparison coverage mismatch')
 rows=rows if rows is not None else load(DATA/('development.jsonl' if compact else 'legacy_full.jsonl'))
 protected=[r['id'] for r in rows if not compact or r.get('eval_panel')=='legacy']
 lost=[i for i in protected if before[i]['checks']['exact'] and not after[i]['checks']['exact']]
 gained=[i for i in before if not before[i]['checks']['exact'] and after[i]['checks']['exact']]
 # Protect each old domain and schema variant, not just a pooled score.
 panels={}
 for key,old in parent['metrics'].items():
  if compact:
   idx=[i for i,r in enumerate(rows) if r.get('eval_panel')=='legacy' and (key=='all' or key=='domain/'+r['source'] or key=='variant/'+r['variant'])]
   if not idx:continue
   rs=[dict(rows[i],pair_id=None) for i in idx];bs=[parent['outputs'][i] for i in idx];cs=[candidate['outputs'][i] for i in idx]
   old=aggregate(rs,bs,[o['output_nll'] for o in bs])['all'];new=aggregate(rs,cs,[o['output_nll'] for o in cs])['all']
  else:
   if key not in ['all'] and not key.startswith(('domain/','variant/')):continue
   new=candidate['metrics'][key]
  deltas={f:new[f]-old[f] for f in FLOORS};errors={f:new['counts'][f]-old['counts'][f] for f in ERRORS}
  panels[key]=dict(metric_deltas=deltas,error_deltas=errors,passed=all(v>=-1e-12 for v in deltas.values()) and all(v<=0 for v in errors.values()))
 paired=candidate['metrics']['paired_requests']['both_correct']-parent['metrics']['paired_requests']['both_correct']
 new_gain=(candidate['metrics']['panel/new_prose']['counts']['exact']-parent['metrics']['panel/new_prose']['counts']['exact']) if compact else None
 return dict(passed=not lost and all(p['passed'] for p in panels.values()) and paired>=0 and (new_gain is None or new_gain>0),lost_previously_correct=lost,gained=gained,new_prose_correct_delta=new_gain,paired_correct_delta=paired,panels=panels,scope='Finite observed request/metric retention gates; cannot guarantee universal no-regression.')
