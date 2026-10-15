"""Frozen-data/scorer and real GPU preflight; no optimizer updates or new weights."""
import json,math,subprocess,sys,time
import numpy as np
from sml_v1.common import atomic_json
from .data import ROOT,CORPUS,REPORT,PARENT,PARENT_SHA,PLAN,TOK,sha,check,load,verify_row
from .train import contract,load_model,order_for,active_workers
from .runtime import accumulate_gradients,objectives
from .evaluate import score,aggregate,retention

def main():
 began=time.monotonic();m=check();deep=json.loads((REPORT/'data-validation.json').read_text())
 if deep['status']!='passed' or deep['files']!=m['files'] or deep['source_hashes']!=m['input_hashes']:raise ValueError('Full data validation missing/stale')
 tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'pretraining/tests'),'-p','test_native_json_10*.py','-v'],capture_output=True,text=True)
 (REPORT/'scorer-tests.txt').write_text(tests.stdout+tests.stderr)
 if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
 if active_workers():raise RuntimeError('Training worker active; not touched')
 from sft.data import Corpus,serialize
 from sml_v1.tokenization import Tokenizer
 import mlx.core as mx
 import mlx.nn as nn
 from mlx.utils import tree_flatten
 from sft.evaluate import loss_sum,generate
 from sml_v1.precision import MasterAdamW
 mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed']);model=load_model(PARENT/'model.safetensors');c=Corpus(CORPUS,'train');batches=order_for(len(c),PLAN)
 if len(batches)!=m['total_steps'] or sorted(np.concatenate(batches).tolist())!=list(range(len(c))):raise ValueError('Not exactly one pass')
 rows=load(CORPUS/'train.jsonl');mix=[r for r in rows if r['source']=='cardinality'];groups={r['base_id'] for r in mix}
 if len(groups)!=256 or len(mix)!=768 or len(rows)!=3072:raise ValueError('Wrong authored/replay mixture')
 chosen=[next(i for i,r in enumerate(rows) if r['source']=='cardinality' and isinstance(r['expected'].get('artist'),list) and len(set(r['expected']['artist']))>1),next(i for i,r in enumerate(rows) if r['source']=='replay' and r['variant']=='nullable'),int(np.argmax(c.lengths))]
 tok=Tokenizer(TOK)
 for i in chosen:
  r=rows[i];verify_row(r);a,b=serialize(tok,r['messages']);start,end=map(int,c.offsets[i:i+2])
  if not np.array_equal(c.ids[start:end],a) or not np.array_equal(c.mask[start:end],b):raise ValueError('Stored masks/targets changed')
 teacher=load_model(PARENT/'model.safetensors');teacher.eval();teacher.freeze();teacher_before=dict(tree_flatten(teacher.parameters()))
 before=dict(tree_flatten(model.parameters()));one=objectives(model,teacher,PLAN['replay_kl_weight']);probe=batches[0];probe=probe[np.argsort(c.lengths[probe],kind='stable')]
 began_batch=time.monotonic();value,g,profile=accumulate_gradients(one,c,probe,policy=PLAN['padding_policy'],microbatch=PLAN['microbatch']);elapsed_batch=time.monotonic()-began_batch
 norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));mx.eval(norm)
 if not math.isfinite(value) or not math.isfinite(norm.item()) or norm.item()<=0:raise ValueError('Invalid gradient')
 del g
 four=np.array([next(i for i in probe if rows[i]['source']=='replay'),next(i for i in probe if rows[i]['source']=='cardinality')]);value4,g,_=accumulate_gradients(one,c,four,policy=PLAN['padding_policy'],microbatch=2);x,y=c.batch(four);whole=float(loss_sum(model,mx.array(x),mx.array(y)).item())/len(four)
 if abs(whole-value4)>.02:raise ValueError('Microbatch loss mismatch')
 del g
 opt=MasterAdamW(learning_rate=PLAN['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=PLAN['weight_decay']);opt.init(model.trainable_parameters());mx.eval(opt.state)
 if int(opt.step.item())!=0:raise ValueError('Optimizer not fresh')
 unchanged=all(bool(mx.array_equal(v,before[k]).item()) for k,v in tree_flatten(model.parameters()))
 teacher_unchanged=all(bool(mx.array_equal(v,teacher_before[k]).item()) for k,v in tree_flatten(teacher.parameters()))
 if not teacher_unchanged or tree_flatten(teacher.trainable_parameters()):raise ValueError('Reference is not frozen')
 if not unchanged or sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Parent changed')
 # DEV smoke only. Raw JSON is scored as generated, never repaired.
 dev=load(CORPUS/'development.jsonl');outputs=[];model.eval()
 for variant in ('canonical','alias','nullable'):
  r=next(r for r in dev if r['source']=='cardinality' and r['variant']==variant);out=generate(model,tok,r['messages'][:-1],limit=PLAN['max_tokens'],temperature=0.)
  outputs.append(dict(id=r['id'],variant=variant,**out,checks=score(r,out['text'],out['stop_reason'])))
 cached_counts={}
 for split in ('development','test'):
  cache=PARENT.parent/('evaluations/development_0000469.json' if split=='development' else 'final-test.json');identity_path=PARENT.parent/('evaluations/development_0000469.identity.json' if split=='development' else 'final-test-identity.json')
  identity=json.loads(identity_path.read_text())
  if identity.get('model_sha256',identity.get('weight_sha256'))!=PARENT_SHA:raise ValueError('Wrong parent cache weight')
  cached={o['id']:o for o in json.loads(cache.read_text())['outputs']};old={r['id']:r for r in load(ROOT/'pretraining/data/sft_100_native_json_20261014'/(split+'.jsonl'))}
  controls=[r for r in load(CORPUS/(split+'.jsonl')) if r['source'].startswith('retention/')]
  for r in controls:
   previous=old[r['id']];out=cached[r['id']]
   if any(r[k]!=previous[k] for k in ['document','messages','schema','expected']):raise ValueError('Cached parent input mismatch')
   if score(r,out['text'],out['stop_reason'])!=out['checks']:raise ValueError('Cached score changed')
  cached_counts[split]=len(controls)
  # Validate inference parity on observed playground cases; no TEST generations.
 import threading
 from json_native_playground import generate_native_raw
 from json_playground import JSONGenerationRequest
 play=load(CORPUS/'playground.jsonl');parity=[]
 for index in (34,40):
  r=play[index];options=JSONGenerationRequest(model_id='sft-105',document=r['document'],schema_text=json.dumps(r['schema']),instruction=r['messages'][0]['content'].split('\n\n',1)[0],max_tokens=256,temperature=0,repetition_penalty=1,constrained=False)
  native=generate_native_raw(model,tok,r['messages'][:-1],options,threading.Event());ordinary=generate(model,tok,r['messages'][:-1],limit=256,temperature=0.)
  if native['text']!=ordinary['text'] or native['stop_reason']!=ordinary['stop_reason']:raise ValueError('Playground generation parity failed')
  parity.append(dict(id=r['id'],exact_text_and_stop=True))
 from .evaluate import evaluate
 baseline=evaluate(model,0,REPORT/'parent-development.json')
 if baseline['metrics']['playground']['counts']['exact']!=42 or baseline['metrics']['category/retention']['counts']['exact']!=200:raise ValueError('Cached parent baseline changed')
 atomic_json(REPORT/'preflight-generations.json',dict(parent_sha256=PARENT_SHA,development_only=True,optimizer_updates=0,outputs=outputs))
 receipt=dict(status='passed',contract=contract(),splits=m['splits'],total_steps=len(batches),authored_training_sources=len(groups),playground_inference_parity=parity,parent_baseline_cached=True,scorer_test_output=str(REPORT/'scorer-tests.txt'),verified_parent_cache_requests=cached_counts,gpu=dict(batch_records=len(probe),batch_seconds=elapsed_batch,output_loss=value,gradient_norm=float(norm.item()),four_record_loss=value4,whole_four_record_loss=whole,accumulation=profile,optimizer_step=0,parent_parameters_unchanged=unchanged,frozen_reference_unchanged=teacher_unchanged,reference_trainable_parameters=0,kl_on_replay_only=True,approximate_96_batch_seconds=elapsed_batch*96),optimizer_updates=0,checkpoints_created=0,test_predictions_created=0,elapsed_seconds=time.monotonic()-began)
 atomic_json(REPORT/'ready.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='contract'},indent=2),flush=True)
if __name__=='__main__':main()
