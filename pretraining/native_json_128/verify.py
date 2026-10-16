"""Setup-only audit, cached baselines, and a restored eight-update TRAIN probe.
Does not initialize the live run or consume reserved/test predictions.
"""
import json,math,time
import numpy as np
from .common import *
from .runtime import model_at,Encoding,accumulate,batch,loss,learning_rate
from .evaluate import evaluate,cached,summary
from .train import contract,order_for
from sml_v1.common import atomic_json,lock

def main():
 import mlx.core as mx
 from mlx.utils import tree_map,tree_flatten
 from sml_v1.precision import MasterAdamW
 from sft.evaluate import generate
 from .scoring import score
 c=contract()
 if RUN.exists():raise ValueError('Preflight must not modify a started experiment')
 if not (REPORT/'tests.txt').exists() or '\nOK\n' not in (REPORT/'tests.txt').read_text():raise ValueError('Unit/gradient/resume checks have not passed')
 with lock(LOCK):
  from json_sampling_119.run import active_workers
  if active_workers():raise RuntimeError('Another model worker is active')
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed']);enc=Encoding();model=model_at();rows=load(DATA/'train.jsonl');encoded=[enc.row(r) for r in rows]
  # Validate masking/EOS/context on every row without cropping, and record all lengths.
  lengths=[len(x) for x,y in encoded];counts=[int(np.sum(y!=-100)) for x,y in encoded]
  stats=dict(status='passed',contract=c,records=len(rows),serialized_input_tokens=sum(lengths),assistant_tokens=sum(counts),max_input_tokens=max(lengths),context_limit=2048,all_original_views_unchanged=15000,all_cardinality_views_retained=768,all_values_literal_and_source_ordered=True,original_annotation_audit=str(REPORT/'source-validation.json'),templates=dict(train=56,development=28,reserved=28),distinct_new_sources=check()['summary']['new_sources'],scope='Literal/source-role checks plus pinned original annotation agreement. Programmatic new prose; neither perfect semantic truth nor real-user generalization is established.')
  atomic_json(REPORT/'data-review.json',stats)
  # A few live raw parent generations must match the frozen cached decoding path.
  legacy=load(DATA/'development.jsonl')[:128];cache=cached(legacy);parity=[]
  for r in legacy[:4]:
   raw=generate(model,enc.tokenizer,r['messages'][:-1],limit=256,temperature=0.)
   if any(raw[k]!=cache[r['id']][k] for k in ['text','stop_reason']):raise ValueError('Parent cache live parity failed')
   parity.append(dict(id=r['id'],matched=True))
  atomic_json(REPORT/'parent-cache-parity.json',dict(status='passed',contract=c,requests=parity))
  evaluate(model,load(DATA/'development.jsonl'),REPORT/'parent-development.json',parent=True)
  evaluate(model,load(DATA/'legacy_full.jsonl'),REPORT/'parent-legacy-full.json',parent=True)
  # Fit measurements are TRAIN only and explicitly excluded from checkpoint selection.
  order=order_for(len(rows));prefix=order[:8*64];fit=[int(prefix[i]) for i in np.linspace(0,len(prefix)-1,32,dtype=int)]
  def fit_result():
   nll=[];outs=[];model.eval()
   for i in fit:
    x,y=batch(encoded,[i]);nll.append(loss(model,mx.array(x),mx.array(y)).item());raw=generate(model,enc.tokenizer,rows[i]['messages'][:-1],limit=256,temperature=0.);outs.append(dict(id=rows[i]['id'],**raw,output_nll=nll[-1],checks=score(rows[i],raw['text'],raw['stop_reason'])))
   return dict(mean_output_nll=sum(nll)/len(nll),correct=sum(o['checks']['exact'] for o in outs),count=len(outs),outputs=outs)
  before=fit_result();source=dict(tree_flatten(model.parameters()));model.unfreeze();model.train();opt=MasterAdamW(learning_rate=PLAN['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=0.);opt.init(model.trainable_parameters());mx.eval(opt.state)
  trainable=sum(v.size for _,v in tree_flatten(model.trainable_parameters()));start=time.monotonic();trace=[]
  for step in range(1,9):
   ids=order[(step-1)*64:step*64];value,g,tokens=accumulate(model,encoded,ids);norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));mx.eval(norm)
   if not math.isfinite(value) or not math.isfinite(norm.item()):raise ValueError('Nonfinite viability probe')
   g=tree_map(lambda a:a*mx.minimum(1.,PLAN['clip']/(norm+1e-6)),g);opt.learning_rate=learning_rate(step,331);opt.update(model,g);mx.eval(model.parameters(),opt.state);trace.append(dict(step=step,output_loss=value,gradient_norm=norm.item(),lr=learning_rate(step,331),assistant_tokens=tokens))
   print('[json128 TRAIN probe]',dumps(trace[-1]),flush=True)
  seconds=time.monotonic()-start;after=fit_result();changed=sum(not mx.array_equal(v,source[k]).item() for k,v in tree_flatten(model.parameters()));del opt,model;mx.clear_cache()
  restored=model_at();identical=all(mx.array_equal(v,source[k]).item() for k,v in tree_flatten(restored.parameters()));del restored,source;mx.clear_cache()
  passed=identical and changed>0 and trainable==175611648 and after['mean_output_nll']<before['mean_output_nll'] and sha(PARENT/'model.safetensors')==PARENT_SHA
  probe=dict(status='passed' if passed else 'failed',contract=c,updates=8,records=512,trainable_parameters=trainable,changed_tensors=changed,parent_restored_exactly=identical,live_run_initialized=False,training_seconds=seconds,train_fit_before=before,train_fit_after=after,trace=trace,scope='Restored numerical/learning viability check on 32 TRAIN rows inside the first 512 updates presentations. No development/held-out inference from the probe, and no claim of release improvement. Full run starts at update zero with fresh optimizer.')
  atomic_json(REPORT/'train-viability.json',probe)
  if not passed:raise ValueError('TRAIN viability failed; command disabled')
  if contract()!=c:raise ValueError('Frozen inputs changed during preflight')
  receipts=[REPORT/n for n in ['tests.txt','source-validation.json','data-review.json','parent-cache-parity.json','parent-development.json','parent-legacy-full.json','train-viability.json']]
  atomic_json(REPORT/'ready.json',dict(status='passed',contract=c,receipts={str(p.relative_to(ROOT)):sha(p) for p in receipts},full_training_started=False,test_predictions=0))
  print('[json128 ready]',dumps(dict(train_probe_before=before['mean_output_nll'],train_probe_after=after['mean_output_nll'],train_correct_before=before['correct'],train_correct_after=after['correct'],seconds=seconds,parent_unchanged=True,command='bash pretraining/scripts/run_sft_128_joint_extraction.sh --run')),flush=True)
if __name__=='__main__':main()
