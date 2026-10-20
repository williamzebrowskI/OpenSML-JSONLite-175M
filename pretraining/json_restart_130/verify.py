"""Setup-only source audit, release baseline, restored eight-update base probe."""
import json,math,time
from .common import *
from .runtime import model_at,Encoding,order_for,gradients,loss,batch,lr,generate
from .train import contract,published_cache
from .evaluate import evaluate

def main():
 import mlx.core as mx
 from mlx.utils import tree_flatten,tree_map
 from sml_v1.precision import MasterAdamW
 from sml_v1.common import lock
 if RUN.exists() or REFINE.exists():raise ValueError('Preflight may not initialize or modify live runs')
 if not (REPORT/'tests.txt').exists() or '\nOK\n' not in (REPORT/'tests.txt').read_text():raise ValueError('Tests must pass first')
 c=contract()
 with lock(LOCK):
  from json_sampling_119.run import active_workers
  if active_workers():raise RuntimeError('Another training worker is active')
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
  enc=Encoding();rows=load(DATA/'train.jsonl');encoded=[enc.row(r,training=True) for r in rows];batches=order_for(encoded,PLAN['seed'],64)
  # Recheck each original source against the previously independently audited SNIPS annotation bank.
  from json_sampling_119.data import source_audit
  old=load(ORIGINAL/'train.jsonl');audit=source_audit([dict(r,source_bank=str((ORIGINAL/'train.jsonl').relative_to(ROOT)),source_row_id=r['id']) for r in old])
  atomic_json(REPORT/'source-validation.json',audit)
  model=model_at();source=dict(tree_flatten(model.parameters()));mx.eval(source)
  fit=[i for b in batches[:8] for i in b][::16]
  def fit_nll():
   vals=[];model.eval()
   for i in fit:
    x,y=batch(encoded,[i]);vals.append(loss(model,mx.array(x),mx.array(y)).item())
   return sum(vals)/len(vals)
  before=fit_nll();model.unfreeze();model.train();opt=MasterAdamW(learning_rate=PLAN['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=0.);opt.init(model.trainable_parameters());mx.eval(opt.state)
  trace=[];start=time.monotonic()
  for step in range(1,9):
   value,g,n=gradients(model,encoded,batches[step-1]);norm=mx.sqrt(sum(mx.sum(a*a) for _,a in tree_flatten(g)));mx.eval(norm)
   if not math.isfinite(value) or not math.isfinite(norm.item()):raise ValueError('Nonfinite base probe')
   opt.learning_rate=lr(step,500,PLAN);g=tree_map(lambda a:a*mx.minimum(1.,1/(norm+1e-6)),g);opt.update(model,g);mx.eval(model.parameters(),opt.state)
   trace.append(dict(step=step,loss=value,lr=lr(step,500,PLAN),gradient_norm=norm.item(),assistant_tokens=n));print('[joint130 TRAIN probe] '+dumps(trace[-1]),flush=True)
  seconds=time.monotonic()-start;after=fit_nll();changed=sum(not mx.array_equal(v,source[k]).item() for k,v in tree_flatten(model.parameters()))
  del model,opt;mx.clear_cache();restored=model_at();identical=all(mx.array_equal(v,source[k]).item() for k,v in tree_flatten(restored.parameters()));del restored,source;mx.clear_cache()
  if not(identical and changed>0 and after<before):raise ValueError('Restored TRAIN numerical viability failed')
  probe=dict(status='passed',contract=c,train_only=True,updates=8,records=512,fit_count=len(fit),fit_output_nll_before=before,fit_output_nll_after=after,changed_tensors=changed,base_restored_exactly=identical,seconds=seconds,trace=trace,
    scope='TRAIN-only numerical and learning viability, not held-out quality or an optimized-LR proof; full run starts from base with fresh optimizer.')
  atomic_json(REPORT/'train-probe.json',probe)
  # Released baseline only. No slow raw-base baseline and no intermediate parent reruns.
  released=model_at(RELEASE);dev=load(DATA/'development.jsonl')
  # Exact serving parity on cached examples before reusing release metrics.
  _,original_cache=published_cache('original1444');assert original_cache['metrics']['all']['counts']['exact']==1271
  pg,cache=published_cache('playground60');assert cache['metrics']['all']['counts']['exact']==49
  byid={o['id']:o for o in cache['outputs']};parity=[]
  for r in pg[:4]:
   raw=generate(released,enc.tokenizer,enc.prefix(r['document'],r['schema']));o=byid[r['id']]
   if (raw['text'],raw['stop_reason'])!=(o['text'],o['stop_reason']):raise ValueError('Release generation parity failed')
   parity.append(r['id'])
  evaluate(released,dev,REPORT/'released-development.json',RELEASE_SHA)
  del released;mx.clear_cache()
  assert contract()==c and sha(BASE)==BASE_SHA and sha(RELEASE)==RELEASE_SHA
  atomic_json(REPORT/'setup-verification.json',dict(status='passed',contract=c,release_cache_live_parity=parity,base_unchanged=True,release_unchanged=True,full_training_started=False,reserved_predictions=0,
   note='Refinement policy and exact resume tested; its actual parent and targeted mixture are frozen only after foundation development selection. No inference adapter/repair/teacher.'))
  receipts=[REPORT/n for n in ['tests.txt','train-probe.json','released-development.json','source-validation.json','setup-verification.json']]
  atomic_json(REPORT/'ready.json',dict(status='passed',contract=c,receipts={str(p.relative_to(ROOT)):sha(p) for p in receipts},full_training_started=False))
  print('[joint130 ready] '+dumps(dict(command='bash pretraining/scripts/run_sft_130_joint_foundation.sh --run',steps=500,eval_steps=[200,400,500],base_restored=True,full_training_started=False)),flush=True)
if __name__=='__main__':main()
