"""Joint whole-object JSON SFT; user launches the actual run after verified setup."""
import argparse,importlib.metadata,json,math,signal,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .common import *
from .runtime import model_at,Encoding,accumulate,learning_rate
from .evaluate import evaluate,compare
from sml_v1.common import atomic_json,fingerprint,lock


def contract():
 m=check();paths=list(Path(__file__).parent.glob('*.py'))+list(TOK.glob('*'))
 paths += [ROOT/'pretraining'/n for n in ['tests/test_native_json_128.py','scripts/run_sft_128_joint_extraction.sh','sft/evaluate.py','sft/data.py','sft/quality.py','sml_v1/model.py','sml_v1/fast_loss.py','sml_v1/fast_swiglu.py','sml_v1/tokenization.py','sml_v1/precision.py','sml_v1/checkpoint_bundle.py','sml_v1/common.py','json_sampling_119/common.py','json_sampling_119/data.py','json_sampling_119/run.py']]
 paths += [DATA/'manifest.json',PARENT/'model.safetensors.json',REPORT/'source-validation.json']
 paths += [ROOT/'reports/model-design/SFT-105_cardinality_distillation_20261014'/n for n in ['original-test-1444-step64.json','original-test-1444-step64.identity.json']]
 if any(not p.exists() for p in paths):raise ValueError('Missing frozen source')
 return dict(plan=PLAN,parent_sha256=PARENT_SHA,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.is_file()},versions={n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','numpy','tokenizers','jsonschema']})

def schedule(total):return sorted(set(range(PLAN['eval_every'],total+1,PLAN['eval_every']))|{total})
def order_for(n):return np.random.default_rng(PLAN['seed']).permutation(n)

def execute(model,encoded,output,c,*,resume=False,stop_after=None,assessor=None):
 import mlx.core as mx
 from mlx.utils import tree_map,tree_flatten,tree_unflatten
 from sml_v1.precision import MasterAdamW
 from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
 output=Path(output);output.mkdir(parents=True,exist_ok=True);order=order_for(len(encoded));order_hash=fingerprint(order.tolist());total=math.ceil(len(order)/64);step=records=tokens=0
 model.unfreeze();model.train();opt=MasterAdamW(learning_rate=PLAN['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=PLAN['weight_decay']);opt.init(model.trainable_parameters())
 if resume:
  path=resolve_bundle(output/'latest.json');meta=json.loads(Path(path+'.json').read_text());state=meta['joint128']
  if state['contract']!=c or state['order_hash']!=order_hash:raise ValueError('Resume identity/order changed')
  model.load_weights(path);opt.state=tree_unflatten(list(mx.load(path+'.optimizer.safetensors').items()));step=meta['step'];records=state['records'];tokens=meta['tokens']
 mx.eval(model.parameters(),opt.state)
 if int(opt.step.item())!=step or records!=min(step*64,len(order)):raise ValueError('Optimizer/cursor mismatch')
 stop=[False];handlers={};saved_step=step if resume else -1
 for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,lambda *_:stop.__setitem__(0,True))
 def save():
  nonlocal saved_step
  meta=dict(step=step,tokens=tokens,recipe=dict(model=asdict(model.cfg)),joint128=dict(contract=c,order_hash=order_hash,records=records,total=total))
  p=Path(save_bundle(output,model,opt,meta,dict(next_batch=step),keep=2,reserve_gib=10));saved_step=step;return p
 def assess(bundle):
  if assessor:return assessor(model,step,bundle)
  baseline=json.loads((REPORT/'parent-development.json').read_text());path=output/'evaluations'/f'development_{step:07d}.json';ip=path.with_suffix('.identity.json');identity=dict(contract=c,model_sha256=sha(bundle/'model.safetensors'))
  if ip.exists() and json.loads(ip.read_text())!=identity:raise ValueError('Evaluation identity changed')
  result=evaluate(model,load(DATA/'development.jsonl'),path,step=step,weight_sha256=identity['model_sha256']);atomic_json(ip,identity)
  gate=compare(result,baseline,compact=True);result['retention_vs_parent']=gate;atomic_json(path,result)
  selected=[result['metrics']['panel/new_prose']['counts']['exact'],result['metrics']['all']['field_accuracy'],result['metrics']['all']['f1']]
  prior=json.loads((output/'best-metrics.json').read_text()) if (output/'best-metrics.json').exists() else None
  if gate['passed'] and (prior is None or selected>prior['selection']):
   atomic_json(output/'best.json',dict(format='sml-pretrain-bundle-v1',bundle=bundle.name));atomic_json(output/'best-metrics.json',dict(step=step,bundle=bundle.name,selection=selected))
  atomic_json(output/'last_eval.json',{k:v for k,v in result.items() if k!='outputs'});print('[json128 retention]',dumps(dict(step=step,passed=gate['passed'],lost=len(gate['lost_previously_correct']),new_correct_delta=gate['new_prose_correct_delta'])),flush=True)
 try:
  if resume and step in schedule(total):assess(Path(resolve_bundle(output/'latest.json')).parent)
  began=time.monotonic();print(f'[ready-json128] Mac1 {step}/{total}; full-weight; batch64/micro2; one joint pass; compact eval240 at {schedule(total)}',flush=True)
  while step<total:
   if stop[0] or (output/'STOP').exists() or (stop_after is not None and step>=stop_after):
    if step!=saved_step:save()
    atomic_json(output/'result.json',dict(status='stopped',step=step,total=total,records=records,tokens=tokens));return
   ids=order[step*64:(step+1)*64];v,g,n=accumulate(model,encoded,ids);norm=mx.sqrt(sum(mx.sum(a*a) for _,a in tree_flatten(g)));mx.eval(norm)
   if not math.isfinite(v) or not math.isfinite(norm.item()):raise ValueError('Nonfinite training loss/gradient')
   g=tree_map(lambda a:a*mx.minimum(1.,PLAN['clip']/(norm+1e-6)),g);opt.learning_rate=learning_rate(step+1,total);opt.update(model,g);mx.eval(model.parameters(),opt.state);step+=1;records+=len(ids);tokens+=n
   atomic_json(output/'progress.json',dict(step=step,total=total,records=records,assistant_tokens=tokens,output_loss=v,gradient_norm=norm.item(),lr=learning_rate(step,total),elapsed_seconds=time.monotonic()-began))
   if step==1 or step%10==0:print(f'[json128 {step}/{total}] output_loss={v:.5f} lr={learning_rate(step,total):.3e} grad_norm={norm.item():.3f} records={records} elapsed={time.monotonic()-began:.1f}s',flush=True)
   if step in schedule(total):assess(save())
  atomic_json(output/'result.json',dict(status='sft_complete',step=step,total=total,records=records,tokens=tokens,release_ready=False))
 finally:
  for sig,h in handlers.items():signal.signal(sig,h)


def finish(model,c):
 import mlx.core as mx
 from sml_v1.checkpoint_bundle import resolve_bundle
 state=json.loads((RUN/'result.json').read_text());best=RUN/'best-metrics.json'
 if not best.exists():state.update(status='complete',selected=None,final_comparison='skipped_no_compact_qualified_candidate',automatic_promotion=False)
 else:
  selected=json.loads(best.read_text());weight=Path(resolve_bundle(RUN/'best.json'));model=model_at(weight)
  # Only one compact-selected candidate sees the full legacy gate.
  candidate=evaluate(model,load(DATA/'legacy_full.jsonl'),RUN/'legacy-full.json',step=selected['step'],weight_sha256=sha(weight));baseline=json.loads((REPORT/'parent-legacy-full.json').read_text());gate=compare(candidate,baseline)
  state.update(selected=selected,full_legacy_gate=gate,automatic_promotion=False)
  if not gate['passed']:state.update(status='complete',final_comparison='skipped_legacy_retention_failed')
  else:
   tests={}
   for name in ['original_test','reserved']:
    rows=load(DATA/(name+'.jsonl'));candidate=evaluate(model,rows,RUN/(name+'-candidate.json'),step=selected['step'],weight_sha256=sha(weight));del model;mx.clear_cache();parent=model_at();baseline=evaluate(parent,rows,RUN/(name+'-parent.json'),parent=True);del parent;mx.clear_cache();model=model_at(weight)
    tests[name]=dict(parent_metrics=baseline['metrics'],candidate_metrics=candidate['metrics'],retention=compare(candidate,baseline,rows=rows))
   atomic_json(RUN/'comparison.json',dict(contract=c,candidate_sha256=sha(weight),tests=tests,automatic_promotion=False,release_ready=False));state.update(status='complete',final_comparison='complete',research_gate_passed=all(v['retention']['passed'] for v in tests.values()))
 atomic_json(RUN/'result.json',state);print(dumps(state),flush=True)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 c=contract()
 if not a.run:print(dumps(dict(training_started=False,parent=str(PARENT),plan=PLAN,data=check()['summary'])));return
 ready=json.loads((REPORT/'ready.json').read_text())
 if ready['status']!='passed' or ready['contract']!=c:raise ValueError('Preflight is missing/stale; refusing training')
 if any(sha(ROOT/n)!=h for n,h in ready['receipts'].items()):raise ValueError('Preflight receipt changed')
 with lock(LOCK):
  from json_sampling_119.run import active_workers
  if active_workers():raise RuntimeError('Another training worker is active')
  session=RUN/'session.json'
  if session.exists() and json.loads(session.read_text())!=c:raise ValueError('Session changed')
  if (RUN/'result.json').exists() and json.loads((RUN/'result.json').read_text())['status']=='complete':print((RUN/'result.json').read_text());return
  has=(RUN/'latest.json').exists()
  if has and not a.resume:raise ValueError('Saved optimizer progress exists; use --run --resume')
  if a.resume and not has:print('[json128] No committed update exists; starting from the preserved parent.',flush=True)
  RUN.mkdir(parents=True,exist_ok=True);atomic_json(session,c)
  import mlx.core as mx
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed']);enc=Encoding();encoded=[enc.row(r) for r in load(DATA/'train.jsonl')];model=model_at();execute(model,encoded,RUN,c,resume=has)
  state=json.loads((RUN/'result.json').read_text())
  if state['status']=='sft_complete':finish(model,c)
  if sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Parent changed')
if __name__=='__main__':main()
