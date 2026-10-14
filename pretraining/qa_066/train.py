"""Explicit manual Mac1 launch: one pass over 8k single-turn grounded QA pairs."""
import argparse,importlib.metadata,json,math,os,signal,subprocess,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from sml_v1.common import atomic_json,fingerprint,lock
from json_sft.corpus import ROOT,file_sha
from .data import CORPUS,OUTPUT,PARENT,PARENT_SHA,PLAN,TOK,check
from .evaluate import evaluate,selection

def contract():
 m=check(True)
 if file_sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Pretrained parent changed')
 meta=json.loads((PARENT/'model.safetensors.json').read_text())
 if meta.get('step')!=469:raise ValueError('Requires preserved SFT-065 update469')
 paths=[]
 for name in ['qa_066','qa_065','qa_064','qa_semantics','extraction_001','sml_v1','json_sft','json_specialist','json_evidence_binding']:
  paths+=list((ROOT/'pretraining'/name).glob('*.py'))
 paths += [ROOT/'pretraining/runs/sft_064_grounded_qa_20261013/evaluations/development_0000469.json',ROOT/'pretraining/runs/sft_064_grounded_qa_20261013/final-test.json']
 paths += [ROOT/'pretraining/runs/sft_065_targeted_qa_20261013/evaluations/development_0000469.json',ROOT/'pretraining/runs/sft_065_targeted_qa_20261013/evaluations/development_0000469.identity.json',ROOT/'pretraining/runs/sft_065_targeted_qa_20261013/final-test.json',ROOT/'pretraining/runs/sft_065_targeted_qa_20261013/final-test-identity.json']
 paths += [ROOT/'pretraining/sft/data.py',ROOT/'pretraining/scripts/run_sft_066_prose_repair.sh']
 paths += list(TOK.glob('*'))
 paths += [ROOT/'pretraining/sft/evaluate.py',ROOT/'pretraining/sft/quality.py',ROOT/'pretraining/qa_066/evaluate.py']
 return dict(plan=PLAN,manifest=file_sha(CORPUS/'manifest.json'),parent=str(PARENT),parent_sha256=PARENT_SHA,parent_metadata=file_sha(PARENT/'model.safetensors.json'),code={str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths)},versions={n:importlib.metadata.version(n) for n in ('mlx','numpy','tokenizers','jsonschema')})
def load_model(path):
 import mlx.core as mx
 from mlx.utils import tree_map
 from sml_v1.model import TransformerLM,TransformerConfig
 m=json.loads(Path(str(path)+'.json').read_text());model=TransformerLM(TransformerConfig(**m['recipe']['model']));model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()));model.load_weights(str(path));mx.eval(model.parameters());return model

def order_for(n,plan):
 batches=[]
 for epoch in range(plan.get('epochs',1)):
  order=np.random.default_rng(plan['seed']+epoch).permutation(n)
  batches.extend(order[i:i+plan['batch_size']] for i in range(0,n,plan['batch_size']))
 return batches
def lr_for(step,total,plan):
 warm=min(plan['warmup_updates'],total)
 if step<=warm:return plan['peak_lr']*step/warm
 return plan['final_lr']+(plan['peak_lr']-plan['final_lr'])*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))/2

def execute(model,corpus,output,plan,c,evaluator,resume=False,stop_after=None):
 import mlx.core as mx
 import mlx.nn as nn
 from mlx.utils import tree_flatten,tree_map,tree_unflatten
 from sml_v1.precision import MasterAdamW
 from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
 from .runtime import accumulate_gradients
 from json_sft.evaluate import loss_sum
 output=Path(output);order=order_for(len(corpus),plan);total=len(order);oh=fingerprint([b.tolist() for b in order]);step=0;tokens=0;records=0
 opt=MasterAdamW(learning_rate=plan['peak_lr'],betas=tuple(plan['betas']),eps=plan['eps'],weight_decay=plan['weight_decay']);opt.init(model.trainable_parameters())
 if resume:
  # Recover a committed checkpoint if the latest pointer was interrupted.
  if not (output/'latest.json').exists():
   bundles=sorted(output.glob('step_*_*'),key=lambda p:json.loads((p/'manifest.json').read_text())['created_ns'])
   if not bundles:raise ValueError('No checkpoint to resume')
   atomic_json(output/'latest.json',dict(format='sml-pretrain-bundle-v1',bundle=bundles[-1].name))
  w=resolve_bundle(output/'latest.json');meta=json.loads(Path(w+'.json').read_text());state=meta['extraction_sft']
  if state['contract']!=c or state['order_hash']!=oh:raise ValueError('Resume contract/order changed')
  step=meta['step'];tokens=meta['tokens'];records=state['records'];model.load_weights(w);opt.state=tree_unflatten(list(mx.load(w+'.optimizer.safetensors').items()))
 mx.eval(model.parameters(),opt.state)
 if int(opt.step.item())!=step:raise ValueError('Optimizer step mismatch')
 assert records==sum(len(b) for b in order[:step]) and tokens==sum(int(sum(corpus.lengths[b]+1)) for b in order[:step])
 model.train();one=nn.value_and_grad(model,lambda x,y:loss_sum(model,x,y));stop=[False]
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
 def save():
  meta=dict(step=step,tokens=tokens,recipe={'model':asdict(model.cfg)},extraction_sft=dict(contract=c,order_hash=oh,records=records,total_steps=total,serialization='native-sft-chat-v1'))
  return Path(save_bundle(output,model,opt,meta,{'next_batch':step},keep=6,reserve_gib=10))
 def assess(bundle):
  path=output/'evaluations'/f'development_{step:07d}.json';identity=dict(contract=c,step=step,model_sha256=file_sha(bundle/'model.safetensors'))
  ip=path.with_suffix('.identity.json')
  if path.exists():
   if json.loads(ip.read_text())!=identity:raise ValueError('Evaluation identity changed')
   result=json.loads(path.read_text())
  else:
   atomic_json(ip,identity);result=evaluator(model,step,path)
  best=output/'best-metrics.json';prior=json.loads(best.read_text()) if best.exists() else None
  if prior is None or selection(result)>tuple(prior['selection']):
   atomic_json(best,dict(step=step,selection=list(selection(result)),bundle=bundle.name))
   atomic_json(output/'best.json',dict(format='sml-pretrain-bundle-v1',bundle=bundle.name))
  atomic_json(output/'last_eval.json',{k:v for k,v in result.items() if k!='outputs'})
  return result
 if not resume:bundle=save()
 else:bundle=Path(resolve_bundle(output/'latest.json')).parent
 if step in plan['eval_steps'] or step==total:assess(bundle)
 began=time.monotonic();print(f'[ready-qa066] local Mac1 {step}/{total}; one pass over 8k grounded QA examples; output-only loss; full-weight; batch={plan["batch_size"]}',flush=True)
 while step<total:
  if stop[0] or (output/'STOP').exists() or (stop_after is not None and step>=stop_after):
   save();atomic_json(output/'result.json',dict(status='stopped',step=step,total_steps=total));return
  value,g,_=accumulate_gradients(one,corpus,order[step],policy=plan['padding_policy'],microbatch=plan['microbatch']);norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));mx.eval(norm)
  if not math.isfinite(value) or not math.isfinite(norm.item()):raise RuntimeError('Nonfinite update')
  g=tree_map(lambda v:v*mx.minimum(1.,plan['grad_clip']/(norm+1e-6)),g);opt.learning_rate=lr_for(step+1,total,plan);opt.update(model,g);mx.eval(model.parameters(),opt.state)
  tokens+=int(sum(corpus.lengths[order[step]]+1));records+=len(order[step]);step+=1
  if step==1 or step%10==0:print(f'[qa066 {step}/{total}] loss={value:.5f} lr={lr_for(step,total,plan):.3e} grad_norm={norm.item():.3f} records={records} tokens={tokens} elapsed={time.monotonic()-began:.1f}s',flush=True)
  if step in plan['eval_steps'] or step==total:assess(save())
 atomic_json(output/'result.json',dict(status='sft_complete',step=step,total_steps=total,records=records,tokens=tokens,release_ready=False))

def active_workers():
 import re
 lines=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines()
 return [s for s in lines if int(s.split(None,1)[0])!=os.getpid() and re.search(r'(?:python\S*\s+(?:-u\s+)?-m\s+(?:extraction_\w+|chat_entities(?:_\d+)?|qa_\w+|json_\w+|sft|sml_v1)\.(?:train|pretrain)|python\S*\s+(?:-u\s+)?\S*/(?:launch_\w+|train_\w+)\.py)',s)]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.resume and not a.run:p.error('--resume requires --run')
 c=contract()
 if not a.run:print(json.dumps(dict(training_started=False,parent=str(PARENT),plan=PLAN,data=check()['splits']),indent=2));return
 with lock(ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'):
  if active_workers():raise RuntimeError('Another training worker is active')
  OUTPUT.mkdir(parents=True,exist_ok=True);session=OUTPUT/'session.json'
  if session.exists():
   if not a.resume:raise ValueError('Run initialized; use --run --resume')
   if json.loads(session.read_text())!=c:raise ValueError('Frozen contract changed')
  else:atomic_json(session,c)
  import mlx.core as mx
  from sft.data import Corpus
  from sml_v1.checkpoint_bundle import resolve_bundle
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
  model=load_model(PARENT/'model.safetensors');restore=a.resume and bool(list(OUTPUT.glob('step_*_*')))
  from .data import load
  runtime_plan=dict(PLAN)
  execute(model,Corpus(CORPUS,'train'),OUTPUT,runtime_plan,c,evaluate,resume=restore)
  state=json.loads((OUTPUT/'result.json').read_text())
  if state['status']=='sft_complete':
   del model;mx.clear_cache();best=Path(resolve_bundle(OUTPUT/'best.json'));model=load_model(best);test=OUTPUT/'final-test.json'
   if not test.exists():
    atomic_json(OUTPUT/'final-test-identity.json',dict(contract=c,weight_sha256=file_sha(best),selected_step=json.loads((OUTPUT/'best-metrics.json').read_text())['step']))
    evaluate(model,json.loads((OUTPUT/'best-metrics.json').read_text())['step'],test,split='test')
   state.update(status='complete',selected=json.loads((OUTPUT/'best-metrics.json').read_text()),final_test=str(test));atomic_json(OUTPUT/'result.json',state)
  if file_sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Parent changed')
  print(json.dumps(state,indent=2))
if __name__=='__main__':main()
