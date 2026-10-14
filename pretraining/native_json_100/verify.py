"""Dataset/scorer and local GPU preflight. No optimizer updates/checkpoints."""
import json,math,subprocess,sys,time
import numpy as np
from sml_v1.common import atomic_json
from .data import ROOT,CORPUS,REPORT,PARENT,PARENT_SHA,PLAN,TOK,sha,check,load
from .train import contract,load_model,order_for,active_workers
from .runtime import accumulate_gradients
from .evaluate import score

def main():
 started=time.monotonic();m=check(deep=True)
 tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'pretraining/tests'),'-p','test_native_json_100.py','-v'],capture_output=True,text=True)
 (REPORT/'scorer-tests.txt').write_text(tests.stdout+tests.stderr)
 if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
 if active_workers():raise RuntimeError('Another training worker is active; preflight left it untouched')
 import mlx.core as mx
 import mlx.nn as nn
 from mlx.utils import tree_flatten
 from sml_v1.precision import MasterAdamW
 from sft.data import Corpus,serialize
 from sft.evaluate import loss_sum,generate
 from sml_v1.tokenization import Tokenizer
 mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed'])
 model=load_model(PARENT/'model.safetensors');corpus=Corpus(CORPUS,'train');batches=order_for(len(corpus),PLAN)
 if len(batches)!=469 or sorted(np.concatenate(batches).tolist())!=list(range(15000)):raise ValueError('Training membership changed')
 before=dict(tree_flatten(model.parameters()));one=nn.value_and_grad(model,lambda x,y:loss_sum(model,x,y));chosen=batches[0][:4]
 value,g,profile=accumulate_gradients(one,corpus,chosen,policy=PLAN['padding_policy'],microbatch=PLAN['microbatch'])
 norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));mx.eval(norm)
 if not math.isfinite(value) or not math.isfinite(norm.item()) or norm.item()<=0:raise ValueError('Nonfinite/zero gradient')
 x,y=corpus.batch(chosen);whole=float(loss_sum(model,mx.array(x),mx.array(y)).item())/len(chosen)
 if abs(whole-value)>0.02:raise ValueError('Accumulated loss disagrees with whole real-record batch')
 del g
 opt=MasterAdamW(learning_rate=PLAN['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=PLAN['weight_decay']);opt.init(model.trainable_parameters());mx.eval(opt.state)
 if int(opt.step.item())!=0:raise ValueError('Optimizer not fresh')
 unchanged=all(bool(mx.array_equal(v,before[k]).item()) for k,v in tree_flatten(model.parameters()))
 if not unchanged or sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Parent changed during preflight')
 # Only DEVELOPMENT smoke generations; no test generation and no hidden JSON adapter.
 rows=load(CORPUS/'development.jsonl');tok=Tokenizer(TOK);outputs=[];model.eval()
 for variant in ('canonical','alias','nullable'):
  r=next(r for r in rows if r['variant']==variant);p,_=serialize(tok,r['messages'][:-1],generation=True)
  if len(p)+PLAN['max_tokens']>model.cfg.max_seq_len:raise ValueError('Context overflow')
  o=generate(model,tok,r['messages'][:-1],limit=PLAN['max_tokens'],temperature=0.)
  outputs.append(dict(id=r['id'],variant=variant,**o,checks=score(r,o['text'],o['stop_reason'])))
 atomic_json(REPORT/'preflight-generations.json',dict(parent_sha256=PARENT_SHA,development_only=True,raw_outputs=True,optimizer_updates=0,outputs=outputs))
 receipt=dict(status='passed',contract=contract(),splits=m['splits'],total_steps=len(batches),scorer_tests=11,
  gpu=dict(loss=value,whole_batch_loss=whole,gradient_norm=float(norm.item()),accumulation=profile,fresh_optimizer_step=0,parent_parameters_unchanged=unchanged),
  optimizer_updates=0,checkpoints_created=0,test_predictions_created=0,elapsed_seconds=time.monotonic()-started)
 atomic_json(REPORT/'ready.json',receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k!='contract'},indent=2),flush=True)
if __name__=='__main__':main()
