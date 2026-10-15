"""Full-vocabulary frozen-parent KL on replay assistant tokens only.

The parent never supplies gold labels and never sees held-out requests here.
Each sequence contributes CE + lambda KL(parent || student), averaged over
assistant output tokens, including the two stop tokens. New contrasts use CE
alone, allowing their distributions to move away from the parent's mistakes.
"""
import numpy as np


def masked_objective(student_logits, teacher_logits, targets, weight):
 import mlx.core as mx
 import mlx.nn as nn
 mask=targets!=-100
 safe=mx.where(mask,targets,0)
 logits=student_logits.astype(mx.float32)
 ce=nn.losses.cross_entropy(logits,safe,reduction='none')
 if teacher_logits is None:
  kl=mx.zeros_like(ce)
 else:
  ref=mx.stop_gradient(teacher_logits.astype(mx.float32))
  logp=ref-mx.logsumexp(ref,axis=-1,keepdims=True)
  logq=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
  kl=mx.sum(mx.exp(logp)*(logp-logq),axis=-1)
 counts=mx.maximum(mx.sum(mask,axis=1),1)
 return mx.sum(mx.sum((ce+weight*kl)*mask,axis=1)/counts)


def objectives(model,teacher,weight):
 import mlx.core as mx
 import mlx.nn as nn
 def replay(x,y):
  reference,_=teacher.logits(x)
  logits,_=model.logits(x)
  return masked_objective(logits,mx.stop_gradient(reference),y,weight)
 def new(x,y):
  logits,_=model.logits(x)
  return masked_objective(logits,None,y,0.)
 return (nn.value_and_grad(model,replay),nn.value_and_grad(model,new))


def accumulate_gradients(one,corpus,indices,*,policy,microbatch=2):
 import mlx.core as mx
 from mlx.utils import tree_map
 from native_json_100.runtime import accumulate_gradients as plain
 from .data import CORPUS,load
 if not hasattr(corpus,'replay_indices'):
  # Explicit prepared membership, never guessed from the contents or gold text.
  rows=load(CORPUS/'train.jsonl')
  if len(rows)!=len(corpus):raise ValueError('Replay membership mismatch')
  corpus.replay_indices={i for i,r in enumerate(rows) if r['source']=='replay'}
 total=None;value=0.;parts=[]
 indices=np.asarray(indices,np.int64)
 for is_replay,fn in [(True,one[0]),(False,one[1])]:
  chosen=np.array([i for i in indices if (int(i) in corpus.replay_indices)==is_replay],np.int64)
  if not len(chosen):continue
  loss,g,profile=plain(fn,corpus,chosen,policy=policy,microbatch=microbatch)
  fraction=len(chosen)/len(indices)
  g=tree_map(lambda v:v*fraction,g)
  total=g if total is None else tree_map(lambda a,b:a+b,total,g)
  value+=loss*fraction;parts.append(dict(replay=is_replay,records=len(chosen),objective=loss,profile=profile))
 if total is None:raise ValueError('Empty batch')
 mx.eval(total)
 return value,total,dict(actual_records=len(indices),parts=parts,optimizer_updates=0)
