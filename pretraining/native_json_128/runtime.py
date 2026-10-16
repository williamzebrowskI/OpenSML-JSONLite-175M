"""Native raw JSON generation and masked full-weight SFT."""
import json,math
from pathlib import Path
import numpy as np
from .common import *

def model_at(weight=None):
 import mlx.core as mx
 from sml_v1.model import TransformerLM,TransformerConfig
 p=Path(weight) if weight else PARENT/'model.safetensors'
 cfg=json.loads(Path(str(p)+'.json').read_text())['recipe']['model']
 m=TransformerLM(TransformerConfig(**cfg));m.load_weights(str(p));mx.eval(m.parameters());return m

class Encoding:
 def __init__(self):
  from sml_v1.tokenization import Tokenizer
  self.tokenizer=Tokenizer(TOK)
 def prefix(self,messages):
  from sft.data import serialize
  return serialize(self.tokenizer,messages,generation=True)[0].tolist()
 def row(self,r):
  prefix=self.prefix(r['messages'][:-1]);target=self.tokenizer.encode(r['messages'][-1]['content'])+[3]
  if len(prefix)+PLAN['max_tokens']>PLAN['context'] or len(prefix)+len(target)>PLAN['context'] or len(target)>256:raise ValueError('Truncation forbidden')
  return np.array((prefix+target)[:-1],np.int32),np.array([-100]*(len(prefix)-1)+target,np.int32)

def batch(encoded,ids):
 length=max(len(encoded[int(i)][0]) for i in ids);length=128*math.ceil(length/128)
 if length>2048:raise ValueError('Context overflow')
 x=np.zeros((len(ids),length),np.int32);y=np.full_like(x,-100)
 for k,i in enumerate(ids):a,b=encoded[int(i)];x[k,:len(a)]=a;y[k,:len(b)]=b
 return x,y

def loss(model,x,y):
 import mlx.core as mx,mlx.nn as nn
 z,_=model.logits(x);mask=y!=-100;safe=mx.where(mask,y,0)
 ce=nn.losses.cross_entropy(z.astype(mx.float32),safe,reduction='none')
 return mx.mean(mx.sum(ce*mask,axis=-1)/mx.maximum(mx.sum(mask,axis=-1),1))

def accumulate(model,encoded,ids,microbatch=None):
 import mlx.core as mx,mlx.nn as nn
 from mlx.utils import tree_map
 microbatch=PLAN['microbatch'] if microbatch is None else microbatch
 vg=nn.value_and_grad(model,lambda x,y:loss(model,x,y));g=None;value=0.;tokens=0
 for start in range(0,len(ids),microbatch):
  group=ids[start:start+microbatch];x,y=batch(encoded,group);v,a=vg(mx.array(x),mx.array(y));mx.eval(v);weight=len(group)/len(ids);value+=v.item()*weight;tokens+=int(np.sum(y!=-100));a=tree_map(lambda q:q.astype(mx.float32)*weight,a);g=a if g is None else tree_map(lambda q,r:q+r,g,a);mx.eval(g)
 return value,g,tokens

def learning_rate(step,total):
 warm=min(PLAN['warmup'],max(1,total//4))
 if step<=warm:return PLAN['peak_lr']*step/warm
 return PLAN['final_lr']+(PLAN['peak_lr']-PLAN['final_lr'])*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))/2
