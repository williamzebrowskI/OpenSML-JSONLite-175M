import json, math
import numpy as np
from .common import *

def model_at(path=BASE):
    import mlx.core as mx
    from mlx.utils import tree_map
    from sml_v1.model import TransformerLM,TransformerConfig
    cfg=json.loads(Path(str(path)+'.json').read_text())['recipe']['model']
    model=TransformerLM(TransformerConfig(**cfg))
    model.update(tree_map(lambda a:a.astype(mx.bfloat16),model.parameters()))
    model.load_weights(str(path));mx.eval(model.parameters());return model

class Encoding:
    def __init__(self):
        from sml_v1.tokenization import Tokenizer
        self.tokenizer=Tokenizer(TOK)
    def prefix(self,document,schema):
        from sft.data import serialize
        return serialize(self.tokenizer,[dict(role='user',content=prompt(document,schema))],generation=True)[0].tolist()
    def row(self,r,training=False):
        p=self.prefix(r['document'],r['schema']);t=self.tokenizer.assert_roundtrip(dumps(r['expected']))+[3]
        if len(p)+256>2048 or len(t)>256 or (training and len(p)+len(t)>PLAN['train_max_sequence']):raise ValueError('No truncation; request exceeds token budget')
        return np.array((p+t)[:-1],np.int32),np.array([-100]*(len(p)-1)+t,np.int32)

def batch(encoded,ids):
    n=32*math.ceil(max(len(encoded[int(i)][0]) for i in ids)/32)
    x=np.zeros((len(ids),n),np.int32);y=np.full_like(x,-100)
    for j,i in enumerate(ids):a,b=encoded[int(i)];x[j,:len(a)]=a;y[j,:len(b)]=b
    return x,y

def loss(model,x,y):
    import mlx.core as mx,mlx.nn as nn
    z,_=model.logits(x);mask=y!=-100;safe=mx.where(mask,y,0)
    ce=nn.losses.cross_entropy(z.astype(mx.float32),safe,reduction='none')
    return mx.mean(mx.sum(ce*mask,axis=-1)/mx.maximum(1,mx.sum(mask,axis=-1)))

def gradients(model,encoded,indices,micro=2):
    import mlx.core as mx,mlx.nn as nn
    from mlx.utils import tree_map
    vg=nn.value_and_grad(model,lambda x,y:loss(model,x,y));g=None;v=0.;n=0
    for j in range(0,len(indices),micro):
        ids=indices[j:j+micro];x,y=batch(encoded,ids);a,b=vg(mx.array(x),mx.array(y));mx.eval(a)
        weight=len(ids)/len(indices);v+=a.item()*weight;n+=int(np.sum(y!=-100))
        b=tree_map(lambda z:z.astype(mx.float32)*weight,b)
        g=b if g is None else tree_map(lambda z,w:z+w,g,b);mx.eval(g)
    return v,g,n

def lr(step,total,plan):
    if step<=plan['warmup']:return plan['peak_lr']*step/plan['warmup']
    return plan['final_lr']+(plan['peak_lr']-plan['final_lr'])*(1+math.cos(math.pi*(step-plan['warmup'])/max(1,total-plan['warmup'])))/2

def order_for(encoded,seed,batch_size):
    rng=np.random.default_rng(seed);shuffled=rng.permutation(len(encoded));batches=[]
    for start in range(0,len(encoded),batch_size*8):
        window=sorted(shuffled[start:start+batch_size*8],key=lambda i:len(encoded[int(i)][0]))
        groups=[window[j:j+batch_size] for j in range(0,len(window),batch_size)];rng.shuffle(groups);batches.extend(groups)
    return [[int(i) for i in b] for b in batches]

def generate(model,tok,prefix,limit=256):
    # Reference answers are not an argument and never enter model inputs.
    import mlx.core as mx
    ids=mx.array([prefix]);cache=None;out=[];reason='token_limit'
    try:
        for _ in range(limit):
            z,cache=model.logits(ids,cache);token=int(mx.argmax(z[0,-1]).item())
            if token in (1,3):reason='turn_end' if token==3 else 'document_end';break
            if token in (0,2):reason='unexpected_structure';break
            out.append(token);ids=mx.array([[token]],dtype=mx.int32)
        return dict(text=tok.decode(out),stop_reason=reason,tokens=len(out))
    finally:mx.clear_cache()
