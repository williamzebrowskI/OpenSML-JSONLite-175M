"""Small checks for the new padding, cursor and distributed weighting logic."""
import copy,json,sys
from pathlib import Path
import numpy as np
from .common import PLAN,MIX,REPORT,atomic_json,TOKENIZER,lr
from .stream import WholeRows
from sml_v1.tokenization import Tokenizer

def main():
    t=Tokenizer(TOKENIZER);w=WholeRows(t,'train',set())
    for k in list(MIX)[5:]:
        for _ in range(5):w.row(k)
    state=w.state_dict();r=WholeRows(t,'train',set(),state)
    for _ in range(32):
        for k in list(MIX)[5:]:
            a,b=w.row(k);x,y=r.row(k)
            assert np.array_equal(a,x) and np.array_equal(b,y)
            n=np.count_nonzero(b!=-100)
            assert n>0 and np.all(a[n:]==0) and np.all(b[n:]==-100)
            assert np.array_equal(a[1:n],b[:n-1]) and b[n-1]==t.eos
    assert w.state_dict()==r.state_dict()
    assert PLAN['additional_tokens'] is None and lr(10**12)==PLAN['peak_lr']
    # Variable token counts across unequal ranks must not overweight padded rows.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten,tree_map
    from .worker import weighted_backward
    mx.random.seed(151)
    class MaskedLoss(nn.Module):
        def __init__(self):super().__init__();self.weight=mx.arange(16,dtype=mx.float32)/16
        def __call__(self,x,targets):
            mask=targets!=-100;error=self.weight[x]-mx.maximum(targets,0)/16
            return {'loss':mx.sum(error**2*mask)/mx.sum(mask)}
    model=MaskedLoss()
    x=np.arange(40,dtype=np.int32).reshape(5,8)%16;y=(x+1)%16
    for i,n in enumerate([2,8,3,6,1]):y[i,n:]=-100
    n=int(np.count_nonzero(y!=-100))
    _,ref=nn.value_and_grad(model,lambda xx,yy:model(xx,targets=yy)['loss'])(mx.array(x),mx.array(y))
    totals=[];values=[]
    for begin,end in [(0,2),(2,3),(3,5)]:
        v,g=weighted_backward(model,x[None,begin:end],y[None,begin:end],n,3)
        totals.append(g);values.append(v)
    got=tree_map(lambda a,b,c:(a+b+c)/3,*totals)
    delta=max(float(mx.max(mx.abs(a-b)).item()) for (_,a),(_,b) in zip(tree_flatten(ref),tree_flatten(got)))
    assert delta<1e-5,delta
    actual=sum(values)/n;expected=float(model(mx.array(x),targets=mx.array(y))['loss'].item())
    assert abs(actual-expected)<1e-5,(actual,expected)
    result=dict(status='passed',whole_document_shift_padding=True,resumed_rows_checked=128,
                unequal_rank_masked_token_gradient_max_difference=delta,loss_difference=abs(actual-expected),
                indefinite_lr_no_token_stop=True)
    atomic_json(REPORT/'unit-checks.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
