"""Single-Mac accumulation, partial-batch and exact-resume proof."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map,tree_flatten
from sml_v1.common import atomic_json,fingerprint
from sml_v1.model import TransformerLM,TransformerConfig
from sml_v1.precision import MasterAdamW
from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
from .evaluate import loss_sum

class LocalControl:
    def broadcast(self,value,label):return value
    def exchange(self,value,label):return [value]

def main():
    from .train import state_digest,restore_state
    p=argparse.ArgumentParser();p.add_argument('--job-dir',type=Path,required=True);a=p.parse_args();job=a.job_dir
    mx.set_default_device(mx.cpu);mx.random.seed(7337)
    model=TransformerLM(TransformerConfig(vocab_size=32,d_model=16,n_heads=2,n_kv_heads=1,n_layers=1,max_seq_len=16,mlp_multiple_of=16,attention_impl='vanilla'))
    x=np.arange(16*12,dtype=np.int32).reshape(16,12)%32;y=(x+1)%32;y[:,:4]=-100
    for i in range(13):y[i,4:4+(i%5)]=-100
    y[13:]=-100
    grad=nn.value_and_grad(model,lambda xi,yi:loss_sum(model,xi,yi))
    def accumulate():
        total=None
        for begin in range(0,16,4):
            _,g=grad(mx.array(x[begin:begin+4]),mx.array(y[begin:begin+4]));g=tree_map(lambda v:v.astype(mx.float32),g)
            total=g if total is None else tree_map(lambda a,b:a+b,total,g);mx.eval(total)
        return tree_map(lambda v:v/13,total)
    actual=accumulate();_,expected=grad(mx.array(x),mx.array(y));expected=tree_map(lambda v:v/13,expected)
    error=max(float(mx.max(mx.abs(g-h)).item()) for (_,g),(_,h) in zip(tree_flatten(actual),tree_flatten(expected)))
    assert error<1e-5,error
    token_grad=nn.value_and_grad(model,lambda xi,yi:loss_sum(model,xi,yi,token_weighted=True))
    token_total=None
    for begin in range(0,16,4):
        _,g=token_grad(mx.array(x[begin:begin+4]),mx.array(y[begin:begin+4]))
        g=tree_map(lambda v:v.astype(mx.float32),g)
        token_total=g if token_total is None else tree_map(lambda a,b:a+b,token_total,g)
        mx.eval(token_total)
    _,token_full=token_grad(mx.array(x),mx.array(y))
    denominator=int((y!=-100).sum())
    token_error=max(float(mx.max(mx.abs((g-h)/denominator)).item()) for (_,g),(_,h) in zip(tree_flatten(token_total),tree_flatten(token_full)))
    assert token_error<1e-5,token_error
    model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()));opt=MasterAdamW(learning_rate=2e-5);opt.init(model.trainable_parameters())
    def update():opt.update(model,accumulate());mx.eval(model.parameters(),opt.state)
    update();before=state_digest(model,opt);order=[np.arange(13),np.arange(13,16)];contract={'fixture':'local-assistant-only-v1'}
    metadata=dict(step=1,tokens=13,replica_hash=before,sft=dict(contract=contract,conversations=13,order=fingerprint([v.tolist() for v in order])))
    bundle=save_bundle(job/'sft_tiny_rank0',model,opt,metadata,dict(next_batch=1),keep=1,reserve_gib=0)
    update();uninterrupted=state_digest(model,opt)
    restore_state(resolve_bundle(bundle),model,opt,contract,order,SimpleNamespace(lengths=np.ones(16,dtype=np.int64)))
    update();assert state_digest(model,opt)==uninterrupted
    assert all(v.dtype==mx.bfloat16 for _,v in tree_flatten(model.parameters()))
    assert all(v.dtype==mx.float32 for _,v in tree_flatten(opt.state) if mx.issubdtype(v.dtype,mx.floating))
    atomic_json(job/'probe_rank0.json',dict(rank=0,world=1,status='passed',assistant_normalization_max_error=error,token_normalization_max_error=token_error,partial_batch_real_conversations=13,bf16_compute=True,fp32_master_optimizer=True,exact_next_update_resume=True))
    print('[sft-local-proof-passed] accumulated conversation gradients / partial batch / exact next update resume',flush=True)
if __name__=='__main__':main()
