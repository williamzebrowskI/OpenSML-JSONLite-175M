"""Tiny five-rank proof of assistant masking, sample weighting and resume."""
import argparse
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from sml_v1.common import atomic_json, fingerprint
from sml_v1.checkpoint_bundle import save_bundle, resolve_bundle
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.precision import MasterAdamW
from sml_v1.jaccl_control import ControlChannel
from .evaluate import loss_sum
from .train import state_digest, restore_state


def main():
    p=argparse.ArgumentParser(); p.add_argument('--job-dir',type=Path,required=True); a=p.parse_args(); job=a.job_dir
    rank=int(os.environ['MLX_RANK']); atomic_json(job/f'worker_rank{rank}.json',dict(pid=os.getpid(),output=str(job)))
    watchdog=threading.Timer(600,lambda:os._exit(70)); watchdog.daemon=True; watchdog.start()
    mx.set_default_device(mx.cpu); group=mx.distributed.init(backend='jaccl',strict=True)
    if group.size()!=5: raise RuntimeError('Five ranks required')
    control=ControlChannel(mx,group,ring_mode='native')
    # Per-conversation normalization with different assistant target counts;
    # three dummy rows also exercise the partial final global batch.
    x=np.arange(16*12,dtype=np.int32).reshape(16,12)%32
    y=(x+1)%32; y[:,:4]=-100
    for i in range(13): y[i,4:4+(i%5)]=-100
    y[13:]=-100; batches=[7,3,2,2,2]; begin=sum(batches[:rank]); end=begin+batches[rank]
    mx.random.seed(7337)
    model=TransformerLM(TransformerConfig(vocab_size=32,d_model=16,n_heads=2,n_kv_heads=1,n_layers=1,max_seq_len=16,mlp_multiple_of=16,attention_impl='vanilla'))
    grad=nn.value_and_grad(model,lambda xi,yi:loss_sum(model,xi,yi))
    local,grads=grad(mx.array(x[begin:end]),mx.array(y[begin:end]))
    reduced=tree_map(lambda g:mx.distributed.all_sum(g,group=group,stream=mx.cpu)/13,grads)
    reference_loss,expected=grad(mx.array(x),mx.array(y))
    expected=tree_map(lambda g:g/13,expected)
    from mlx.utils import tree_flatten
    errors=[float(mx.max(mx.abs(g-h)).item()) for (_,g),(_,h) in zip(tree_flatten(reduced),tree_flatten(expected))]
    if max(errors)>1e-5: raise RuntimeError('Conversation-weighted distributed gradient differs from full reference')
    model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()))
    optimizer=MasterAdamW(learning_rate=2e-5); optimizer.init(model.trainable_parameters())
    def update():
        _,gradients=grad(mx.array(x[begin:end]),mx.array(y[begin:end]))
        gradients=tree_map(lambda g:mx.distributed.all_sum(g.astype(mx.float32),group=group,stream=mx.cpu)/13,gradients)
        optimizer.update(model,gradients); mx.eval(model.parameters(),optimizer.state)
    update(); before=state_digest(model,optimizer)
    if len(set(control.exchange(before,'bf16-replicas')))!=1: raise RuntimeError('SFT replicas diverged')
    scratch=job/f'sft_tiny_rank{rank}'; order=[np.arange(13),np.arange(13,16)]; contract={'fixture':'assistant-only-v1'}
    metadata=dict(step=1,tokens=13,replica_hash=before,sft=dict(contract=contract,conversations=13,order=fingerprint([a.tolist() for a in order])))
    bundle=save_bundle(scratch,model,optimizer,metadata,dict(next_batch=1),keep=1,reserve_gib=0)
    update(); uninterrupted=state_digest(model,optimizer)
    weights=resolve_bundle(bundle); restore_state(weights,model,optimizer,contract,order,SimpleNamespace(lengths=np.ones(16,dtype=np.int64)))
    update(); resumed=state_digest(model,optimizer)
    if resumed!=uninterrupted: raise RuntimeError('Resumed next optimizer update differs from uninterrupted update')
    if len(set(control.exchange(resumed,'resumed-replicas')))!=1: raise RuntimeError('Resumed replicas diverged')
    atomic_json(job/f'probe_rank{rank}.json',dict(rank=rank,status='passed',world=5,assistant_normalization_max_error=max(errors),
        partial_batch_real_conversations=13,bf16_compute=True,fp32_master_optimizer=True,exact_next_update_resume=True))
    control.exchange(True,'done'); watchdog.cancel()
    print(f'[sft-proof-passed] rank={rank} masked conversation gradients / partial batch / exact next update resume',flush=True)


if __name__=='__main__': main()
