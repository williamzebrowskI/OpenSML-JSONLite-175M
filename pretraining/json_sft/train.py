"""Local JSON-only SFT; invoked only by explicit --run launcher."""
import argparse
from dataclasses import asdict
import gzip
import math
from pathlib import Path
import signal
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_map,tree_unflatten
from sml_v1.common import read_json,atomic_json,fingerprint
from sml_v1.checkpoint_bundle import resolve_bundle,save_bundle
from sml_v1.model import TransformerConfig,TransformerLM
from sml_v1.precision import MasterAdamW
from sft.data import Corpus,epoch_order,learning_rate
from .corpus import load_tokenizer
from .evaluate import loss_sum,evaluate
from .schedule import evaluation_due

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    output=a.output;session=read_json(output/'session.json');plan=session['plan'];contract=session['contract']
    weighted=plan.get('value_weight',1)!=1
    if weighted:
        from .value_weights import ValueWeightedCorpus,loss_sum as weighted_loss_sum
        corpus=ValueWeightedCorpus(plan['corpus'],'train',plan['value_weight'])
    else:corpus=Corpus(plan['corpus'],'train')
    order=[batch for epoch in range(plan['epochs']) for batch in epoch_order(corpus.lengths,plan['seed']+epoch,plan['batch_size'])]
    order_hash=fingerprint([v.tolist() for v in order]);parent=read_json(output/'parent/model.safetensors.json')
    mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(plan['seed'])
    model=TransformerLM(TransformerConfig(**parent['recipe']['model']))
    model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()))
    model.load_weights(str(output/'parent/model.safetensors'))
    lora=plan.get('adaptation')=='lora'
    if lora:
        from .adapters import install,merged_parameters,ExportView
        install(model,plan['lora_rank'],plan['lora_alpha'])
    optimizer=MasterAdamW(learning_rate=plan['peak_lr'],betas=tuple(plan['betas']),eps=plan['eps'],weight_decay=plan['weight_decay'])
    step=tokens=records=0;best=None
    if a.resume:
        weights=resolve_bundle(output/'latest.json');metadata=read_json(weights+'.json');saved=metadata['json_sft']
        if saved['contract']!=contract or saved['order_hash']!=order_hash:raise ValueError('Resume contract/order differs')
        step=metadata['step'];tokens=metadata['tokens'];records=saved['records'];best=saved['best_selection']
        if type(step) is not int or not 0<=step<=len(order):raise ValueError('Invalid resume step')
        indices=np.concatenate(order[:step]) if step else np.array([],np.int64)
        expected_tokens=sum(int(corpus.offsets[i+1]-corpus.offsets[i]) for i in indices)
        with gzip.open(weights+'.rank0.data_state.json.gz','rt') as f:cursor=__import__('json').load(f)
        if tokens!=expected_tokens or records!=len(indices) or cursor!={'step':step,'stream_state':{'next_batch':step}}:raise ValueError('Resume cursor/counters differ')
        if not lora:model.load_weights(weights)
        optimizer.init(model.trainable_parameters())
        state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
        if lora:model.update(state.pop('lora_weights'))
        optimizer.state=state
        mx.eval(model.parameters(),optimizer.state)
        if int(optimizer.step.item())!=step:raise ValueError('Optimizer step differs')
    else:
        # FP32 master parameters must start from loaded parent weights.
        optimizer.init(model.trainable_parameters())
    if any(v.dtype!=mx.bfloat16 for _,v in tree_flatten(model.parameters())):raise ValueError('Parent compute weights must be BF16')
    mx.eval(model.parameters(),optimizer.state)
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    one=(nn.value_and_grad(model,lambda x,y,w:weighted_loss_sum(model,x,y,w)) if weighted
         else nn.value_and_grad(model,lambda x,y:loss_sum(model,x,y)))
    def save(is_best=False):
        metadata={'step':step,'tokens':tokens,'recipe':{'model':asdict(model.cfg)},'json_sft':{
            'format':'opensml-json-sft-run-v1','contract':contract,'parent':session['parent'],
            'records':records,'total_steps':len(order),'order_hash':order_hash,'best_selection':best,
            'adaptation':plan.get('adaptation','full'),'lora_rank':plan.get('lora_rank'),'lora_alpha':plan.get('lora_alpha'),
            'serialization':session.get('serialization','extract-envelope-output-json-v1'),'loss_weighting':('record mean, output tokens only; value weight '+str(plan['value_weight']) if weighted else 'record mean, output tokens only')}}
        if lora:optimizer.state['lora_weights']=model.trainable_parameters()
        try:save_bundle(output,ExportView(model) if lora else model,optimizer,metadata,{'next_batch':step},best=is_best,keep=3,reserve_gib=10)
        finally:
            if lora:optimizer.state.pop('lora_weights',None)
    def assess():
        evaluated_model=model
        if lora:
            evaluated_model=TransformerLM(model.cfg);evaluated_model.update(merged_parameters(model));mx.eval(evaluated_model.parameters())
        result=evaluate(evaluated_model,load_tokenizer(),plan['corpus'],step,output/'evaluations',per_source=plan.get('eval_per_source',12))
        atomic_json(output/'last_eval.json',dict(tokens=tokens,**result));return [result['macro_exact_json'],-result['macro_output_nll']]
    if not a.resume:best=assess();save(True)
    began=time.monotonic();last_saved=step
    print(f'[ready-json-sft] local Mac1; step={step}/{len(order)}; batch={plan["batch_size"]}; output-only loss; passes={plan["epochs"]}; adaptation={plan.get("adaptation","full")}; trainable={sum(v.size for _,v in tree_flatten(model.trainable_parameters())):,}',flush=True)
    while step<len(order):
        if stop[0] or (output/'STOP').exists():break
        indices=order[step];x,y=corpus.batch(indices,size=plan['batch_size']);total=None;loss_total=0.
        if weighted:weights=corpus.weight_batch(indices,y.shape)
        for offset in range(0,plan['batch_size'],plan['microbatch']):
            args=[mx.array(x[offset:offset+plan['microbatch']]),mx.array(y[offset:offset+plan['microbatch']])]
            if weighted:args.append(mx.array(weights[offset:offset+plan['microbatch']]))
            loss,grads=one(*args)
            grads=tree_map(lambda g:g.astype(mx.float32),grads)
            total=grads if total is None else tree_map(lambda u,v:u+v,total,grads)
            mx.eval(total,loss);loss_total+=float(loss.item())
        total=tree_map(lambda g:g/len(indices),total);norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(total)))
        mx.eval(total,norm)
        if not math.isfinite(float(norm.item())) or not math.isfinite(loss_total):raise RuntimeError('Nonfinite loss/gradient')
        total=tree_map(lambda g:g*mx.minimum(1.,plan['grad_clip']/(norm+1e-6)),total)
        lr=learning_rate(step+1,len(order),plan['peak_lr'],plan['final_lr'],plan['warmup_fraction'])
        optimizer.learning_rate=lr;optimizer.update(model,total);mx.eval(model.parameters(),optimizer.state)
        before=tokens;tokens+=sum(int(corpus.offsets[i+1]-corpus.offsets[i]) for i in indices);records+=len(indices);step+=1
        if step==1 or step%10==0:print(f'[json-sft {step}/{len(order)}] output_loss={loss_total/len(indices):.4f} lr={lr:.3e} records={records} nonpadding_tokens={tokens} elapsed={time.monotonic()-began:.1f}s',flush=True)
        cadence=dict(plan,pass_tokens=session['pass_tokens'])
        due=evaluation_due(step,len(order),before,tokens,cadence)
        if due:
            candidate=assess();improved=candidate>best
            if improved:best=candidate
            save(improved);last_saved=step
    if last_saved!=step:save(False)
    atomic_json(output/'result.json',{'status':'complete' if step==len(order) else 'stopped','step':step,'total_steps':len(order),'records':records,'tokens':tokens})

if __name__=='__main__':main()
