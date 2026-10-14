"""Five-Mac full-parameter SFT with exact one-pass cursor and atomic resume."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import math
import os
from pathlib import Path
import signal
import threading
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v1.checkpoint_bundle import resolve_bundle, save_bundle
from sml_v1.jaccl_control import ControlChannel
from sml_v1.model import TransformerLM, TransformerConfig, count_parameters
from sml_v1.precision import MasterAdamW
from sml_v1.tokenization import Tokenizer
from .data import Corpus, training_order, learning_rate
from .evaluate import loss_sum, evaluate
from .behavior import selection


def state_digest(model, optimizer):
    digest=hashlib.sha256()
    for prefix,state in [('model',model.parameters()),('optimizer',optimizer.state)]:
        for name,value in sorted(tree_flatten(state)):
            digest.update((prefix+name+str(value.shape)+str(value.dtype)).encode())
            digest.update(np.asarray(value.astype(mx.float32) if value.dtype==mx.bfloat16 else value).tobytes())
    return digest.hexdigest()


def restore_state(weights, model, optimizer, contract, order, corpus):
    metadata=read_json(weights+'.json'); sft=metadata['sft']
    if sft['contract']!=contract or sft['order']!=fingerprint([x.tolist() for x in order]): raise ValueError('SFT resume contract/order changed')
    with gzip.open(weights+'.rank0.data_state.json.gz','rt') as f: cursor=__import__('json').load(f)
    step=metadata['step']
    if cursor!=dict(step=step,stream_state=dict(next_batch=step)) or not 0<=step<=len(order): raise ValueError('SFT cursor mismatch')
    expected_indices=np.concatenate(order[:step]) if step else np.array([],dtype=np.int64)
    if metadata['tokens']!=int(corpus.lengths[expected_indices].sum()) or sft['conversations']!=len(expected_indices): raise ValueError('SFT token/example counter mismatch')
    model.load_weights(weights)
    optimizer.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(model.parameters(),optimizer.state)
    if state_digest(model,optimizer)!=metadata['replica_hash']: raise ValueError('Resumed model/optimizer digest differs')
    if int(optimizer.step.item())!=step: raise ValueError('Optimizer step/cursor differs')
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job-dir',type=Path,required=True); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-steps',type=int,default=0)
    args=parser.parse_args(); source=args.source; job=args.job_dir; output=args.output
    plan=read_json(source/'plan.json'); local=plan.get('execution')=='local'
    rank=0 if local else int(os.environ['MLX_RANK']); job.mkdir(parents=True,exist_ok=True)
    atomic_json(job/f'worker_rank{rank}.json',dict(pid=os.getpid(),output=str(job),save_dir=str(output)))
    heartbeat=[time.monotonic()]; done=threading.Event()
    def watch():
        while not done.wait(10):
            if time.monotonic()-heartbeat[0]>1800: os._exit(70)
    threading.Thread(target=watch,daemon=True).start()
    stop=[False]
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    if local:
        from .local_probe import LocalControl
        group=None; world=1; control=LocalControl()
    else:
        if os.environ.get('MLX_JACCL_RING')!='1': raise RuntimeError('Native ring required')
        group=mx.distributed.init(backend='jaccl',strict=True)
        if group.size()!=5 or group.rank()!=rank: raise RuntimeError('Five ranks required')
        world=5; control=ControlChannel(mx,group,ring_mode='native')
    mx.set_default_device(mx.gpu)
    def root_call(label, fn):
        result=None; status=None
        if rank==0:
            try: result=fn(); status=dict(ok=True)
            except Exception as exc: status=dict(ok=False,error=f'{type(exc).__name__}: {str(exc)[:2500]}')
        status=control.broadcast(status,label+'-status'); heartbeat[0]=time.monotonic()
        if not status['ok']: raise RuntimeError(status['error'])
        return result
    plan=read_json(source/'plan.json'); manifest=read_json(source/'corpus/manifest.json')
    parent=read_json(source/'parent/model.safetensors.json')
    # A new SFT continuation loads parent model weights with a fresh optimizer;
    # selection still uses the original document-validation ceiling.
    continuation=plan.get('continuation')
    plan['parent_val_loss']=continuation['stage_a_val_loss'] if continuation else parent['best_val_loss']
    tokenizer=Tokenizer(source/'tokenizer'); corpus=Corpus(source/'corpus','train')
    size=sum(plan['batches'])*plan['grad_accum']; order=training_order(corpus.lengths,plan['seed'],size,plan.get('epochs',1))
    if len(order)!=manifest['total_steps']: raise ValueError('SFT pass/step count differs')
    contract=read_json(source/'contract.json')
    if len(set(fingerprint(x) for x in control.exchange(contract,'contracts')))!=1: raise ValueError('Worker contracts differ')
    mx.set_cache_limit(int(plan['cache_gib_by_mac'][[0,1,2,4,3][rank]]*1024**3))
    mx.random.seed(plan['seed'])
    model=TransformerLM(TransformerConfig(**parent['recipe']['model']))
    model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()))
    model.load_weights(str(source/'parent/model.safetensors'))
    optimizer=MasterAdamW(learning_rate=plan['peak_lr'],betas=tuple(plan['betas']),eps=plan['eps'],weight_decay=plan['weight_decay'])
    optimizer.init(model.trainable_parameters()); mx.eval(model.parameters(),optimizer.state)
    step=tokens=conversations=0; best=None; metadata=None; chat_regressions=0
    if args.resume:
        weights=resolve_bundle(source/'resume')
        metadata=restore_state(weights,model,optimizer,contract,order,corpus)
        step,tokens=metadata['step'],metadata['tokens']; conversations=metadata['sft']['conversations']; best=metadata['sft']['best_selection']
        chat_regressions=metadata['sft'].get('chat_regression_streak',0)
        if plan.get('natural_transfer'):
            baseline=read_json(output/'baseline.json')
            plan['selection_reference']={k:baseline[k] for k in ('chat_behavior','natural_dev')}
    if any(v.dtype!=mx.bfloat16 for _,v in tree_flatten(model.parameters())): raise ValueError('Model must be BF16')
    if any(mx.issubdtype(v.dtype,mx.floating) and v.dtype!=mx.float32 for _,v in tree_flatten(optimizer.state)): raise ValueError('Optimizer must be FP32')
    replica=state_digest(model,optimizer)
    if len(set(control.exchange(replica,'initial-replicas')))!=1: raise RuntimeError('Initial replicas differ')
    token_weighted=plan.get('loss_weighting','conversation')=='token'
    if rank==0: print(f'[ready-sft] params={count_parameters(model):,} {"SFT continuation" if continuation else "Stage A"} parent={parent["step"]} sft_step={step}/{len(order)} batch={size}; fresh optimizer on new run; assistant-only {"token" if token_weighted else "conversation"} mean',flush=True)
    one=nn.value_and_grad(model,lambda x,y:loss_sum(model,x,y,token_weighted=token_weighted))
    one=mx.compile(one,inputs=[model.state],outputs=[model.state])
    def assess(retention):
        return control.broadcast(root_call('evaluation',lambda:evaluate(model,tokenizer,source/'corpus',plan,step=step,output=output/'evaluations',retention=retention)),'evaluation-summary')
    last_saved=[None]
    def save(is_best=False):
        digest=state_digest(model,optimizer)
        if len(set(control.exchange(digest,'checkpoint-replicas')))!=1: raise RuntimeError('Replica divergence; refusing checkpoint')
        meta=dict(step=step,tokens=tokens,replica_hash=digest,recipe=dict(model=asdict(model.cfg)),
            sft=dict(format='opensml-sft-run-v1',contract=contract,parent_step=parent['step'],parent_tokens=parent['tokens'],parent_val_loss=continuation['parent_retention_loss'] if continuation else parent['best_val_loss'],
                     continuation=continuation,reference_parent_val_loss=plan['parent_val_loss'],
                     conversations=conversations,total_steps=len(order),order=fingerprint([x.tolist() for x in order]),best_selection=best,chat_regression_streak=chat_regressions,
                     serialization=manifest['serialization'],selection='Original Stage A retention ceiling; all-turn generated development contracts in both system variants; evidence/dialogue/format/Python coverage eligibility, macro score, common chat, assistant NLL. Separate test gate never selects checkpoints.' if plan.get('evidence_instructions') else 'Original Stage A retention ceiling; generated source/family-heldout development answers in both system variants; common chat regression; assistant NLL only as final tie-break. Separate test gate never selects checkpoints.' if plan.get('verified_extension') else 'Document retention first, chat and contrast acceptance eligibility, paired generated-answer diagnostics, common chat diagnostics, component behavior, assistant NLL' if plan.get('contrast_instructions') else 'Retention eligibility, fixed non-ARC science dev accuracy, chat diagnostics, assistant NLL' if plan.get('science_search') else 'Document retention first, chat acceptance eligibility, conversational diagnostics, component behavior, assistant NLL' if plan.get('short_chat_focus') else 'Dev chat relevance and retention eligibility, then component behavior and assistant NLL' if plan.get('behavior_suite') else 'automatic dev QA/formatting/termination composite; tie: assistant NLL'))
        if plan.get('natural_transfer'):
            meta['sft']['selection']='Original document ceiling and parent broad/core chat nonregression; per-presentation transfer coverage nonregression and >=5pp macro gain; broad chat precedes transfer and NLL. Separate test never selects. Ineligible candidates cannot displace update-zero parent.'
        if plan.get('continuation_trials'):
            meta['sft']['selection']='Original document retention ceiling; generated trial development category macro with provisional social/Python coverage floor. Frozen separate test and raw review decide retention; no benchmark selection.'
        keep=math.ceil(len(order)/plan['eval_every_steps'])+4 if plan.get('keep_all_eval_checkpoints') else 3
        root_call('save',lambda:save_bundle(output,model,optimizer,meta,dict(next_batch=step),best=is_best,keep=keep,reserve_gib=16))
        last_saved[0]=step
    if not args.resume:
        baseline=assess(True); best=selection(baseline,plan)
        if plan.get('natural_transfer'):
            plan['selection_reference']={k:baseline[k] for k in ('chat_behavior','natural_dev')}
        root_call('baseline-record',lambda:atomic_json(output/'baseline.json',baseline)); save(True)
    start_step=step; started=time.monotonic()
    while step<len(order):
        requested=control.broadcast((stop[0] or (output/'STOP').exists()) if rank==0 else None,'stop')
        if requested: break
        indices=order[step]
        # Every worker has the same immutable mmap corpus and seed. Verify batch
        # indices and bytes before any collective gradient communication.
        x,y=corpus.batch(indices,size,max_length=plan['max_seq_len'])
        denominator=int((y!=-100).sum()) if token_weighted else len(indices)
        if denominator<1: raise ValueError('Empty supervised global batch')
        batch_digest=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        if len(set(control.exchange(batch_digest,'batch-hash')))!=1: raise RuntimeError('Rank batch bytes differ')
        micro_size=sum(plan['batches']); begin=sum(plan['batches'][:rank]); end=begin+plan['batches'][rank]
        x=x.reshape(plan['grad_accum'],micro_size,-1)[:,begin:end]; y=y.reshape(plan['grad_accum'],micro_size,-1)[:,begin:end]
        total=None; local_loss=mx.array(0.,dtype=mx.float32)
        for xi,yi in zip(x,y):
            loss,grads=one(mx.array(xi),mx.array(yi))
            grads=tree_map(lambda g:g.astype(mx.float32),grads)
            total=grads if total is None else tree_map(lambda a,b:a+b,total,grads)
            local_loss=local_loss+loss.astype(mx.float32); mx.eval(total,local_loss)
        total=tree_map(lambda g:g*world/denominator,total)
        if not local:
            total=nn.average_gradients(total,group=group,all_reduce_size=32*1024**2,communication_stream=mx.cpu)
        norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(total)))
        mx.eval(total,norm); norm_value=float(norm.item())
        if not math.isfinite(norm_value): raise RuntimeError('Nonfinite SFT gradient')
        total=tree_map(lambda g:g*mx.minimum(1.,plan['grad_clip']/(norm+1e-6)),total)
        lr=learning_rate(step+1,len(order),plan['peak_lr'],plan['final_lr'],plan['warmup_fraction'])
        optimizer.learning_rate=lr; optimizer.update(model,total); mx.eval(model.parameters(),optimizer.state)
        global_loss=sum(control.exchange(float(local_loss.item()),'loss'))/denominator
        if not math.isfinite(global_loss): raise RuntimeError('Nonfinite SFT loss')
        before=tokens; tokens+=int(corpus.lengths[indices].sum()); conversations+=len(indices); step+=1; heartbeat[0]=time.monotonic()
        if rank==0 and (step%10==0 or step==start_step+1):
            print(f'[sft {step}/{len(order)}] assistant_loss={global_loss:.4f} lr={lr:.3e} grad_norm={norm_value:.3f} conversations={conversations} input_tokens={tokens} elapsed={time.monotonic()-started:.1f}s',flush=True)
        final=step==len(order); bounded=args.stop_after_steps and step-start_step>=args.stop_after_steps
        due=tokens//plan['eval_every_tokens']!=before//plan['eval_every_tokens']
        if plan.get('eval_every_steps'):
            due=due or step%plan['eval_every_steps']==0
        is_best=False
        if final or due:
            retention=bool(plan.get('behavior_suite')) or final or tokens//plan['retention_every_tokens']!=before//plan['retention_every_tokens']
            result=assess(retention); candidate=selection(result,plan)
            if plan.get('natural_transfer'):
                from .natural_transfer import eligibility
                checks=eligibility(result,plan)
                regressed=not all(checks[k] for k in ('retained','broad_nonregression','core_nonregression'))
                chat_regressions=chat_regressions+1 if regressed else 0
                result['chat_regression_streak']=chat_regressions
                if chat_regressions>=plan['early_stop_chat_regressions']:
                    stop[0]=True
                    if rank==0:print('[early-stop] Consecutive evaluations regressed retention or broad/core chat. All evaluated checkpoints are preserved; best may remain parent update zero.',flush=True)
            result['selection_key']=candidate
            if plan.get('behavior_suite') and not plan.get('science_search'): result['chat_candidate_eligible']=bool(candidate[1] if plan.get('short_chat_focus') else candidate[0])
            if best is None or candidate>best: best=candidate; is_best=True
            root_call('latest-eval',lambda:atomic_json(output/'last_eval.json',dict(tokens=tokens,**result)))
        if final or due or bounded: save(is_best)
        if bounded: break
    # Ctrl-C/STOP at the boundary saves the exact next global batch.
    if last_saved[0]!=step: save(False)
    root_call('result',lambda:atomic_json(job/'result.json',dict(status='complete' if step==len(order) else 'stopped',step=step,total_steps=len(order),tokens=tokens,conversations=conversations)))
    control.exchange(True,'finished'); done.set()


if __name__=='__main__': main()
