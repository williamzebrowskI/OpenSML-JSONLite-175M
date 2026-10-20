"""Isolated five-rank open-ended continued pretraining; no publication side effects."""
import argparse,copy,gzip,hashlib,json,math,os,signal,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten,tree_map,tree_unflatten
from sml_v1.common import atomic_json,fingerprint,file_sha256
from sml_v1.checkpoint_bundle import resolve_bundle,save_bundle
from sml_v1.jaccl_control import ControlChannel
from sml_v1.model import TransformerLM,TransformerConfig
from sml_v1.precision import MasterAdamW
from sml_v1.pretrain import initialize,state_digest,clip_gradients
from sml_v1.stream import PrefetchedStream
from sml_v1.tokenization import Tokenizer
from .common import PLAN,MIX,OLD_WEIGHTS,NEW_WEIGHTS,BASE_SHA,BASE_OPT_SHA,lr
from .stream import MixedStream

def weighted_backward(model,xs,ys,total_tokens,world,one=None):
    """Global token-mean gradient, including unequal ranks and masked rows."""
    one=one or nn.value_and_grad(model,lambda x,y:model(x,targets=y)['loss'])
    total=None;loss_sum=0.
    for micro in range(xs.shape[0]):
        n=int(np.count_nonzero(ys[micro]!=-100))
        if not n:raise ValueError('Entirely masked microbatch')
        loss,g=one(mx.array(xs[micro].copy()),mx.array(ys[micro].copy()))
        weight=world*n/total_tokens
        g=tree_map(lambda t:t.astype(mx.float32)*weight,g)
        total=g if total is None else tree_map(lambda a,b:a+b,total,g)
        mx.eval(loss,total);loss_sum+=float(loss.item())*n
    return loss_sum,total

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--job-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=0);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();source=a.input;output=a.output;job=a.job_dir
    recipe=json.loads((source/'config.json').read_text());assert recipe['plan']==PLAN
    if not a.smoke and a.steps:raise ValueError('Production continuation has no bounded step stop')
    if not a.smoke and not os.environ.get('MLX_JACCL_RING')=='1':raise ValueError('Use the five-Mac launcher')
    # The job-scoped SSH transport waits for this record before opening reverse
    # coordinator tunnels. Writing it after init would deadlock peer startup.
    atomic_json(job/f'worker_rank{os.environ.get("MLX_RANK","0")}.json',dict(pid=os.getpid(),output=str(job)))
    group=mx.distributed.init(backend='jaccl',strict=True);rank,world=group.rank(),group.size()
    assert world==5
    physical=[0,1,2,4,3][rank];mx.set_cache_limit(PLAN['cache_gib_by_mac'][physical]*2**30)
    control=ControlChannel(mx,group,ring_mode='native')
    def root_call(label,fn):
        value=None;error=None
        if rank==0:
            try:value=fn()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'[:1500]
        status=control.broadcast(dict(error=error) if rank==0 else None,label)
        if status['error']:raise RuntimeError(status['error'])
        return value
    contract=recipe['contract']
    identity=fingerprint(contract)
    if any(c!=identity for c in control.exchange(identity,'input-contract')):raise ValueError('Worker contracts differ')
    weights=resolve_bundle(source/'checkpoint');metadata=json.loads(Path(weights+'.json').read_text())
    resumed=metadata.get('cpt151',{}).get('format')==PLAN['format']
    if resumed:
        if metadata['cpt151']['contract']!=contract:raise ValueError('Saved experiment contract changed')
        origin=metadata['parent_metadata'];added=metadata['cpt151']['added_tokens'];updates=metadata['cpt151']['updates']
        best=metadata['best_val_loss']
    else:
        if file_sha256(weights)!=BASE_SHA or file_sha256(weights+'.optimizer.safetensors')!=BASE_OPT_SHA:
            raise ValueError('Wrong pretrained starting checkpoint')
        if metadata['step']!=132005:raise ValueError('Unexpected base step')
        origin=metadata;added=updates=0;best=None
    with gzip.open(weights+'.rank0.data_state.json.gz','rt') as f:saved=json.load(f)
    assert saved['step']==metadata['step']
    mixed_state=saved['stream_state'] if resumed else None
    base_cursor=None if resumed else saved['stream_state']
    if resumed:assert sum(mixed_state['offsets'].values())==added
    model=TransformerLM(TransformerConfig(**origin['recipe']['model']));initialize(model,origin['recipe']);model.load_weights(weights)
    optimizer=MasterAdamW(learning_rate=PLAN['start_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=PLAN['weight_decay'])
    optimizer.init(model.trainable_parameters());optimizer.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(model.parameters(),optimizer.state)
    assert int(optimizer.state['step'].item())==metadata['step'],'Optimizer step reset'
    if state_digest(model,optimizer)!=metadata['replica_hash']:raise ValueError('Loaded model/optimizer differs from saved state')
    if any(v.dtype!=mx.bfloat16 for _,v in tree_flatten(model.parameters())):raise ValueError('Expected BF16 compute')
    tokenizer=Tokenizer(source/'tokenizer');assert tokenizer.fingerprint==contract['tokenizer_sha256']
    guard=set(json.loads((source/'exclusion-guard.json').read_text())['source_keys'])
    corpus=json.loads((source/'corpus.json').read_text())
    stream=root_call('restore-data',lambda:PrefetchedStream(MixedStream(corpus,origin['recipe'],tokenizer,base_cursor,guard,mixed_state),2))
    panel=root_call('load-validation',lambda:np.load(source/'validation.npz',allow_pickle=False))
    step=metadata['step'];tokens=metadata['tokens'];ema=metadata.get('ema');saved_step=-1
    heartbeat=[time.monotonic()];finished=threading.Event()
    def watchdog():
        while not finished.wait(10):
            if time.monotonic()-heartbeat[0]>1800:os._exit(70)
    threading.Thread(target=watchdog,daemon=True).start()
    def stop(sig,frame):
        if rank==0:(output/'STOP').touch()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    one=mx.compile(nn.value_and_grad(model,lambda x,y:model(x,targets=y)['loss']),inputs=[model.state],outputs=[model.state])
    def evaluate():
        losses={};model.eval()
        try:
            for label in MIX:
                xs,ys=panel[label+'_x'],panel[label+'_y'];numerator=denominator=0
                count=1 if a.smoke else len(xs)
                for x,y in zip(xs[:count],ys[:count]):
                    n=int(np.count_nonzero(y!=-100));loss=float(model(mx.array(x[None]),targets=mx.array(y[None]))['loss'].item())
                    numerator+=loss*n;denominator+=n
                losses[label]=numerator/denominator
        finally:model.train()
        if not all(math.isfinite(v) for v in losses.values()):raise ValueError('Nonfinite validation')
        return dict(sources=losses,general_nll=sum(losses[k]*w for k,w in OLD_WEIGHTS.items())/100,
                    structured_nll=sum(losses[k]*w for k,w in NEW_WEIGHTS.items())/100,
                    scope='Small fixed next-token-loss panels; not extraction accuracy or historical validation loss.')
    def save(is_best=False):
        nonlocal saved_step
        digest=state_digest(model,optimizer)
        if len(set(control.exchange(digest,'replica-state')))!=1:raise RuntimeError('Replica state mismatch: checkpoint refused')
        meta=dict(step=step,tokens=tokens,ema=ema,best_val_loss=best,replica_hash=digest,
                  recipe=dict(model=origin['recipe']['model'],name='cpt151-structured-open-ended',**{k:PLAN[k] for k in ['batches','grad_accum','betas','eps','weight_decay']}),
                  parent_metadata=origin,cpt151=dict(format=PLAN['format'],contract=contract,added_tokens=added,updates=updates,plan=PLAN),
                  validation_fingerprint=recipe['validation_sha256'])
        path=root_call('commit-checkpoint',lambda:save_bundle(output,model,optimizer,meta,stream.state_dict(),
                      best=is_best,keep=2**31-1,reserve_gib=16))
        if rank==0:
            atomic_json(output/'progress.json',dict(status='active',step=step,tokens=tokens,added_tokens=added,updates=updates,checkpoint=path,
                        source_tokens=stream.state_dict()['offsets']))
            print(f'[checkpoint] step={step} added_tokens={added:,} {path}',flush=True)
        saved_step=step
    start_updates=updates;start_added=added;started=time.monotonic()
    try:
        if not resumed:
            result=control.broadcast(root_call('baseline-eval',evaluate),'baseline-result')
            best=(result['general_nll']+result['structured_nll'])/2
            root_call('record-baseline',lambda:atomic_json(output/'baseline_eval.json',dict(step=step,added_tokens=added,**result)))
            save(True)
        print(f'[ready] rank={rank} step={step} added_tokens={added:,}; no token/time stop; Ctrl+C requests save',flush=True)
        model.train()
        while True:
            heartbeat[0]=time.monotonic()
            if control.broadcast((output/'STOP').exists() if rank==0 else None,'stop'):break
            if a.smoke and updates-start_updates>=a.steps:break
            full=root_call('next-batch',lambda:stream.global_batch(PLAN['grad_accum'],PLAN['batches']))
            shape=(2,PLAN['grad_accum'],sum(PLAN['batches']),PLAN['context'])
            digest=control.broadcast(hashlib.sha256(full.tobytes()).hexdigest() if rank==0 else None,'batch-hash')
            with mx.stream(mx.cpu):
                wire=mx.array(full) if rank==0 else mx.zeros(shape,dtype=mx.int32)
                wire=mx.distributed.all_sum(wire,group=group,stream=mx.cpu);mx.eval(wire);array=np.asarray(wire)
            if not all(control.exchange(hashlib.sha256(array.tobytes()).hexdigest()==digest,'verify-batch')):raise RuntimeError('Batch transfer corrupt')
            n=int(np.count_nonzero(array[1]!=-100));begin=sum(PLAN['batches'][:rank]);end=begin+PLAN['batches'][rank]
            local_loss,grads=weighted_backward(model,array[0,:,begin:end],array[1,:,begin:end],n,world,one)
            values=control.exchange(local_loss,'loss-sums')
            if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite loss; no update')
            grads=nn.average_gradients(grads,group=group,all_reduce_size=32*1024**2,communication_stream=mx.cpu)
            grads,norm=clip_gradients(grads,PLAN['clip']);mx.eval(grads,norm)
            if not all(control.exchange(bool(mx.isfinite(norm).item()),'finite-gradient')):raise ValueError('Nonfinite gradient; no update')
            optimizer.learning_rate=lr(added+n);optimizer.update(model,grads);mx.eval(model.parameters(),optimizer.state)
            before=added;added+=n;tokens+=n;step+=1;updates+=1
            value=sum(values)/n;ema=value if ema is None else .98*ema+.02*value
            if rank==0 and (updates==start_updates+1 or updates%PLAN['log_every']==0):
                progress=dict(status='active',step=step,added_tokens=added,updates=updates,loss=value,ema=ema,lr=lr(added),
                              grad_norm=float(norm.item()),real_tokens_per_second=(added-start_added)/(time.monotonic()-started),
                              source_tokens=stream.state_dict()['offsets'])
                atomic_json(output/'progress.json',progress);print('[train] '+json.dumps(progress),flush=True)
            due=before//PLAN['eval_every_tokens']!=added//PLAN['eval_every_tokens']
            is_best=False
            if due:
                result=control.broadcast(root_call('evaluate',evaluate),'evaluation-result')
                score=(result['general_nll']+result['structured_nll'])/2;is_best=score<best
                if is_best:best=score
                def record():
                    v=dict(step=step,added_tokens=added,updates=updates,**result)
                    atomic_json(output/'last_eval.json',v)
                    with (output/'evaluations.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
                    print('[eval] '+json.dumps(v),flush=True)
                root_call('record-eval',record)
            if due or before//PLAN['save_every_tokens']!=added//PLAN['save_every_tokens']:save(is_best)
        if saved_step!=step:save()
        root_call('finish',lambda:atomic_json(job/'result.json',dict(status='stopped',step=step,tokens=tokens,added_tokens=added,updates=updates)))
        root_call('finish-progress',lambda:atomic_json(output/'progress.json',dict(status='stopped',step=step,tokens=tokens,added_tokens=added,updates=updates)))
        print(f'[finished] rank={rank} step={step} checkpoint committed',flush=True)
    finally:
        finished.set()
        if rank==0:
            stream.close();panel.close()

if __name__=='__main__':
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
