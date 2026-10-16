"""Continue the requested SFT-128 replay, preserving every scheduled checkpoint.
Original code/data/optimizer settings are frozen; this process overrides storage retention only.
A replay with different weight bytes is reported as a replay, never an exact restored original.
"""
import json
from pathlib import Path
import mlx.core as mx
from native_json_128.common import *
from native_json_128.train import contract,execute
from native_json_128.runtime import model_at,Encoding
from native_json_128.evaluate import evaluate,compare
from sml_v1.common import atomic_json,lock
import sml_v1.checkpoint_bundle as bundles

OUTPUT=ROOT/'pretraining/runs/sft_128_no_prune_20261016'


def main():
 c=contract()
 if c!=json.loads((RUN/'session.json').read_text()):raise ValueError('Original training contract changed')
 policy=dict(pruning=False,preserve_all_scheduled_checkpoints=True,original_training_contract=c,runner_sha256=sha(Path(__file__)),output=str(OUTPUT),scope='Original numerical training recipe; storage policy overridden to retain every checkpoint. This is a replay because update100 did not match original weight bytes. No public promotion.')
 session=OUTPUT/'no-prune-session.json'
 if session.exists() and json.loads(session.read_text())!=policy:raise ValueError('Replay policy changed')
 if (OUTPUT/'no-prune-result.json').exists() and json.loads((OUTPUT/'no-prune-result.json').read_text())['status']=='complete':print((OUTPUT/'no-prune-result.json').read_text());return
 with lock(LOCK):
  from json_sampling_119.run import active_workers
  if active_workers():raise RuntimeError('Another model worker active')
  OUTPUT.mkdir(parents=True,exist_ok=True);atomic_json(session,policy)
  original_save=bundles.save_bundle
  def save_all(root,model,optimizer,metadata,stream_state,**kwargs):
   before={p.name for p in Path(root).glob('step_*') if p.is_dir()};kwargs['keep']=1000000000
   metadata=dict(metadata,preservation_policy=dict(pruning=False,user_requested=True))
   result=original_save(root,model,optimizer,metadata,stream_state,**kwargs)
   after={p.name for p in Path(root).glob('step_*') if p.is_dir()}
   if not before<=after:raise RuntimeError('No-pruning invariant violated')
   print('[json128 preserve all]',len(after),'checkpoint bundles',flush=True);return result
  bundles.save_bundle=save_all
  baseline=json.loads((REPORT/'parent-development.json').read_text())
  def assess(model,step,bundle):
   weight_sha=sha(bundle/'model.safetensors');path=OUTPUT/'evaluations'/f'development_{step:07d}.json';result=evaluate(model,load(DATA/'development.jsonl'),path,step=step,weight_sha256=weight_sha);gate=compare(result,baseline,compact=True);result['retention_vs_parent']=gate;atomic_json(path,result)
   archived_path=RUN/'evaluations'/f'development_{step:07d}.json';archived=json.loads(archived_path.read_text());archived_identity=json.loads(archived_path.with_suffix('.identity.json').read_text())
   matches=sum(a['text']==b['text'] and a['stop_reason']==b['stop_reason'] for a,b in zip(result['outputs'],archived['outputs']))
   identity=dict(policy_sha256=__import__('hashlib').sha256(dumps(policy).encode()).hexdigest(),weight_sha256=weight_sha,step=step,original_weight_sha256=archived_identity['model_sha256'],matches_original_weight_bytes=weight_sha==archived_identity['model_sha256'],same_generated_answers=matches,total=240)
   atomic_json(path.with_suffix('.identity.json'),identity)
   metrics=result['metrics'];rank=[metrics['all']['counts']['exact'],metrics['all']['field_accuracy'],metrics['all']['f1']];best_path=OUTPUT/'best-overall-metrics.json';prior=json.loads(best_path.read_text()) if best_path.exists() else None
   if prior is None or rank>prior['selection']:
    atomic_json(best_path,dict(step=step,bundle=bundle.name,selection=rank,retention_gate_passed=gate['passed']));atomic_json(OUTPUT/'best-overall.json',dict(format='sml-pretrain-bundle-v1',bundle=bundle.name))
   atomic_json(OUTPUT/'last_eval.json',{k:v for k,v in result.items() if k!='outputs'})
   print('[json128 no-prune evaluation]',dumps(dict(step=step,correct=rank[0],count=240,legacy_correct=metrics['panel/legacy']['counts']['exact'],new_correct=metrics['panel/new_prose']['counts']['exact'],retention_passed=gate['passed'],lost=len(gate['lost_previously_correct']),same_original_answers=matches,original_weight_bytes=identity['matches_original_weight_bytes'])),flush=True)
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);mx.random.seed(PLAN['seed']);enc=Encoding();encoded=[enc.row(r) for r in load(DATA/'train.jsonl')];model=model_at();execute(model,encoded,OUTPUT,c,resume=(OUTPUT/'latest.json').exists(),assessor=assess)
  state=json.loads((OUTPUT/'result.json').read_text())
  if state['status']!='sft_complete':print(dumps(state),flush=True);return
  verified=[]
  for step in [100,200,300,331]:
   paths=list(OUTPUT.glob(f'step_{step:07d}_*'))
   if len(paths)!=1:raise ValueError('Missing/duplicate preserved checkpoint')
   weight=Path(bundles.resolve_bundle(paths[0]));verified.append(dict(step=step,bundle=paths[0].name,weight_sha256=sha(weight)))
  if contract()!=c or sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Original inputs changed')
  result=dict(status='complete',training=state,pruning=False,preserve_all=True,checkpoints=verified,best_overall=json.loads((OUTPUT/'best-overall-metrics.json').read_text()),parent_unchanged=True,public_models_and_playground_unchanged=True,automatic_promotion=False,scope=policy['scope'])
  atomic_json(OUTPUT/'KEEP.json',dict(preserve=True,user_requested=True,pruning=False,checkpoints=verified));atomic_json(OUTPUT/'no-prune-result.json',result);atomic_json(REPORT/'no-prune-replay.json',result);print('[json128 no-prune complete]',dumps(result),flush=True)
if __name__=='__main__':main()
