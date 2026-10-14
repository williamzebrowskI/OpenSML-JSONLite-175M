"""Recreate only093/160 and retain it at the user's explicit request."""
import gc,json,shutil
from pathlib import Path
from unittest.mock import patch
import mlx.core as mx
from json_approaches_090 import train as trainer
from json_approaches_090.data import *
from json_approaches_090.evaluate import controls
from extract_068.train import load_model,active_workers
from sml_v1.tokenization import Tokenizer
from sml_v1.common import atomic_json,lock

def main():
 tok=Tokenizer(TOK);assert not runpath(93).exists() and not datapath(93).exists()
 manifest=prepare(93,tok);c=trainer.contract()
 prior=json.loads((REPORT/'loop.json').read_text())['contract']
 assert c==prior,'Recreation recipe/input/code identity changed'
 original_rmtree=shutil.rmtree
 def retain_requested(path,*args,**kwargs):
  if Path(path).resolve() in {runpath(93).resolve(),datapath(93).resolve()}:return
  return original_rmtree(path,*args,**kwargs)
 def document(result):
  result.update(recreated_at_user_request=True,preserved_at_user_request=True,decision='Retained user-selected diagnostic; original correctness/retention gates still reported',artifact_policy='Retain update160 and its reproduction recipe/evaluation at user request; no qualified or release-ready claim.')
  with DOCUMENT.open('a') as f:
   f.write('\n\n### SFT-093/update160 user-requested recreation for the playground (2026-10-14)\n\nSame parent, source rows, order, optimizer, training settings and evaluator as the original trial; the contract matches byte-for-byte. The user requested recreation/preservation despite retention failures. Numeric weights and actual generated scores are checked after recreation; bitwise GPU reproducibility is measured, not assumed.\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
 with lock(ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'):
  assert not active_workers(),'Another training worker is active'
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3)
  model=load_model(PARENT/'model.safetensors');base=controls(model,tok,'development');del model;gc.collect();mx.clear_cache()
  with patch.object(trainer,'write_summary',document),patch.object(trainer.shutil,'rmtree',retain_requested):
   trainer.train_one(93,tok,c,base)
  out=runpath(93);r=json.loads((out/'result.json').read_text());bundles=list(out.glob('step_0000160_*'));assert len(bundles)==1
  bundle=bundles[0];metrics=json.loads((out/'evaluations/development_0000160.json').read_text())
  actual_sha=file_sha(bundle/'model.safetensors')
  expected=json.loads((REPORT/'SFT-093_update160_all_metrics.json').read_text())
  receipt=dict(checkpoint=str(bundle),step=160,model_sha256=actual_sha,original_weight_sha256=expected['weight_sha256'],bitwise_recreated=actual_sha==expected['weight_sha256'],original_json_correct=167,recreated_json_correct=metrics['metrics']['json']['all']['counts']['exact'],development_requests=196,retained_at_user_request=True,qualified=metrics['readiness']['passed'],release_ready=False)
  atomic_json(out/'best.json',dict(format='sml-pretrain-bundle-v1',bundle=bundle.name));atomic_json(out/'latest.json',dict(format='sml-pretrain-bundle-v1',bundle=bundle.name));atomic_json(out/'user-selected.json',receipt)
  for b in out.glob('step_*_*'):
   if b!=bundle:original_rmtree(b)
  atomic_json(REPORT/'recreated-093-playground.json',receipt)
  with DOCUMENT.open('a') as f:f.write('\nRecreation verification:\n\n```json\n'+json.dumps(receipt,indent=2)+'\n```\n')
  print('[recreated093]',json.dumps(receipt),flush=True)

if __name__=='__main__':main()
