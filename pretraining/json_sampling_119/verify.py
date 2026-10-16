"""Tests and four TRAIN-only inference smoke calls; no full audit or training."""
import json
import subprocess
import sys
import time
from pathlib import Path
from sml_v1.common import atomic_json,lock
from .data import ROOT,DATA,REPORT,OUTPUT,PARENT,PARENT_SHA,PLAN,check,load,sha,TOK
from .run import contract,active_workers,sample_seed,checked_draw

def main():
 began=time.monotonic();check(deep=True);curation=json.loads((REPORT/'curation-review.json').read_text())
 if curation['status']!='reviewed_before_generation' or curation['cases_sha256']!=sha(DATA/'cases.jsonl'):raise ValueError('Curation review missing/stale')
 tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'pretraining/tests'),'-p','test_json_sampling_119.py','-v'],capture_output=True,text=True);(REPORT/'tests.txt').write_text(tests.stdout+tests.stderr)
 if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
 with lock(ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'):
  if active_workers():raise RuntimeError('Another worker active; not touched')
  import mlx.core as mx
  from mlx.utils import tree_flatten
  from sft.data import serialize
  from sml_v1.tokenization import Tokenizer
  from .runtime import load_model,make_generator
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);model=load_model();before=dict(tree_flatten(model.parameters()));generator=make_generator(model);tok=Tokenizer(TOK);rows=load(DATA/'cases.jsonl');lengths=[]
  for row in rows:
   prefix,_=serialize(tok,row['messages'][:-1],generation=True);lengths.append(len(prefix))
   if len(prefix)+PLAN['max_tokens']>model.cfg.max_seq_len:raise ValueError('Would truncate generation budget')
  smoke=[];first=rows[0];second=next(r for r in rows if r['audit_stratum']=='multiple_values')
  for row in [first,second]:
   seed=sample_seed(row['id'],-1);raw=generator(row['messages'][:-1],0.,seed);smoke.append(checked_draw(row,raw,0.,seed))
  seed=sample_seed(first['id'],0);a=generator(first['messages'][:-1],PLAN['temperature'],seed);b=generator(first['messages'][:-1],PLAN['temperature'],seed)
  if a!=b:raise ValueError('Fixed-seed sampling not reproducible')
  smoke.extend([checked_draw(first,a,PLAN['temperature'],seed),checked_draw(first,b,PLAN['temperature'],seed)])
  if any(not isinstance(o['text'],str) or not 0<=o['tokens']<=PLAN['max_tokens'] for o in smoke):raise ValueError('Invalid generation record')
  if tree_flatten(model.trainable_parameters()) or not all(mx.array_equal(v,before[k]).item() for k,v in tree_flatten(model.parameters())) or sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Parent parameters changed')
  if OUTPUT.exists():raise ValueError('Preflight must not initialize full audit')
  atomic_json(REPORT/'preflight-inference.json',dict(parent_sha256=PARENT_SHA,training_only=True,optimizer_updates=0,full_audit_started=False,same_seed_identical=True,outputs=smoke))
  receipt=dict(status='passed',contract=contract(),tests_passed=12,smoke_generations=4,fixed_seed_reproducible=True,parent_parameters_unchanged=True,optimizer_updates=0,checkpoints_created=0,full_audit_started=False,max_prompt_tokens=max(lengths),full_audit_generations=576,elapsed_seconds=time.monotonic()-began)
  atomic_json(REPORT/'ready.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='contract'},indent=2),flush=True)

if __name__=='__main__':main()
