"""64 TRAIN requests, greedy plus eight samples each; inference only and resumable."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from sml_v1.common import atomic_json,fingerprint,lock
from .data import ROOT,DATA,REPORT,OUTPUT,PARENT,PARENT_SHA,TOK,PLAN,check,load,sha
from .scoring import score
from .reporting import summarize,markdown

def sample_seed(identifier,index):
 raw=f"{PLAN['seed']}:{identifier}:{index}".encode()
 return int.from_bytes(hashlib.sha256(raw).digest()[:4],'little')&0x7fffffff

def contract():
 check()
 paths=list((ROOT/'pretraining/json_sampling_119').glob('*.py'))+list(TOK.glob('*'))
 paths += [DATA/'cases.jsonl',DATA/'manifest.json',REPORT/'source-validation.json',REPORT/'curation-review.json',ROOT/'pretraining/json_sampling_119/requirements.txt',ROOT/'pretraining/scripts/run_json_sampling_audit_119.sh',ROOT/'pretraining/tests/test_json_sampling_119.py',PARENT/'model.safetensors.json']
 paths += [ROOT/('pretraining/'+s) for s in ['sml_v1/model.py','sml_v1/tokenization.py','sml_v1/fast_loss.py','sml_v1/fast_swiglu.py','sml_v1/common.py','sft/evaluate.py','sft/data.py','sft/quality.py']]
 return dict(parent_sha256=PARENT_SHA,plan=PLAN,files={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()},versions={n:importlib.metadata.version(n) for n in ['mlx','mlx-metal','numpy','tokenizers','jsonschema']})

def active_workers():
 lines=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines()
 return [s for s in lines if int(s.split(None,1)[0])!=os.getpid() and re.search(r'python\S*\s+(?:-u\s+)?-m\s+(?:native_json_\w+|extract_\w+|qa_\w+|json_\w+|sml_v1|sft)\.(?:train|run|pretrain)',s)]

def checked_draw(row,raw,temperature,seed):
 return dict(id=row['id'],text=raw['text'],tokens=raw['tokens'],stop_reason=raw['stop_reason'],temperature=temperature,seed=seed,checks=score(row,raw['text'],raw['stop_reason']))

def validate_record(row,record,identity):
 if record['id']!=row['id'] or record['contract_sha256']!=identity or len(record['samples'])>PLAN['samples_per_request']:raise ValueError('Stale/corrupt audit request identity')
 draws=[]
 if record.get('greedy') is not None:draws.append((record['greedy'],0.,sample_seed(row['id'],-1)))
 elif record['samples']:raise ValueError('Samples without greedy baseline')
 draws += [(o,PLAN['temperature'],sample_seed(row['id'],j)) for j,o in enumerate(record['samples'])]
 for o,t,s in draws:
  if o['id']!=row['id'] or o['temperature']!=t or o['seed']!=s or o['checks']!=score(row,o['text'],o['stop_reason']):raise ValueError('Saved prediction/scoring/seed mismatch')
 if record['complete']!=(record.get('greedy') is not None and len(record['samples'])==PLAN['samples_per_request']):raise ValueError('Saved completion mismatch')
 return record

def audit_case(row,path,identity,generator,stop):
 path=Path(path)
 if path.exists():record=validate_record(row,json.loads(path.read_text()),identity)
 else:record=dict(id=row['id'],audit_stratum=row['audit_stratum'],contract_sha256=identity,greedy=None,samples=[],complete=False)
 def save():
  record['complete']=record['greedy'] is not None and len(record['samples'])==PLAN['samples_per_request'];atomic_json(path,record)
 if stop():return record
 if record['greedy'] is None:
  seed=sample_seed(row['id'],-1);raw=generator(row['messages'][:-1],0.,seed);record['greedy']=checked_draw(row,raw,0.,seed);save()
 while len(record['samples'])<PLAN['samples_per_request'] and not stop():
  index=len(record['samples']);seed=sample_seed(row['id'],index);raw=generator(row['messages'][:-1],PLAN['temperature'],seed);record['samples'].append(checked_draw(row,raw,PLAN['temperature'],seed));save()
 return record

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 if args.resume and not args.run:parser.error('--resume requires --run')
 c=contract();rows=load(DATA/'cases.jsonl')
 if not args.run:
  print(json.dumps(dict(audit_started=False,training=False,parent=str(PARENT),plan=PLAN,generations=64*9,results=str(OUTPUT),report=str(REPORT)),indent=2));return
 ready=json.loads((REPORT/'ready.json').read_text())
 if ready['status']!='passed' or ready['contract']!=c:raise ValueError('Readiness missing/stale; run json_sampling_119.verify')
 with lock(ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'):
  if active_workers():raise RuntimeError('Another model worker active; not touched')
  session=OUTPUT/'session.json';result_path=OUTPUT/'result.json'
  if session.exists():
   if json.loads(session.read_text())!=c:raise ValueError('Audit contract changed; refusing mixed results')
   if result_path.exists() and json.loads(result_path.read_text()).get('status')=='complete':print(result_path.read_text(),flush=True);return
   if not args.resume:raise ValueError('Partial audit exists; use --run --resume')
  elif args.resume:raise ValueError('No audit progress to resume')
  OUTPUT.mkdir(parents=True,exist_ok=True);atomic_json(session,c)
  import mlx.core as mx
  from .runtime import load_model,make_generator
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3);model=load_model();generator=make_generator(model);stop=[False];handlers={}
  for sig in [signal.SIGINT,signal.SIGTERM]:handlers[sig]=signal.signal(sig,lambda *_:stop.__setitem__(0,True))
  records=[];began=time.monotonic();identity=fingerprint(c)
  try:
   for i,row in enumerate(rows):
    path=OUTPUT/'requests'/f"{i:03d}_{hashlib.sha256(row['id'].encode()).hexdigest()[:10]}.json";record=audit_case(row,path,identity,generator,lambda:stop[0]);records.append(record)
    if not record['complete']:break
    correct=sum(o['checks']['exact'] for o in record['samples']);print(f"[json-audit119] {i+1}/64 {row['audit_stratum']} greedy={record['greedy']['checks']['exact']} sampled_exact={correct}/8 elapsed={time.monotonic()-began:.1f}s",flush=True)
   complete=len(records)==64 and all(x['complete'] for x in records)
   if not complete:
    draws=sum(int(r.get('greedy') is not None)+len(r['samples']) for r in records);atomic_json(result_path,dict(status='stopped',completed_requests=sum(r['complete'] for r in records),visited_request_draws=draws,optimizer_updates=0,resume='--run --resume'));return
   result=summarize(rows,records);result.update(status='complete',parent_sha256=PARENT_SHA,contract_sha256=identity,generations=576,optimizer_updates=0,checkpoints_created=0,elapsed_seconds_this_invocation=time.monotonic()-began)
   if sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Parent file changed')
   atomic_json(OUTPUT/'results.json',result);atomic_json(result_path,{k:v for k,v in result.items() if k not in ['greedy_metrics','sampled_draw_metrics']});(REPORT/'RESULTS.md').write_text(markdown(result));atomic_json(REPORT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['greedy_metrics','sampled_draw_metrics']},indent=2),flush=True)
  finally:
   for sig,handler in handlers.items():signal.signal(sig,handler)

if __name__=='__main__':main()
