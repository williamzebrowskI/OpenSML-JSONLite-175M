"""Sequential Mac1 representation experiments; no automatic model promotion."""
import argparse,collections,gc,hashlib,json,math,os,shutil,signal,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .data import *
from .evaluate import *
from json_sft.corpus import file_sha
from sml_v1.common import lock,atomic_json

def training_batches():
 rng=np.random.default_rng(PLAN['seed']);a=rng.permutation(2560);b=rng.permutation(2560)+2560;out=[]
 for i in range(160):
  x=np.concatenate([a[i*16:(i+1)*16],b[i*16:(i+1)*16]]);rng.shuffle(x);out.append(x)
 assert sorted(int(x) for batch in out for x in batch)==list(range(5120))
 return out

def accumulate(one,corpus,indices):
 import mlx.core as mx
 from mlx.utils import tree_map
 from json_evidence_binding.runtime import microbatches
 total=None;value=0.
 for piece in microbatches(corpus,indices,policy=PLAN['padding_policy'],size=64,microbatch=2):
  if not piece.actual_records:continue
  v,g=one(mx.array(piece.inputs),mx.array(piece.labels));g=tree_map(lambda v:v.astype(mx.float32),g)
  total=g if total is None else tree_map(lambda a,b:a+b,total,g);mx.eval(total,v);value+=float(v.item())
 total=tree_map(lambda v:v/len(indices),total);mx.eval(total)
 return value/len(indices),total

def contract():
 import importlib.metadata
 paths=[]
 for name in ('json_approaches_090','sml_v1','sft','extract_068','json_evidence_binding','qa_065','qa_066','qa_semantics'):
  paths+=list((ROOT/'pretraining'/name).glob('*.py'))
 inputs=[SNIPS/'train.jsonl',SNIPS/'development.jsonl',SNIPS/'test.jsonl',QA/'development.jsonl',QA/'test.jsonl']+list(SOURCE.glob('*'))+list(TOK.glob('*'))
 return dict(plan=PLAN,parent_sha256=file_sha(PARENT/'model.safetensors'),parent_metadata_sha256=file_sha(PARENT/'model.safetensors.json'),code={str(p.relative_to(ROOT)):file_sha(p) for p in sorted(paths)},inputs={str(p.relative_to(ROOT)):file_sha(p) for p in inputs if p.is_file()},versions={n:importlib.metadata.version(n) for n in ('mlx','numpy','tokenizers','jsonschema')})

def verify():
 from sml_v1.tokenization import Tokenizer
 from sft.data import Corpus,serialize
 from .evaluate import decode,json_score
 assert file_sha(PARENT/'model.safetensors')==PARENT_SHA
 assert json.loads((PARENT/'model.safetensors.json').read_text())['step']==379
 REPORT.mkdir(parents=True,exist_ok=True);tok=Tokenizer(TOK);train,demos=sources();receipts=[]
 for n in APPROACHES:
  m=prepare(n,tok) if not datapath(n).exists() else json.loads((datapath(n)/'manifest.json').read_text())
  for k,h in m['files'].items():assert file_sha(datapath(n)/k)==h
  c=Corpus(datapath(n),'train');rs=load(datapath(n)/'train.jsonl');assert len(c)==len(rs)==5120
  for i,r in enumerate(rs):
   a,b=serialize(tok,r['messages'])
   if n==99 and r['kind']=='target':p,_=serialize(tok,r['messages'][:-1],generation=True);b[:len(p)]=False
   lo,hi=c.offsets[i:i+2];assert np.array_equal(a,c.ids[lo:hi]) and np.array_equal(b,c.mask[lo:hi])
  for r in load(SNIPS/'development.jsonl')+load(SNIPS/'test.jsonl'):
   requests=[singleton(r,f) for f in r['fields']] if n in (93,95) else [r];combined={}
   for rr in requests:
    ms=messages(rr,n,demos);p,_=serialize(tok,ms[:-1],generation=True);assert len(p)+160<=2048
    combined.update(decode(rr,n,ms[-1]['content']))
   assert combined==r['expected']
   assert json_score(r,dumps(combined),'turn_end')['exact']==1
   assert not json_score(r,dumps(combined),'token_limit')['exact']
  receipts.append(m)
  print('[preflight data]',n,APPROACHES[n],m['max_sequence'],m['target_present_fields'],m['target_absent_fields'],flush=True)
 # Bad structures must not be accepted or silently repaired.
 r=train[0]
 for text in ('{}','null','[]','{"x":NaN}','{"x":[],"x":[]}'):
  assert not json_score(r,text,'turn_end')['exact']
 assert len(training_batches())==160
 from .verify import checks
 checks(tok)
 frozen=contract();assert frozen['parent_sha256']==PARENT_SHA
 atomic_json(REPORT/'preflight.json',dict(passed=True,contract=frozen,data=receipts,scope='Raw source/target reconstruction, source-disjoint splits, serializer/mask/bounds, gold representation round trips and strict scoring. Not a model quality claim.'))
 return tok,frozen

def write_summary(result):
 with DOCUMENT.open('a') as f:
  f.write(f'\n\n### SFT-{result["approach"]:03d} representation-loop result (2026-10-14)\n\n')
  f.write('All weights independently fork068/379; fresh full-weight FP32-master AdamW onMac1, batch32/micro2,160updates, one pass over2560originalTRAINqueries in2560labeled replay+2560target views; peak5e-6/final1e-6,warmup12,clip1,wd0.01,betas(0.9,0.95). Fixed observed development196JSON+196original labeled+136QA+32TRAINFIT; checks0/80/160. No test-derived training targets, no synthetic factual substitutions. Fieldwise views have fewer requested fields in TRAIN; generation assembles all requested fields at inference. All final answers scored against the same source/gold; intermediate formats and model-call counts differ.\n\n')
  f.write('```json\n'+json.dumps(result,indent=2)+'\n```\n')

def train_one(n,tok,c,base_controls):
 import mlx.core as mx
 import mlx.nn as nn
 from mlx.utils import tree_flatten,tree_map
 from sml_v1.precision import MasterAdamW
 from sml_v1.checkpoint_bundle import save_bundle
 from extract_068.train import load_model,lr_for
 from sft.data import Corpus
 from sft.evaluate import loss_sum
 out=runpath(n);assert not out.exists();out.mkdir(parents=True);atomic_json(out/'session.json',dict(contract=c,approach=n,data_manifest=file_sha(datapath(n)/'manifest.json')))
 train,demos=sources();model=load_model(PARENT/'model.safetensors');mx.random.seed(PLAN['seed']);corpus=Corpus(datapath(n),'train')
 initial=evaluate(model,tok,n,0,demos,train,base_controls,None,out/'evaluations/development_0000000.json')
 opt=MasterAdamW(learning_rate=PLAN['peak_lr'],betas=tuple(PLAN['betas']),eps=PLAN['eps'],weight_decay=PLAN['weight_decay']);opt.init(model.trainable_parameters());mx.eval(opt.state)
 one=nn.value_and_grad(model,lambda x,y:loss_sum(model,x,y));order=training_batches();records=0;tokens=0;began=time.monotonic();evaluations=[initial];bundles={};stopped=[False]
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
 for step,batch in enumerate(order,1):
  if stopped[0] or (REPORT/'STOP').exists():raise InterruptedError('Loop stop requested; keep partial artifacts')
  model.train();value,g=accumulate(one,corpus,batch);norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));mx.eval(norm)
  assert math.isfinite(value) and math.isfinite(norm.item()),'nonfinite gradient'
  g=tree_map(lambda v:v*mx.minimum(1.,PLAN['grad_clip']/(norm+1e-6)),g);opt.learning_rate=lr_for(step,160,PLAN);opt.update(model,g);mx.eval(model.parameters(),opt.state)
  records+=len(batch);tokens+=int(sum(corpus.lengths[batch]+1))
  if step==1 or step%20==0:print(f'[SFT-{n:03d} {step}/160] loss={value:.5f} lr={lr_for(step,160,PLAN):.3e} records={records} tokens={tokens} elapsed={time.monotonic()-began:.1f}s',flush=True)
  if step in (80,160):
   meta=dict(step=step,tokens=tokens,recipe={'model':asdict(model.cfg)},extraction_sft=dict(contract=c,approach=n,records=records,total_steps=160))
   bundle=Path(save_bundle(out,model,opt,meta,{'next_batch':step},keep=2,reserve_gib=8));bundles[step]=bundle
   e=evaluate(model,tok,n,step,demos,train,base_controls,initial['metrics'],out/f'evaluations/development_{step:07d}.json');e['weight_sha256']=file_sha(bundle/'model.safetensors');evaluations.append(e)
 qualified=[e for e in evaluations[1:] if e['readiness']['passed']]
 diagnostic=max(qualified or evaluations[1:],key=lambda e:(e['metrics']['json']['all']['counts']['exact'],e['metrics']['json']['all']['f1'],-e['metrics']['json']['all']['counts']['unsupported_values']))
 selected=diagnostic['step'] if qualified else 0
 final=None;accepted=False
 if qualified:
  # Freeze the DEV choice before any full final-test generation.
  model.load_weights(str(bundles[selected]/'model.safetensors'))
  parent_model=load_model(PARENT/'model.safetensors');test_controls=controls(parent_model,tok,'test')
  parent_eval=evaluate(parent_model,tok,n,0,demos,train,test_controls,None,out/'final-parent.json',split='test')
  del parent_model;mx.clear_cache()
  final=evaluate(model,tok,n,selected,demos,train,test_controls,parent_eval['metrics'],out/'final-candidate.json',split='test');accepted=final['readiness']['passed']
 result=dict(approach=n,name=APPROACHES[n],parent=str(PARENT),parent_sha256=PARENT_SHA,status='complete',updates=160,records=records,tokens=tokens,baseline=initial['metrics'],evaluations=[{k:e[k] for k in ('step','metrics','readiness','protocol','seconds')} for e in evaluations[1:]],diagnostic_step=diagnostic['step'],diagnostic_weight_sha256=diagnostic['weight_sha256'],selected_step=selected,accepted=accepted,release_ready=False,final_test=None if final is None else {k:final[k] for k in ('metrics','readiness','protocol')},decision='Retained pending broader independent validation' if accepted else 'Rejected: no qualifying DEV checkpoint' if not qualified else 'Rejected: final-test gates failed',artifact_policy='Keep only qualified final-passing checkpoint; failed run weights/data/logs/evaluation outputs deleted after this report entry. Shared recipes retained for reproducibility; main Markdown is the narrative record.')
 write_summary(result);atomic_json(out/'result.json',result)
 del model,opt,one,g;gc.collect();mx.clear_cache()
 if accepted:
  atomic_json(out/'best.json',dict(format='sml-pretrain-bundle-v1',bundle=bundles[selected].name))
  for bundle in bundles.values():
   if bundle!=bundles[selected]:shutil.rmtree(bundle)
  # latest must resolve to retained weights after pruning.
  atomic_json(out/'latest.json',dict(format='sml-pretrain-bundle-v1',bundle=bundles[selected].name))
 else:
  shutil.rmtree(out);shutil.rmtree(datapath(n))
 assert file_sha(PARENT/'model.safetensors')==PARENT_SHA
 return dict(approach=n,name=APPROACHES[n],accepted=accepted,diagnostic_step=diagnostic['step'],json_correct=diagnostic['metrics']['json']['all']['counts']['exact'],json_count=196,baseline_json_correct=initial['metrics']['json']['all']['counts']['exact'],labeled_correct=diagnostic['metrics']['labeled']['all']['counts']['exact'],qa_correct=diagnostic['metrics']['qa']['all']['counts']['exact'],failed_gates=diagnostic['readiness']['failed_gates'],model_calls=diagnostic['protocol']['model_calls'],run_directory=str(out) if accepted else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
 if not (a.verify or a.run):p.error('choose --verify or --run')
 tok,c=verify()
 if not a.run:return
 from extract_068.train import active_workers,load_model
 import mlx.core as mx
 with lock(ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'):
  assert not active_workers(),'another training process active'
  mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3)
  m=load_model(PARENT/'model.safetensors');base=controls(m,tok,'development');del m;gc.collect();mx.clear_cache()
  ledger=REPORT/'loop.json';assert not ledger.exists(),'Loop already initialized'
  results=[];atomic_json(ledger,dict(status='running',results=results,contract=c))
  with DOCUMENT.open('a') as f:
   f.write('\n\n### SFT-090–099 predeclared representation experiment loop (2026-10-14)\n\nTen independentMac1 trials from preserved068/379. Different representations/input protocols, fixed facts/settings/evaluator. Strategies:'+dumps(APPROACHES)+'. Explicit assemblers operate only on generated strings or source offsets, never on expected answers. New routes are evaluated at parent baseline; native labeled/QA controls are generated once and rescored under matching identities. Gates retain previous per-source accuracy/error criteria, require JSON extraction at least as good as original labeled parent and no regression from the same route baseline; TRAIN FIT>=29/32. Full final1388JSON+1388labeled+272QA comparison only for DEV-qualified trials. Observed panels are not fresh blind release benchmarks; narrower SNIPS conventions remain a limitation. All metric groups are printed at scheduled0/80/160checks.\n')
  for n in APPROACHES:
   assert contract()==c,'Frozen code/data changed during loop'
   result=train_one(n,tok,c,base);results.append(result);atomic_json(ledger,dict(status='running',results=results,contract=c));print('[loop completed]',dumps(result),flush=True)
  atomic_json(ledger,dict(status='complete',results=results,contract=c))
  with DOCUMENT.open('a') as f:
   f.write('\n\n### SFT-090–099 completed-loop summary (2026-10-14)\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n\n1,600completed updates and51,200training presentations across10independent forks, with overlapping2560sourcequeries. All retained models require broader validation; syntax-only success is not extraction correctness. Failed trial artifacts deleted after documentation; shared recipe/evaluator/preflight and compact ledger remain. Parent068/379 SHA verified unchanged after every trial. Playground unchanged.\n')
  print('[loop final]',dumps(results),flush=True)

if __name__=='__main__':main()
