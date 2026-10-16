"""Freeze 64 TRAIN-only diagnostic inputs without consulting model predictions."""
import collections
import copy
import json
import re
from pathlib import Path
from jsonschema import Draft202012Validator
from .common import sha,load,dumps,prompt
from .scoring import strict_json
from sml_v1.common import atomic_json

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'pretraining/runs/sft_105_cardinality_distillation_20261014/step_0000064_cdd8be1e8d5a'
PARENT_SHA='655fb5cfba009163cd8a7f7fe7b170b998bb452c217b1b473cfe10b33de0ccd1'
TOK=ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
DATA=ROOT/'pretraining/data/json_sampling_audit_119_20261015'
OUTPUT=ROOT/'pretraining/runs/json_sampling_audit_119_20261015'
REPORT=ROOT/'reports/model-design/JSON-AUDIT-119_sampling_20261015'
ORIGINAL=ROOT/'pretraining/data/sft_100_native_json_20261014/train.jsonl'
AUTHORED=ROOT/'reports/model-design/SFT-111_attention_binding_20261015/retired-experiment/train.jsonl'
HELDOUT=ROOT/'reports/model-design/SFT-118_activation_subspace_20261015/retired-experiment'
SOURCE=ROOT/'pretraining/data/snips_sources_20261013'
PLAN=dict(seed=202610150119,requests=64,strata=['copying','multiple_values','missing_fields','role_assignment'],requests_per_stratum=16,samples_per_request=8,temperature=.7,max_tokens=256,repetition_penalty=1,top_k=None,top_p=None,serialization='native-sft-chat-v1',constrained_decoding=False,output_repair=False,json_assembly=False,optimizer_updates=0)

def normalized(text):return ' '.join(re.findall(r'\w+|[^\w\s]',text.casefold()))
def rank(row,label):
 import hashlib
 return hashlib.sha256(f"{PLAN['seed']}:{label}:{row['id']}".encode()).hexdigest()
def present_values(row):
 return [v for values in row['expected'].values() for v in (values if isinstance(values,list) else ([] if values is None else [values]))]

def select_cases():
 original=[r for r in load(ORIGINAL) if not any(k.removeprefix('out_')=='entity_name' for k in r['expected'])];authored=[r for r in load(AUTHORED) if r.get('category')=='roles']
 excluded={normalized(r['document']) for s in ['development','test'] for r in load(HELDOUT/(s+'.jsonl'))}
 used=set();chosen=[]
 def pick(pool,label,count):
  candidates=sorted(pool,key=lambda r:rank(r,label));found=[]
  for base in candidates:
   key=normalized(base['document'])
   if key in used or key in excluded:continue
   used.add(key);r=copy.deepcopy(base);r.update(id=f"audit119:{label}:{base['id']}",audit_stratum=label,source_row_id=base['id'],source_bank=str((AUTHORED if base.get('source')=='roles' else ORIGINAL).relative_to(ROOT)),original_pair_id=base.get('pair_id'),pair_id=None)
   found.append(r)
   if len(found)==count:break
  if len(found)!=count:raise ValueError(f'Insufficient unique TRAIN inputs for {label}')
  chosen.extend(found)
 for variant,n in [('canonical',8),('alias',4),('nullable',4)]:
  pool=[r for r in original if r['variant']==variant and any(len(v)>=18 and ' ' in v for v in present_values(r)) and all(not isinstance(v,list) or len(v)<=1 for v in r['expected'].values())]
  pick(pool,'copying',n)
 for cardinality in [2,3,4,5]:
  for family in ['artist','city']:
   pick([r for r in authored if r['view']==0 and r['cardinality']==cardinality and r['family']==family],'multiple_values',2)
 for variant,n in [('canonical',8),('alias',4),('nullable',4)]:
  pick([r for r in original if r['variant']==variant and any(v is None or v==[] for v in r['expected'].values()) and present_values(r)],'missing_fields',n)
 for cardinality in [0,1,2,3]:
  for family in ['artist','city']:
   for view in [1+cardinality%2,3]:
    pick([r for r in authored if r['view']==view and r['cardinality']==cardinality and r['family']==family],'role_assignment',1)
 return chosen

def verify_case(row,banks=None):
 Draft202012Validator.check_schema(row['schema']);Draft202012Validator(row['schema']).validate(row['expected'])
 if len(row['messages'])!=2 or row['messages'][0]!=dict(role='user',content=prompt(row['document'],row['schema'])) or row['messages'][1]['role']!='assistant' or strict_json(row['messages'][1]['content'])!=row['expected']:raise ValueError('Input/target serialization changed')
 if row['schema'].get('additionalProperties') is not False or set(row['schema']['required'])!=set(row['expected']):raise ValueError('Field contract')
 evidence=collections.defaultdict(list)
 for span in sorted(row['spans'],key=lambda s:s['start']):
  if row['document'][span['start']:span['end']]!=span['value']:raise ValueError('Span grounding')
  evidence[row.get('key_mapping',{}).get(span['field'],span['field'])].append(span['value'])
 for key,value in row['expected'].items():
  spans=evidence[key]
  if (isinstance(value,list) and value!=spans) or (not isinstance(value,list) and (len(spans)>1 or value!=(spans[0] if spans else None))):raise ValueError('Missing, merged or unsupported reference')
 if banks is not None:
  source=banks[row['source_bank']][row['source_row_id']]
  for key in ['document','schema','expected','messages','spans','key_mapping']:
   if row[key]!=source[key]:raise ValueError('Not an unchanged TRAIN request')


def source_audit(rows):
 original={r['id']:r for r in load(ORIGINAL)};manifest=json.loads((SOURCE/'source-manifest.json').read_text());loaded={};hashes={};count=0
 for row in rows:
  if row['source_bank']!=str(ORIGINAL.relative_to(ROOT)):continue
  r=original[row['source_row_id']];p=r['provenance'];name=p['file'];f=SOURCE/name
  if name not in loaded:
   if sha(f)!=manifest['files'][name]['sha256']:raise ValueError('Original SNIPS source changed')
   hashes[str(f.relative_to(ROOT))]=sha(f);loaded[name]=json.loads(f.read_bytes().decode(p['encoding']))
  chunks=loaded[name][r['source']][p['row']]['data'];labels=collections.defaultdict(list)
  for c in chunks:
   if c.get('entity') and c['text'].strip():labels[c['entity']].append(c['text'].strip())
  if ''.join(c['text'] for c in chunks)!=r['document'] or {k:labels[k] for k in r['canonical_expected']}!=r['canonical_expected']:raise ValueError('Original annotation witness mismatch')
  for field,key in r['key_mapping'].items():
   values=labels[field];gold=(values[0] if values else None) if field in r['scalar_fields'] else values
   if r['expected'][key]!=gold:raise ValueError('Original schema conversion mismatch')
  count+=1
 return dict(status='passed',original_requests=count,authored_requests=len(rows)-count,source_hashes=hashes,source_manifest_sha256=sha(SOURCE/'source-manifest.json'),scope='Original annotation agreement and literal authored spans/roles, not independent production semantic certification.')

def build():
 if DATA.exists() or OUTPUT.exists():raise ValueError('Refusing overwrite')
 DATA.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True);rows=select_cases()
 (DATA/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 inputs=[ORIGINAL,AUTHORED,HELDOUT/'development.jsonl',HELDOUT/'test.jsonl',SOURCE/'source-manifest.json']
 manifest=dict(format='json119-training-only-sampling-audit-v1',parent_sha256=PARENT_SHA,plan=PLAN,cases_sha256=sha(DATA/'cases.jsonl'),inputs={str(f.relative_to(ROOT)):sha(f) for f in inputs},selection='Source/schema/reference-based deterministic stratification; no model predictions or held-out failures consulted.')
 atomic_json(DATA/'manifest.json',manifest);return check(deep=True)

def check(deep=False):
 m=json.loads((DATA/'manifest.json').read_text())
 if m['plan']!=PLAN or m['parent_sha256']!=PARENT_SHA or sha(PARENT/'model.safetensors')!=PARENT_SHA or sha(DATA/'cases.jsonl')!=m['cases_sha256']:raise ValueError('Frozen audit identity changed')
 for name,h in m['inputs'].items():
  if sha(ROOT/name)!=h:raise ValueError('Audit source changed '+name)
 rows=load(DATA/'cases.jsonl')
 if len(rows)!=64 or len({r['id'] for r in rows})!=64 or len({normalized(r['document']) for r in rows})!=64 or collections.Counter(r['audit_stratum'] for r in rows)!={s:16 for s in PLAN['strata']}:raise ValueError('Audit coverage')
 if deep:
  if rows!=select_cases():raise ValueError('Frozen input selection changed')
  banks={str(f.relative_to(ROOT)):{r['id']:r for r in load(f)} for f in [ORIGINAL,AUTHORED]}
  for r in rows:verify_case(r,banks)
  excluded={normalized(r['document']) for s in ['development','test'] for r in load(HELDOUT/(s+'.jsonl'))}
  if any(normalized(r['document']) in excluded for r in rows):raise ValueError('Audit/evaluation source overlap')
  atomic_json(REPORT/'source-validation.json',source_audit(rows))
 return m

if __name__=='__main__':print(json.dumps(build(),indent=2))
