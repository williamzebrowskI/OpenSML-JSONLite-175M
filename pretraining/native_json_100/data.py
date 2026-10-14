"""Audited SNIPS annotation -> source/schema/JSON supervision, never model labels."""
import collections, hashlib, json, math
from pathlib import Path
import numpy as np
from jsonschema import Draft202012Validator
from sml_v1.common import atomic_json
from sft.data import serialize, Corpus
from sml_v1.tokenization import Tokenizer
from extract_068.data import GLOSS

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'pretraining/data/sft_068_snips_extraction_20261013'
SOURCE=ROOT/'pretraining/data/snips_sources_20261013'
CORPUS=ROOT/'pretraining/data/sft_100_native_json_20261014'
OUTPUT=ROOT/'pretraining/runs/sft_100_native_json_20261014'
REPORT=ROOT/'reports/model-design/SFT-100_native_json_20261014'
TOK=ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
PARENT=ROOT/'pretraining/runs/sft_093_json_approaches_20261014/step_0000160_82ffab052730'
PARENT_SHA='d552c1380c71c7c0569e9d692ec9c46c5c94bb87a7b024e56df0f0e36df4d798'
PLAN=dict(seed=20261014,batch_size=32,microbatch=2,epochs=1,peak_lr=5e-6,final_lr=1e-6,warmup_updates=20,grad_clip=1.,weight_decay=.01,betas=[.9,.95],eps=1e-8,eval_steps=[0,100,200,300,469],padding_policy='micro_multiple128',max_tokens=256)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def load(p):return [json.loads(l) for l in Path(p).open() if l.strip()]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def prompt(document,schema):
 # Gold values, offsets, intent, provenance, and key-mapping metadata NEVER enter this function.
 return ('Extract information from the supplied source_text according to the supplied JSON Schema. '
         'Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; '
         'do not normalize dates or numbers. Include every required key and no other keys. '
         'For string arrays, include all matching source spans in source order, or [] when absent. '
         'For nullable strings, use the exact source span or null when absent.\n\n'+
         dumps(dict(source_text=document,schema=schema)))

def make(row,variant,split):
 fields=row['fields'];mapping={f:(('out_' if split=='train' else 'selected_')+f if variant=='alias' else f) for f in fields}
 properties={};expected={};scalars=set()
 for f in fields:
  scalar=variant=='nullable' and len(row['expected'][f])<=1
  if scalar:scalars.add(f)
  properties[mapping[f]]=dict(type=['string','null'] if scalar else 'array',description=GLOSS[f])
  if not scalar:properties[mapping[f]]['items']={'type':'string'}
  expected[mapping[f]]=(row['expected'][f][0] if row['expected'][f] else None) if scalar else list(row['expected'][f])
 # Vary property order, independently of golds, on extra training views.
 if variant!='canonical' and int(digest(row['id'])[:8],16)%2:properties=dict(reversed(list(properties.items())))
 schema={'$schema':'https://json-schema.org/draft/2020-12/schema','type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
 out=dict(id=row['id']+':native:'+variant,base_id=row['id'],group=row['group'],source=row['source'],variant=variant,
          document=row['document'],schema=schema,expected=expected,canonical_expected=row['expected'],key_mapping=mapping,
          scalar_fields=sorted(scalars),provenance=row['provenance'],spans=row['spans'],
          pair_id=row['pair_id'] if variant=='canonical' else None)
 out['messages']=[dict(role='user',content=prompt(out['document'],schema)),dict(role='assistant',content=dumps(expected))]
 return out

def original_sources():
 sm=json.loads((SOURCE/'source-manifest.json').read_text());raw={}
 for name,info in sm['files'].items():
  if sha(SOURCE/name)!=info['sha256']:raise ValueError('Changed original source '+name)
  if name.endswith('.json'):
   b=(SOURCE/name).read_bytes()
   try:raw[name]=json.loads(b.decode('utf-8-sig'))
   except UnicodeDecodeError:raw[name]=json.loads(b.decode('latin-1'))
 return sm,raw

def verify_row(r,raw):
 provenance=r['provenance'];chunks=raw[provenance['file']][r['source']][provenance['row']]['data']
 if r['document']!=''.join(c['text'] for c in chunks):raise ValueError('Source changed')
 gold=collections.defaultdict(list)
 for c in chunks:
  if c.get('entity') and c['text'].strip():gold[c['entity']].append(c['text'].strip())
 if r['canonical_expected']!={f:gold.get(f,[]) for f in r['canonical_expected']}:raise ValueError('Gold disagrees with original annotation')
 for s in r['spans']:
  if r['document'][s['start']:s['end']]!=s['value']:raise ValueError('Invalid evidence offset')
 for f,k in r['key_mapping'].items():
  vs=gold.get(f,[]);want=(vs[0] if vs else None) if f in r['scalar_fields'] else vs
  if r['expected'][k]!=want:raise ValueError('Incorrect schema conversion')
 if r['messages']!=[dict(role='user',content=prompt(r['document'],r['schema'])),dict(role='assistant',content=dumps(r['expected']))]:raise ValueError('Prompt/target mismatch')
 Draft202012Validator.check_schema(r['schema']);Draft202012Validator(r['schema']).validate(r['expected'])

def pack(rows,split,tok):
 ids=[];masks=[];offsets=[0];max_prompt=0;max_target=0
 for r in rows:
  a,b=serialize(tok,r['messages']);p,_=serialize(tok,r['messages'][:-1],generation=True)
  t=tok.encode(r['messages'][-1]['content']);max_prompt=max(max_prompt,len(p));max_target=max(max_target,len(t))
  if len(a)>2048 or len(p)+PLAN['max_tokens']>2048 or len(t)>=PLAN['max_tokens']:raise ValueError('Context/output overflow, no truncation allowed')
  # Exactly JSON target and turn/document stops are supervised, never source/schema.
  if not np.array_equal(a[:len(p)],p) or np.any(b[:len(p)]) or not np.all(b[len(p):]):raise ValueError('Incorrect assistant mask')
  if not np.array_equal(a[len(p):-2],np.array(t,np.int32)) or list(a[-2:])!=[3,1]:raise ValueError('Target serialization mismatch')
  ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
 (CORPUS/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for k,v in [('tokens',np.concatenate(ids)),('mask',np.concatenate(masks)),('offsets',np.array(offsets,np.int64))]:np.save(CORPUS/f'{split}.{k}.npy',v)
 return dict(records=len(rows),unique_source_texts=len({r['group'] for r in rows}),serialized_tokens=offsets[-1],
             max_prompt_tokens=max_prompt,max_target_tokens=max_target,by_domain=dict(collections.Counter(r['source'] for r in rows)),
             by_variant=dict(collections.Counter(r['variant'] for r in rows)))

def build():
 if OUTPUT.exists() or CORPUS.exists():raise ValueError('Refusing to overwrite initialized run or data')
 CORPUS.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
 sm,raw=original_sources();old={s:load(OLD/f'{s}.jsonl') for s in ('train','development','test')}
 if len(old['train'])!=12115 or len(old['development'])!=196 or len(old['test'])!=1388:raise ValueError('Original split counts changed')
 sets={s:[make(r,'canonical',s) for r in rs] for s,rs in old.items()}
 bank=sorted(old['train'],key=lambda r:digest('native100:'+r['id']))[:2885]
 sets['train'] += [make(r,'alias' if i%2==0 else 'nullable','train') for i,r in enumerate(bank)]
 # Balanced compact schema stress controls; these texts stay out of training.
 for split in ('development','test'):
  chosen=[]
  for domain in sorted({r['source'] for r in old[split]}):
   chosen += sorted([r for r in old[split] if r['source']==domain],key=lambda r:digest('views100:'+r['id']))[:4]
  sets[split]+=[make(r,v,split) for r in chosen for v in ('alias','nullable')]
 tok=Tokenizer(TOK);stats={}
 for s,rows in sets.items():
  if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate IDs')
  for other,ors in sets.items():
   if other!=s and {r['group'] for r in rows}&{r['group'] for r in ors}:raise ValueError('Cross-split source overlap')
  for r in rows:verify_row(r,raw)
  stats[s]=pack(rows,s,tok)
 files={p.name:sha(p) for p in CORPUS.iterdir() if p.is_file()}
 inputs=[SOURCE/'source-manifest.json']+[SOURCE/n for n in sm['files']]+[OLD/f'{s}.jsonl' for s in old]
 atomic_json(CORPUS/'manifest.json',dict(format='native-json100-source-schema-json-v1',plan=PLAN,parent=str(PARENT),parent_sha256=PARENT_SHA,
     splits=stats,total_steps=math.ceil(len(sets['train'])/PLAN['batch_size']),files=files,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},
     limitations='Original SNIPS annotation agreement, not independently perfect semantic truth. Flat objects with arrays of strings and nullable strings; seven SNIPS domains and 39 known field definitions. Extra views reuse TRAIN texts, not extra distinct requests. DEV/test were observed in previous research; no fresh blind benchmark or general web-extraction claim.'))
 return check(deep=True)

def check(deep=False):
 m=json.loads((CORPUS/'manifest.json').read_text())
 if m['plan']!=PLAN or m['parent_sha256']!=PARENT_SHA or sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Changed plan/parent')
 for name,h in m['files'].items():
  if sha(CORPUS/name)!=h:raise ValueError('Changed prepared data '+name)
 for name,h in m['input_hashes'].items():
  if sha(ROOT/name)!=h:raise ValueError('Changed source '+name)
 if deep:
  _,raw=original_sources();tok=Tokenizer(TOK);groups={}
  for split in m['splits']:
   rows=load(CORPUS/f'{split}.jsonl');c=Corpus(CORPUS,split);groups[split]={r['group'] for r in rows}
   if len(rows)!=len(c) or len(rows)!=m['splits'][split]['records']:raise ValueError('Packed record count changed')
   for i,r in enumerate(rows):
    verify_row(r,raw);a,b=serialize(tok,r['messages']);start,end=map(int,c.offsets[i:i+2])
    if not np.array_equal(c.ids[start:end],a) or not np.array_equal(c.mask[start:end],b):raise ValueError('Packed target/mask changed')
  for s,g in groups.items():
   for t,h in groups.items():
    if s!=t and g&h:raise ValueError('Source leakage')
 return m
if __name__=='__main__':print(json.dumps(build(),indent=2))
