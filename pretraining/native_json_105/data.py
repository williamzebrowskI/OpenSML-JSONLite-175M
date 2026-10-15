"""Literal cardinality contrasts plus original TRAIN replay; no teacher labels."""
import collections,hashlib,json,math,re
from pathlib import Path
import numpy as np
from jsonschema import Draft202012Validator
from sml_v1.common import atomic_json
from sft.data import serialize,Corpus
from sml_v1.tokenization import Tokenizer
from extract_068.data import GLOSS
from native_json_100.data import prompt,dumps,sha,load,TOK,ROOT
CORPUS=ROOT/'pretraining/data/sft_105_cardinality_distillation_20261014'
OUTPUT=ROOT/'pretraining/runs/sft_105_cardinality_distillation_20261014'
REPORT=ROOT/'reports/model-design/SFT-105_cardinality_distillation_20261014'
OLD=ROOT/'pretraining/data/sft_100_native_json_20261014'
PARENT=ROOT/'pretraining/runs/sft_100_native_json_20261014/step_0000469_25528943984d'
PARENT_SHA='656f45015aab548465d93bed0b57e687497163741885da7364872851fa17e9a4'
PLAN=dict(seed=20261119,batch_size=32,microbatch=2,epochs=1,peak_lr=2e-6,final_lr=2e-7,warmup_updates=6,order_policy='prepared_prefix',grad_clip=1.,weight_decay=.01,betas=[.9,.95],eps=1e-8,eval_steps=[0,32,64,96],padding_policy='micro_multiple128',max_tokens=256,total_updates=96,replay_kl_weight=1.,distillation_temperature=1.)
ARCHIVE=ROOT/'reports/model-design/SFT-102_replay_focused_mix_20261014/retired-run-evidence'
PREVIOUS=ROOT/'reports/model-design/SFT-104_single_request_lists_20261014/retired-run-evidence/recipe'
FIXTURES=ROOT/'pretraining/playground/native-json-examples.json'
PLAYGROUND_CACHE=ROOT/'reports/model-design/SFT-100_native_json_20261014/playground-native-54-results.json'

def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def normalized(s):return tuple(re.findall(r'\w+|[^\w\s]',s.casefold()))

def verify_row(r):
 Draft202012Validator.check_schema(r['schema']);Draft202012Validator(r['schema']).validate(r['expected'])
 if r['schema']['additionalProperties'] is not False or set(r['schema']['required'])!=set(r['expected']) or not 1<=len(r['expected'])<=8:raise ValueError('Invalid field contract')
 if r['messages']!=[dict(role='user',content=prompt(r['document'],r['schema'])),dict(role='assistant',content=dumps(r['expected']))]:raise ValueError('Messages do not match source/schema/target')
 evidence=collections.defaultdict(list)
 for s in r['spans']:
  if r['document'][s['start']:s['end']]!=s['value']:raise ValueError('Wrong source span')
  evidence[r.get('key_mapping',{}).get(s['field'],s['field'])].append(s['value'])
 for key,val in r['expected'].items():
  want=evidence.get(key,[])
  if isinstance(val,list):
   if val!=want:raise ValueError('Omitted, merged, reordered or invented span')
  elif len(want)>1 or val!=(want[0] if want else None):raise ValueError('Wrong nullable value')

def pack(rows,split,tok):
 ids=[];masks=[];offsets=[0];max_prompt=0;max_target=0
 for r in rows:
  verify_row(r);a,b=serialize(tok,r['messages']);p,_=serialize(tok,r['messages'][:-1],generation=True);target=tok.encode(r['messages'][-1]['content'])
  if len(p)+PLAN['max_tokens']>2048 or len(target)>=PLAN['max_tokens'] or len(a)>2048:raise ValueError('Context overflow; no truncation')
  if not np.array_equal(a[:len(p)],p) or np.any(b[:len(p)]) or not np.all(b[len(p):]):raise ValueError('Bad output-only mask')
  if not np.array_equal(a[len(p):-2],np.asarray(target,np.int32)) or list(a[-2:])!=[3,1]:raise ValueError('Bad target/stops')
  ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));max_prompt=max(max_prompt,len(p));max_target=max(max_target,len(target))
 (CORPUS/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for k,v in [('tokens',np.concatenate(ids)),('mask',np.concatenate(masks)),('offsets',np.array(offsets,np.int64))]:np.save(CORPUS/f'{split}.{k}.npy',v)
 return dict(records=len(rows),unique_sources=len({r['group'] for r in rows}),tokens=offsets[-1],max_prompt_tokens=max_prompt,max_target_tokens=max_target,by_source=dict(collections.Counter(r['source'] for r in rows)),by_variant=dict(collections.Counter(r['variant'] for r in rows)),multi_value_requests=sum(any(isinstance(v,list) and len(set(v))>1 for v in r['expected'].values()) for r in rows),multiple_artist_requests=sum(isinstance(r['expected'].get('artist',r['expected'].get('selected_artist')),list) and len(set(r['expected'].get('artist',r['expected'].get('selected_artist'))))>1 for r in rows))

def playground_rows():
 rows=[]
 for case in json.loads(FIXTURES.read_text())['cases']:
  text=case['document'];gold=case['expected'];spans=[]
  if case['instruction']!=prompt('',{}).split('\n\n',1)[0]:raise ValueError('Playground instruction changed')
  for key,val in gold.items():
   values=val if isinstance(val,list) else ([] if val is None else [val]);pos=0
   for v in values:
    start=text.find(v,pos)
    if start<0:raise ValueError('Playground gold is not exact source evidence')
    spans.append(dict(field=key,start=start,end=start+len(v),value=v));pos=start+len(v)
  r=dict(id=case['id'],group=digest(text),source='playground',variant='canonical',document=text,schema=case['schema'],expected=gold,pair_id=None,key_mapping={},spans=spans,provenance={'origin':'Observed playground diagnostic; no training use'},messages=[dict(role='user',content=prompt(text,case['schema'])),dict(role='assistant',content=dumps(gold))])
  verify_row(r);rows.append(r)
 if len(rows)!=54:raise ValueError('Expected54playgroundfixtures')
 return rows

TEMPLATES={
 'train':{
  'artist':['Play songs by {values}.','Queue music from {values}.','I want to hear tracks performed by {values}.','Include recordings by {values} in my playlist.','Put on some music by {values}.','The performers I want are {values}.','Please choose songs sung by {values}.','Start a listening session featuring {values}.'],
  'city':['Show the weather in {values}.','Get forecasts for the cities {values}.','I need weather reports for {values}.','Compare conditions in {values}.','Give me the forecast for {values}.','The weather locations are {values}.','Look up temperatures in {values}.','Please check the weather for {values}.'],
  'artist0':['Play the album "{related}".','Start playback of the album titled "{related}".','Queue the record "{related}".','I want to listen to the album "{related}".'],
  'city0':['Get the weather in the country {related}.','Show a national weather report for {related}.','Give me conditions for the country {related}.','I need a forecast for the country {related}.']},
 'development':{
  'artist':['For my music selection, include performances by {values}.','Could you put together songs from {values}?'],
  'city':['Please retrieve forecasts covering {values}.','I would like weather information for the cities {values}.'],
  'artist0':['The album I would like played is "{related}".'],
  'city0':['The requested country for the weather bulletin is {related}.']},
 'test':{
  'artist':['My listening choices should feature {values}.','Can the music queue include tracks by {values}?'],
  'city':['My forecast destinations are {values}.','Fetch local weather updates for the cities {values}.'],
  'artist0':['For the listening session, select the album "{related}".'],
  'city0':['Prepare a weather summary for the country {related}.']}}

def make_row(field,values,related,template,index,split,variant):
 joined=(values[0] if len(values)==1 else ' and '.join(values) if len(values)==2 else ', '.join(values[:-1])+', and '+values[-1]) if values else ''
 text=template.format(values=joined,related=related)
 second={'artist':'album','city':'country'}[field]
 canonical={field:list(values),second:[] if values else [related]}
 mapping={f:('out_' if split=='train' else 'requested_')+f if variant=='alias' else f for f in canonical}
 props={};gold={};spans=[]
 for f,vs in canonical.items():
  key=mapping[f];scalar=variant=='nullable' and len(vs)<=1
  props[key]=dict(type=['string','null'] if scalar else 'array',description=GLOSS[f])
  if not scalar:props[key]['items']={'type':'string'}
  gold[key]=(vs[0] if vs else None) if scalar else vs
  pos=0
  for value in vs:
   start=text.find(value,pos)
   if start<0:raise ValueError('Missing literal binding')
   spans.append(dict(field=f,start=start,end=start+len(value),value=value));pos=start+len(value)
 if index%2:props=dict(reversed(list(props.items())))
 schema=dict(type='object',properties=props,required=list(props),additionalProperties=False)
 r=dict(id=f'cardinality105:{split}:{index}:{variant}',base_id=f'cardinality105:{split}:{index}',group=digest(text),source='cardinality',variant=variant,family=field,cardinality=len(values),bindings={field:values,second:[] if values else [related]},template=template,document=text,schema=schema,expected=gold,key_mapping=mapping,spans=spans,pair_id=None,contrast_group=f'card105:{split}:{field}:{index//4}',provenance={'origin':'OpenSML-authored literal role/cardinality contrasts; not naturally sampled user traffic','split':split},messages=[dict(role='user',content=prompt(text,schema)),dict(role='assistant',content=dumps(gold))])
 verify_row(r);return r

def value_pools(original,protected):
 text='\n'.join(protected).casefold();out={}
 for f in ['artist','city','album','country']:
  vs={v for r in original for v in r.get('canonical_expected',{}).get(f,[]) if len(v)>=3 and len(v.split())<=4 and re.fullmatch(r"[A-Za-zÀ-ž][A-Za-zÀ-ž '.-]*",v) and v.casefold() not in text}
  if f=='artist':vs={v for v in vs if len(v.split())>=2 and ' and ' not in v.casefold()}
  vs=sorted(vs,key=lambda v:digest('pool105:'+f+v))
  if len(vs)<48:raise ValueError(f'Not enough disjoint values for {f}: {len(vs)}')
  out[f]={'train':vs[:32],'development':vs[32:40],'test':vs[40:48]}
 return out

def authored(pool,split):
 rng=np.random.default_rng(PLAN['seed']+{'train':0,'development':1,'test':2}[split]);rows=[];index=0
 for field in ['artist','city']:
  second={'artist':'album','city':'country'}[field];groups={'train':32,'development':6,'test':8}[split]
  for g in range(groups):
   first=pool[field][split][g];values=[first]+list(rng.choice([v for v in pool[field][split] if v!=first],2,replace=False));related=pool[second][split][g]
   for n in range(4):
    choices=TEMPLATES[split][field if n else field+'0'];template=choices[g%len(choices)]
    variants=['canonical','alias','nullable'] if split=='train' else [('canonical','alias','nullable')[(index//4+n)%3]]
    for variant in variants:rows.append(make_row(field,values[:n],related,template,index,split,variant))
    index+=1
 return rows

def build():
 if CORPUS.exists() or OUTPUT.exists():raise ValueError('Refusing overwrite')
 original=load(OLD/'train.jsonl');legacy_dev=load(ARCHIVE/'recipe/development.jsonl');legacy_test=load(ARCHIVE/'recipe/test.jsonl');previous_dev=[r for r in load(PREVIOUS/'development.jsonl') if r['source']=='single_lists'];previous_test=[r for r in load(PREVIOUS/'test.jsonl') if r['source']=='single_lists'];play=playground_rows()
 protected=[r['document'] for r in legacy_dev+legacy_test+previous_dev+previous_test+play]
 pool=value_pools(original,protected);fresh={s:authored(pool,s) for s in ['train','development','test']}
 held=[v.casefold() for p in pool.values() for s in ['development','test'] for v in p[s]];replay={}
 for variant,count in [('canonical',1344),('alias',480),('nullable',480)]:
  bank=sorted([r for r in original if r['variant']==variant and not any(v in r['document'].casefold() for v in held)],key=lambda r:digest('replay105:'+r['id']))[:count]
  if len(bank)!=count:raise ValueError('Insufficient replay')
  replay[variant]=[dict(json.loads(json.dumps(r)),id='replay105:'+r['id'],source='replay') for r in bank]
 rng=np.random.default_rng(PLAN['seed']);new=[fresh['train'][i] for i in rng.permutation(len(fresh['train']))];train=[]
 for batch in range(96):train+=replay['canonical'][14*batch:14*(batch+1)]+replay['alias'][5*batch:5*(batch+1)]+replay['nullable'][5*batch:5*(batch+1)]+new[8*batch:8*(batch+1)]
 sets=dict(train=train,development=legacy_dev+previous_dev+fresh['development'],test=legacy_test+previous_test+fresh['test'],playground=play)
 CORPUS.mkdir(parents=True);REPORT.mkdir(exist_ok=True,parents=True);tok=Tokenizer(TOK);stats={s:pack(rs,s,tok) for s,rs in sets.items()}
 inputs=[OLD/'manifest.json',OLD/'train.jsonl',ARCHIVE/'recipe/development.jsonl',ARCHIVE/'recipe/test.jsonl',ARCHIVE/'evaluations/development_0000000.json',ARCHIVE/'evaluations/development_0000000.identity.json',ARCHIVE/'final-test.json',ARCHIVE/'final-test-identity.json',PREVIOUS/'development.jsonl',PREVIOUS/'test.jsonl',FIXTURES,PLAYGROUND_CACHE]
 files={p.name:sha(p) for p in CORPUS.iterdir() if p.is_file()};m=dict(format='native-json105-cardinality-kl-v1',parent=str(PARENT),parent_sha256=PARENT_SHA,plan=PLAN,total_steps=96,splits=stats,replay_records=2304,authored_training_views=768,authored_training_sources=len({r['group'] for r in fresh['train']}),value_pools=pool,files=files,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs},limitations='Programmatic cardinality/role supervision; no independently collected real user benchmark claim. New authored values/phrases split-disjoint, exact source spans validated. Frozen parent output-distribution distillation only on original TRAIN replay, never DEV/TEST. KL cannot guarantee retention and may preserve parent mistakes. All benchmark panels observed research controls. Raw whole-JSON generation; no assembly/repair/constrained decoding.')
 atomic_json(CORPUS/'manifest.json',m);check(deep=True);atomic_json(REPORT/'data-validation.json',dict(status='passed',files=files,source_hashes=m['input_hashes'],splits=stats));return m

def check(deep=False):
 m=json.loads((CORPUS/'manifest.json').read_text())
 if m['plan']!=PLAN or sha(PARENT/'model.safetensors')!=PARENT_SHA:raise ValueError('Plan/parent changed')
 for name,h in m['files'].items():
  if sha(CORPUS/name)!=h:raise ValueError('Prepared data changed '+name)
 for name,h in m['input_hashes'].items():
  if sha(ROOT/name)!=h:raise ValueError('Source changed '+name)
 if deep:
  tok=Tokenizer(TOK);sets={};allrows={};held=[v.casefold() for p in m['value_pools'].values() for s in ['development','test'] for v in p[s]]
  for split in m['splits']:
   rows=load(CORPUS/(split+'.jsonl'));allrows[split]=rows;c=Corpus(CORPUS,split);sets[split]={normalized(r['document']) for r in rows}
   if len(rows)!=len(c) or len({r['id'] for r in rows})!=len(rows):raise ValueError('Membership mismatch')
   for i,r in enumerate(rows):
    verify_row(r)
    if r['source']=='cardinality':
     n=int(r['base_id'].split(':')[-1]);second={'artist':'album','city':'country'}[r['family']];related=r['bindings'][second][0] if not r['cardinality'] else 'unused'
     reconstructed=make_row(r['family'],r['bindings'][r['family']],related,r['template'],n,r['provenance']['split'],r['variant'])
     if r!=reconstructed:raise ValueError('Cardinality/binding/source/target mismatch')
    a,b=serialize(tok,r['messages']);start,end=map(int,c.offsets[i:i+2])
    if not np.array_equal(c.ids[start:end],a) or not np.array_equal(c.mask[start:end],b):raise ValueError('Stored masks/targets mismatch')
  for s in ['development','test','playground']:
   if sets['train']&sets[s]:raise ValueError('Train/evaluation source overlap')
  if sets['development']&sets['test']:raise ValueError('DEV/TEST source overlap')
  rows=allrows['train']
  if len(rows)!=3072 or any(v in r['document'].casefold() for r in rows for v in held):raise ValueError('Membership/lexical leakage')
  for i in range(0,len(rows),32):
   if collections.Counter(r['source'] for r in rows[i:i+32])!={'replay':24,'cardinality':8}:raise ValueError('Batch mixture changed')
  fresh=[r for r in rows if r['source']=='cardinality']
  if collections.Counter(r['cardinality'] for r in fresh)!={0:192,1:192,2:192,3:192}:raise ValueError('Unbalanced cardinality')
  for p in m['value_pools'].values():
   if any(set(p[a])&set(p[b]) for a in p for b in p if a!=b):raise ValueError('Lexical split overlap')
 return m
if __name__=='__main__':print(json.dumps(build(),indent=2))
