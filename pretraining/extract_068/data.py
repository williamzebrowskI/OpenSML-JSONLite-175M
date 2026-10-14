"""SNIPS chunks -> source-preserving requested-field extraction. No JSON target."""
import json,re,collections,hashlib,math
from pathlib import Path
import numpy as np
from qa_066.data import ROOT,PARENT,PARENT_SHA,TOK
from qa_065.data import load,h
from sml_v1.common import atomic_json
from json_sft.corpus import file_sha
CORPUS=ROOT/'pretraining/data/sft_068_snips_extraction_20261013'
OUTPUT=ROOT/'pretraining/runs/sft_068_snips_extraction_20261013'
REPORT=ROOT/'reports/model-design/SFT-068_snips_extraction_20261013'
SOURCE=ROOT/'pretraining/data/snips_sources_20261013'
PLAN=dict(seed=20261013,batch_size=32,microbatch=2,epochs=1,peak_lr=5e-6,final_lr=1e-6,warmup_updates=20,grad_clip=1.,weight_decay=.01,betas=[.9,.95],eps=1e-8,eval_steps=[0,100,200,300],padding_policy='micro_multiple128')
GLOSS={'artist':'performer or musical artist','entity_name':'named item to add to a playlist','music_item':'word specifying the requested music category, such as artist, track, song, or album','playlist':'playlist name','playlist_owner':'person owning the playlist','city':'city or town','country':'country','cuisine':'cuisine style','facility':'requested restaurant facility','party_size_description':'description of who is attending','party_size_number':'explicit party size','poi':'named nearby point of interest','restaurant_name':'restaurant name','restaurant_type':'kind of restaurant','served_dish':'food or dish','sort':'requested ranking, ordering, or preference','spatial_relation':'spatial or proximity expression','state':'state or region','timeRange':'date, time, or time-range expression','condition_description':'weather condition','condition_temperature':'temperature description','current_location':'expression referring to current location','geographic_poi':'geographic point of interest','album':'album name','genre':'music genre','service':'music service','track':'track name','year':'year expression','best_rating':'maximum rating scale','object_name':'name of the work being rated or searched for','object_part_of_series_type':'type of book series','object_select':'selection expression referring to the work','object_type':'kind of work or object','rating_unit':'rating unit','rating_value':'given rating','location_name':'screening location name','movie_name':'movie name','movie_type':'kind of movie','object_location_type':'type of screening venue'}

def key(text):return ' '.join(text.casefold().split())
def render(fields,slots):
 return '\n'.join(f'{f}: {value}' for f in fields for value in (slots.get(f) or ['NOT_STATED']))
def prompt(text,fields,style):
 variants=['Extract only the requested fields from the text.','Read the request below and copy the values for the listed fields.','Identify the exact text spans for each requested field.','Return the requested information using only the supplied text.']
 return variants[style%4]+" Copy each value exactly, preserving spelling and case. Do not normalize numbers or dates. Output one field: value line per value, in requested-field order and source order within a field. If the text does not supply a requested field, output field: NOT_STATED. Do not add explanations.\n\nRequested fields:\n"+'\n'.join(f'- {f}: {GLOSS[f]}' for f in fields)+'\n\nText:\n'+text

def make(base,view,split,requested=None):
 rng=np.random.default_rng(int(h('extract068:'+base['id']+str(view))[:16],16));present=sorted(base['slots']);absent=[f for f in base['ontology'] if f not in present]
 if view==0:
  # Most views request observed fields plus at most one explicitly empty slot.
  count=min(len(present),int(rng.integers(1,5)));fields=list(rng.choice(present,count,replace=False)) if count else []
  if absent and rng.random()<.4:fields.append(str(rng.choice(absent)))
 else:
  # Contrast the same source with a different requested field set.
  fields=list(rng.choice(present,min(1,len(present)),replace=False))
  if absent:
   if rng.random()<.2:fields=[] # The requested fields can all be absent.
   fields+=list(rng.choice(absent,min(2,len(absent)),replace=False))
 if requested is not None:fields=list(requested)
 if not fields:raise ValueError('No requested fields')
 rng.shuffle(fields);text=base['text'];style=int(rng.integers(0,4));target=render(fields,base['slots'])
 return dict(id=base['id']+f':view{view}',group=key(text),source=base['intent'],document=text,fields=fields,expected={f:base['slots'].get(f,[]) for f in fields},spans=[s for s in base['spans'] if s['field'] in fields],pair_id=base['id'] if split!='train' else None,messages=[dict(role='user',content=prompt(text,fields,style)),dict(role='assistant',content=target)],provenance=base['provenance'])

def pack(rows,split,tok):
 from sft.data import serialize
 ids=[];masks=[];offsets=[0]
 for r in rows:
  a,b=serialize(tok,r['messages']);p,_=serialize(tok,r['messages'][:-1],generation=True)
  assert len(a)<=1024 and len(p)+160<=2048 and len(tok.encode(r['messages'][-1]['content']))<=150
  ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
 (CORPUS/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for k,v in [('tokens',np.concatenate(ids)),('mask',np.concatenate(masks)),('offsets',np.array(offsets,np.int64))]:np.save(CORPUS/f'{split}.{k}.npy',v)
 return dict(records=len(rows),unique_texts=len({r['group'] for r in rows}),tokens=offsets[-1],by_intent=dict(collections.Counter(r['source'] for r in rows)),requested_fields=sum(len(r['fields']) for r in rows),absent_fields=sum(not v for r in rows for v in r['expected'].values()))

def build():
 from sml_v1.tokenization import Tokenizer
 from sft.data import serialize
 assert not OUTPUT.exists();CORPUS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 sm=json.loads((SOURCE/'source-manifest.json').read_text());pool={'train':[],'test':[]};ontologies={};counts=collections.Counter();encodings={}
 for name,meta in sm['files'].items():
  assert file_sha(SOURCE/name)==meta['sha256']
  if not name.endswith('.json'):continue
  raw=(SOURCE/name).read_bytes()
  try:parsed=json.loads(raw.decode('utf-8-sig'));encoding='utf-8'
  except UnicodeDecodeError:parsed=json.loads(raw.decode('latin-1'));encoding='latin-1'
  encodings[name]=encoding;split='test' if '__validate_' in name else 'train'
  for intent,records in parsed.items():
   ontologies.setdefault(intent,set())
   for i,r in enumerate(records):
    counts['raw_'+split]+=1;chunks=r['data'];text=''.join(c['text'] for c in chunks);spans=[];pos=0;slots=collections.defaultdict(list)
    for c in chunks:
     field=c.get('entity');chunk=c['text'];start=pos+len(chunk)-len(chunk.lstrip());end=pos+len(chunk.rstrip());pos+=len(chunk)
     if field:
      assert field in GLOSS
      ontologies[intent].add(field)
      if end>start:
       value=text[start:end];spans.append(dict(field=field,start=start,end=end,value=value));slots[field].append(value)
    if not text.strip() or not slots or any('\n' in v or '\r' in v or v=='NOT_STATED' for vs in slots.values() for v in vs) or re.search(r'[\x00-\x1f\x7f-\x9f]',text):counts['unsupported_text']+=1;continue
    assert all(text[s['start']:s['end']]==s['value'] for s in spans)
    pool[split].append(dict(id='snips:'+intent+':'+split+':'+str(i),intent=intent,text=text,slots=dict(slots),spans=spans,provenance=dict(repository=sm['repository'],revision=sm['revision'],file=name,row=i,encoding=encoding,license='CC0-1.0',annotation='Original SNIPS entity chunks; label agreement is not independent semantic certification.')))
 # Exclude conflicting annotations for duplicate source text globally.
 grouped=collections.defaultdict(list)
 for rows in pool.values():
  for r in rows:grouped[key(r['text'])].append(r)
 bad={k for k,rs in grouped.items() if len({json.dumps([r['intent'],r['slots']],sort_keys=True) for r in rs})>1};counts['conflicting_text_groups']=len(bad)
 test_keys={key(r['text']) for r in pool['test']};sets={'train':[],'development':[],'test':[]};tok=Tokenizer(TOK)
 for split,rows in pool.items():
  seen=set()
  for r in sorted(rows,key=lambda r:h(r['id'])):
   k=key(r['text'])
   if k in bad:continue
   if k in seen:counts['duplicate_'+split]+=1;continue
   seen.add(k)
   if split=='train' and k in test_keys:counts['official_test_overlap_removed']+=1;continue
   r['ontology']=sorted(ontologies[r['intent']])
   dest='test' if split=='test' else ('development' if int(h('snips068split:'+k)[:8],16)%10==0 else 'train')
   views=[make(r,1 if int(h('requestview:'+r['id'])[:8],16)%4==0 else 0,dest)] if dest=='train' else [make(r,v,dest) for v in (0,1)]
   if dest!='train' and set(views[0]['fields'])==set(views[1]['fields']):
    alternative=[f for f in r['ontology'] if f not in views[0]['fields']]
    replacement=[alternative[0]] if alternative else views[0]['fields'][:-1]
    assert replacement
    views[1]=make(r,1,dest,requested=replacement)
   if any(len(serialize(tok,v['messages'])[0])>1024 or len(tok.encode(v['messages'][-1]['content']))>150 for v in views):counts['length_excluded']+=1;continue
   sets[dest]+=views
 # Compact fixed checkpoint panel: 14 sources * 2 views per each of seven intents.
 dev=[]
 for intent in sorted(ontologies):
  rs=[r for r in sets['development'] if r['source']==intent];groups=sorted({r['pair_id'] for r in rs},key=lambda s:h('compact:'+s))[:14]
  dev += [r for r in rs if r['pair_id'] in groups]
 sets['development']=dev;assert len(dev)==196
 for s,rows in sets.items():
  assert len({r['id'] for r in rows})==len(rows)
  for t,others in sets.items():
   if s!=t:assert not {r['group'] for r in rows}&{r['group'] for r in others}
 splits={s:pack(rs,s,tok) for s,rs in sets.items()};total=math.ceil(len(sets['train'])/PLAN['batch_size']);assert 300<total<600
 # Final update is automatically evaluated in addition to the scheduled points.
 atomic_json(CORPUS/'manifest.json',dict(plan=PLAN,total_steps=total,splits=splits,source_manifest_sha256=file_sha(SOURCE/'source-manifest.json'),sources={n:v['sha256'] for n,v in sm['files'].items()},files={p.name:file_sha(p) for p in CORPUS.iterdir() if p.is_file() and p.name!='manifest.json'},scope='SNIPS-only annotated slot extraction. Original validate files are held-out final test; compact dev is drawn from withheld full-training queries. No normalization, JSON target, synthetic facts, or unrelated QA mixing. Exact copied spans and annotation-defined absence; not a general entity-relation or natural-paragraph benchmark.'))
 atomic_json(REPORT/'source-audit.json',dict(revision=sm['revision'],counts=dict(counts),encodings=encodings,ontologies={k:sorted(v) for k,v in ontologies.items()},split_stats=splits))
 atomic_json(REPORT/'review-packet.json',{s:[r for intent in sorted(ontologies) for r in [x for x in sets[s] if x['source']==intent][:4]] for s in sets})
 return check(True)

def check(deep=False):
 from sft.data import Corpus,serialize
 from sml_v1.tokenization import Tokenizer
 m=json.loads((CORPUS/'manifest.json').read_text());assert m['plan']==PLAN
 assert file_sha(SOURCE/'source-manifest.json')==m['source_manifest_sha256']
 for n,v in m['files'].items():assert file_sha(CORPUS/n)==v
 for n,v in m['sources'].items():assert file_sha(SOURCE/n)==v
 assert file_sha(PARENT/'model.safetensors')==PARENT_SHA
 if deep:
  tok=Tokenizer(TOK);original={}
  for split in m['splits']:
   c=Corpus(CORPUS,split)
   for i,r in enumerate(load(CORPUS/f'{split}.jsonl')):
    prov=r['provenance'];name=prov['file']
    if name not in original:original[name]=json.loads((SOURCE/name).read_bytes().decode(prov['encoding']))
    chunks=original[name][r['source']][prov['row']]['data']
    assert r['document']==''.join(c['text'] for c in chunks)
    gold=collections.defaultdict(list)
    for c0 in chunks:
     if c0.get('entity') and c0['text'].strip():gold[c0['entity']].append(c0['text'].strip())
    assert r['expected']=={f:gold.get(f,[]) for f in r['fields']}
    assert all(r['document'][s['start']:s['end']]==s['value'] for s in r['spans'])
    assert r['messages'][-1]['content']==render(r['fields'],r['expected'])
    a,b=serialize(tok,r['messages']);start,end=c.offsets[i:i+2];assert np.array_equal(a,c.ids[start:end]) and np.array_equal(b,c.mask[start:end])
 return m
if __name__=='__main__':print(json.dumps(build(),indent=2))
