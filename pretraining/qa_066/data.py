"""8k conservative continuation: natural prose, balanced support, old-skill replay."""
import json,collections
from pathlib import Path
from qa_065.data import ROOT,TOK,make,load,h,normalized
from sml_v1.common import atomic_json
from json_sft.corpus import file_sha
from sml_v1.tokenization import Tokenizer
from sft.data import serialize
import numpy as np
PARENT=ROOT/'pretraining/runs/sft_065_targeted_qa_20261013/step_0000469_9c2f49e737d3'
# Frozen below at setup; check every launch.
PARENT_SHA='0991b805a878c15733c909ef7f4127baa2b8866f8ca7d7db232ad39434f80bc4'
CORPUS=ROOT/'pretraining/data/sft_066_prose_repair_20261013'
OUTPUT=ROOT/'pretraining/runs/sft_066_prose_repair_20261013'
REPORT=ROOT/'reports/model-design/SFT-066_prose_repair_20261013'
OLD=ROOT/'pretraining/data/sft_065_targeted_qa_20261013'
REPLAY=ROOT/'pretraining/data/sft_064_grounded_qa_20261013'
PLAN=dict(seed=20261013,batch_size=32,microbatch=2,epochs=1,peak_lr=2e-6,final_lr=5e-7,warmup_updates=10,grad_clip=1.,weight_decay=.01,betas=[.9,.95],eps=1e-8,eval_steps=[0,83,167,250],padding_policy='micro_multiple128')

def prose(r):
 o=r['oracle'];a=o['person'];b=o['other_person'];x=o['first_value'];z=o['other_first'];y=o['second_value'];w=o['other_second'];t=o['revised_second'];domain=o['domain'];f=o['first_field'];g=o['second_field']
 frames={
 'work assignment':(f'{a} works on {x} in {y}.',f'{b} works on {z} in {w}.',f'{a} works on {x}, but the office location was not provided.'),
 'workshop booking':(f'{a} booked a {x} course at {y}.',f'{b} booked {z} at {w}.',f'{a} booked a {x} course; the venue was not specified.'),
 'shipment':(f'{a} arranged for {x} to be sent to {y}.',f'{b} ordered {z} for delivery to {w}.',f'{a} arranged a shipment of {x}, without giving a destination.'),
 'research assignment':(f'{a} is studying {x} at {y}.',f'{b} studies {z} at {w}.',f'{a} is studying {x}; no laboratory is named for this work.'),
 'travel booking':(f'{a} is booked on {x} to {y}.',f'{b} will take {z} to {w}.',f'{a} is booked on {x}, but the destination is not given.'),
 'exhibition entry':(f'{a} submitted {x} for display at {y}.',f'{b} submitted {z} to {w}.',f'{a} submitted {x} for an exhibition; the gallery was not specified.'),
 'volunteer assignment':(f'{a} is {x} at {y}.',f'{b} is {z} at {w}.',f'{a} is {x}, but no site was given for the assignment.'),
 'library request':(f'{a} requested {x} for collection at {y}.',f'{b} requested {z} at {w}.',f'{a} requested {x}, without specifying a pickup branch.')}
 target,other,missing=frames[domain];target=missing if r['answer_absent'] else target
 order=int(h(r['group'])[:8],16)%2
 doc='Fictional scenario. '+' '.join([target,other] if order==0 else [other,target])
 if r['source']=='corrections':
  variant=int(h(r['group'])[:8],16)%3
  changes=[f'A later message for {a} replaces {y} with {t}. The {f} stays {x}.',f'{a} subsequently changed the {g} to {t}; it had initially been {y}. The {f} is unchanged.',f'The first arrangement for {a} used {y}. This was superseded by {t}, with {x} still the {f}.']
  doc+=' '+changes[variant]
 r['document']=doc;r['messages'][0]['content']=r['messages'][0]['content'].split('\n\nPassage:')[0]+'\n\nPassage:\n'+doc+'\n\nQuestion: '+r['question']
 r['id']=r['id'].replace('qa065:','qa066:');r['frame_id']='prose066:'+domain
 r['bucket']='prose_'+r['source'];r['provenance']['quality']='Authored natural-prose templates and explicit fact oracle; controlled, not broad web extraction'
 assert o['person'] in doc and o['other_person'] in doc
 for v in r['required_values']:assert v in doc and v in r['answers'][0]
 if r['answer_absent']:assert y not in doc and t not in doc
 return r

def pack(rows,split,tok):
 ids=[];masks=[];offsets=[0]
 for r in rows:
  a,b=serialize(tok,r['messages']);prompt,_=serialize(tok,r['messages'][:-1],generation=True)
  assert len(a)<=1024 and len(prompt)+160<=2048
  ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
 (CORPUS/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for k,v in [('tokens',np.concatenate(ids)),('mask',np.concatenate(masks)),('offsets',np.array(offsets,np.int64))]:np.save(CORPUS/f'{split}.{k}.npy',v)
 return dict(records=len(rows),tokens=offsets[-1],absent=sum(r['answer_absent'] for r in rows),buckets=dict(collections.Counter(r.get('bucket','replay') for r in rows)))

def check(deep=False):
 m=json.loads((CORPUS/'manifest.json').read_text());assert m['plan']==PLAN
 for n,v in m['files'].items():assert file_sha(CORPUS/n)==v
 for n,v in m['sources'].items():assert file_sha(ROOT/n)==v
 assert file_sha(PARENT/'model.safetensors')==PARENT_SHA
 if deep:
  from sft.data import Corpus
  tok=Tokenizer(TOK)
  for split in m['splits']:
   c=Corpus(CORPUS,split)
   for i,r in enumerate(load(CORPUS/f'{split}.jsonl')):
    if r.get('bucket','').startswith('prose_'):audit_prose_support(r)
    a,b=serialize(tok,r['messages']);start,end=c.offsets[i:i+2];assert np.array_equal(a,c.ids[start:end]) and np.array_equal(b,c.mask[start:end])
 return m

def build():
 from .controlled import generate,validate
 assert not CORPUS.exists()
 old=load(REPLAY/'train.jsonl');dolly=[r for r in old if r['source']=='dolly_closed_qa'];positive=[r for r in old if r['source']=='squad2' and not r['answer_absent']][:3000-len(dolly)];missing=[r for r in old if r['answer_absent']][:1000]
 replay=[dict(r,bucket='natural_replay') for r in dolly+positive+missing]
 multipart=[dict(r,bucket='multipart_replay') for r in load(OLD/'train.jsonl') if r['source']=='multipart'][:1000]
 repair=[]
 for source,n in [('calibration',1000),('corrections',2000)]:
  for r in generate('train',source,n):validate(r);repair.append(prose(r))
 rows=replay+multipart+repair;assert len(rows)==8000
 sets={'train':rows,'development':load(OLD/'development.jsonl'),'test':load(OLD/'test.jsonl')}
 for split,rs in sets.items():
  assert len({r['id'] for r in rs})==len(rs)
  assert len({normalized(r['document'])+'\n'+normalized(r['question']) for r in rs})==len(rs)
  for other,os in sets.items():
   if split!=other:
    assert not {normalized(r['document']) for r in rs}&{normalized(r['document']) for r in os}
    assert not {r['group'] for r in rs}&{r['group'] for r in os}
 CORPUS.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True);tok=Tokenizer(TOK)
 splits={s:pack(rs,s,tok) for s,rs in sets.items()}
 sources=[REPLAY/'train.jsonl']+[OLD/f'{s}.jsonl' for s in sets]
 atomic_json(CORPUS/'manifest.json',dict(plan=PLAN,parent=str(PARENT),parent_sha256=PARENT_SHA,splits=splits,files={p.name:file_sha(p) for p in CORPUS.iterdir() if p.is_file()},sources={str(p.relative_to(ROOT)):file_sha(p) for p in sources},limitations='Continuation experiment; new controlled prose templates plus replay, not independently authored natural corpus. Existing dev/test already observed. No training examples copied from eval. Semantic equivalence is conservative, not a universal judge. SQuAD/Dolly provenance/licenses retained.'))
 atomic_json(REPORT/'review-packet.json',{s:[r for r in repair if r['source']==s][:24] for s in ('calibration','corrections')})
 return check(True)

def audit_prose_support(r):
 import re
 from .controlled import validate
 validate(r);o=r['oracle'];p=re.escape(o['person']);doc=r['document'];x=o['first_value'];y=o['second_value'];t=o['revised_second']
 pats={
 'work assignment':rf'{p} works on ([^.]*) in ([^.]*)\.',
 'workshop booking':rf'{p} booked a ([^.]*) course at ([^.]*)\.',
 'shipment':rf'{p} arranged for ([^.]*) to be sent to ([^.]*)\.',
 'research assignment':rf'{p} is studying ([^.]*) at ([^.]*)\.',
 'travel booking':rf'{p} is booked on ([^.]*) to ([^.]*)\.',
 'exhibition entry':rf'{p} submitted ([^.]*) for display at ([^.]*)\.',
 'volunteer assignment':rf'{p} is ([^.]*) at ([^.]*)\.',
 'library request':rf'{p} requested ([^.]*) for collection at ([^.]*)\.'}
 found=re.search(pats[o['domain']],doc)
 if r['answer_absent']:
  assert not found and y not in doc
  missing={
 'work assignment':rf'{p} works on ([^.]*), but the office location was not provided\.',
 'workshop booking':rf'{p} booked a ([^.]*) course; the venue was not specified\.',
 'shipment':rf'{p} arranged a shipment of ([^.]*), without giving a destination\.',
 'research assignment':rf'{p} is studying ([^.]*); no laboratory is named for this work\.',
 'travel booking':rf'{p} is booked on ([^.]*), but the destination is not given\.',
 'exhibition entry':rf'{p} submitted ([^.]*) for an exhibition; the gallery was not specified\.',
 'volunteer assignment':rf'{p} is ([^.]*), but no site was given for the assignment\.',
 'library request':rf'{p} requested ([^.]*), without specifying a pickup branch\.'}
  m=re.search(missing[o['domain']],doc);assert m and m.group(1)==x
 else:assert found and found.groups()==(x,y)
 if r['source']=='corrections':
  field=re.escape(o['second_field']);patterns=[rf'A later message for {p} replaces ([^.]*) with ([^.]*)\.',rf'{p} subsequently changed the {field} to ([^.]*); it had initially been ([^.]*)\.',rf'The first arrangement for {p} used ([^.]*)\. This was superseded by ([^.]*),']
  matches=[(i,re.search(pat,doc)) for i,pat in enumerate(patterns)];matches=[(i,m) for i,m in matches if m];assert len(matches)==1
  i,m=matches[0];assert m.groups()==((t,y) if i==1 else (y,t))

if __name__=='__main__':print(json.dumps(build(),indent=2))
