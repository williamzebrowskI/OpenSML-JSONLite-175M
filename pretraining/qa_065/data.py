"""SFT-065 revised 25k corpus; frozen original QA plus fact-oracle interventions."""
import collections,json
from pathlib import Path
import numpy as np
from qa_064.data import ROOT,PARENT,PARENT_SHA,TOK,PLAN as OLD_PLAN,make,load,h,normalized
from sml_v1.tokenization import Tokenizer
from sft.data import serialize
from json_sft.corpus import file_sha
from sml_v1.common import atomic_json
CORPUS=ROOT/'pretraining/data/sft_065_targeted_qa_20261013'
OUTPUT=ROOT/'pretraining/runs/sft_065_targeted_qa_20261013'
REPORT=ROOT/'reports/model-design/SFT-065_targeted_qa_20261013'
OLD=ROOT/'pretraining/data/sft_064_grounded_qa_20261013'
PLAN=dict(OLD_PLAN)

def pack(rows,split,tok):
 ids=[];masks=[];offsets=[0]
 for r in rows:
  a,b=serialize(tok,r['messages']);prompt,_=serialize(tok,r['messages'][:-1],generation=True)
  assert len(a)<=1024 and len(prompt)+160<=2048 and len(tok.encode(r['answers'][0]))<=150
  ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
 (CORPUS/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 for key,value in [('tokens',np.concatenate(ids)),('mask',np.concatenate(masks)),('offsets',np.array(offsets,np.int64))]:np.save(CORPUS/f'{split}.{key}.npy',value)
 return dict(records=len(rows),tokens=offsets[-1],buckets=dict(collections.Counter(r['bucket'] for r in rows)),sources=dict(collections.Counter(r['source'] for r in rows)),absent=sum(r['answer_absent'] for r in rows))

def natural():
 rows=load(OLD/'train.jsonl');by=collections.defaultdict(list)
 for r in rows:
  if r['source']=='squad2' and not r['answer_absent']:by[normalized(r['document'])].append(r)
 pairs=[];used=set()
 for doc,rs in sorted(by.items(),key=lambda x:h(x[0])):
  found=None
  for i,a in enumerate(rs):
   for b in rs[i+1:]:
    if not {normalized(x) for x in a['answers']}&{normalized(x) for x in b['answers']}:
     found=(a,b);break
   if found:break
  if not found:continue
  for r in found:
   pairs.append(dict(r,bucket='contrast',training_pair='qa065:'+h(doc)));used.add(r['id'])
  if len(pairs)==5000:break
 assert len(pairs)==5000
 available=[r for r in rows if r['id'] not in used]
 dolly=[r for r in available if r['source']=='dolly_closed_qa']
 absent=[r for r in available if r['answer_absent']][:750]
 positive=[r for r in available if r['source']=='squad2' and not r['answer_absent']][:12000-len(dolly)-len(absent)]
 base=[dict(r,bucket='natural') for r in dolly+absent+positive]
 assert len(base)==12000
 return base+pairs

def check(deep=False):
 m=json.loads((CORPUS/'manifest.json').read_text());assert m['plan']==PLAN
 for name,digest in m['files'].items():assert file_sha(CORPUS/name)==digest,name
 for name,digest in m['sources'].items():assert file_sha(ROOT/name)==digest,name
 assert file_sha(PARENT/'model.safetensors')==PARENT_SHA
 if deep:
  from .controlled import validate,validate_passage
  from sft.data import Corpus
  tok=Tokenizer(TOK)
  for split in m['splits']:
   c=Corpus(CORPUS,split)
   for i,r in enumerate(load(CORPUS/f'{split}.jsonl')):
    if 'oracle' in r:validate(r);validate_passage(r)
    a,b=serialize(tok,r['messages']);start,end=c.offsets[i:i+2]
    assert np.array_equal(a,c.ids[start:end]) and np.array_equal(b,c.mask[start:end])
 return m

def build():
 from .controlled import generate,validate,validate_passage
 assert not CORPUS.exists(),'Refusing overwrite'
 sets={'train':natural()}
 for split in ('development','test'):
  sets[split]=[dict(r,bucket='legacy',panel='legacy') for r in load(OLD/f'{split}.jsonl')]
 sizes={'train':(3000,3000,2000),'development':(16,16,16),'test':(32,32,32)}
 for split in sets:
  for category,count in zip(('calibration','multipart','corrections'),sizes[split]):
   rows=list(generate(split,category,count));assert len(rows)==count
   for r in rows:validate(r);validate_passage(r);r['panel']='targeted'
   sets[split].extend(rows)
 assert len(sets['train'])==25000
 for split,rows in sets.items():
  assert len({r['id'] for r in rows})==len(rows)
  assert len({normalized(r['document'])+'\n'+normalized(r['question']) for r in rows})==len(rows)
  for other,rs in sets.items():
   if split==other:continue
   assert not {r['group'] for r in rows}&{r['group'] for r in rs}
   assert not {normalized(r['document']) for r in rows}&{normalized(r['document']) for r in rs}
   assert not {r['frame_id'] for r in rows if 'frame_id' in r}&{r['frame_id'] for r in rs if 'frame_id' in r}
 # Assert exact preservation of legacy evaluation content, ignoring added panel labels.
 for split in ('development','test'):
  old=load(OLD/f'{split}.jsonl');current=sets[split][:len(old)]
  assert [dict(r, bucket='legacy',panel='legacy') for r in old]==current
 CORPUS.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
 tok=Tokenizer(TOK);splits={s:pack(rs,s,tok) for s,rs in sets.items()}
 sources=[OLD/'manifest.json']+[OLD/f'{s}.jsonl' for s in sets]
 atomic_json(CORPUS/'manifest.json',dict(plan=PLAN,parent=str(PARENT),parent_sha256=PARENT_SHA,splits=splits,files={p.name:file_sha(p) for p in CORPUS.iterdir() if p.is_file()},sources={str(p.relative_to(ROOT)):file_sha(p) for p in sources},limitations='17000 original published QA rows plus8000 deterministic fictional examples across8 domains. Synthetic frames are controlled examples, not independently authored natural documents. Frame IDs held out across splits; structural tasks/domains reused. Legacy evaluation observed previously. Source article exposure in earlier experiments/pretraining unknown. Symbolic checks do not guarantee natural-language generalization. Original SQuAD CC-BY-SA4.0 and Dolly CC-BY-SA3.0 provenance retained.'))
 review={}
 for split in sets:
  review[split]={cat:[r for r in sets[split] if r['source']==cat][:16] for cat in ('calibration','multipart','corrections')}
 atomic_json(REPORT/'review-packet.json',review)
 return check(True)
if __name__=='__main__':print(json.dumps(build(),indent=2))
