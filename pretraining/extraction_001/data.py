"""Original paragraph QA; fixed one-field JSON, no schema composition or chat."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
import numpy as np
from json_sft.corpus import ROOT,file_sha,load_tokenizer,norm
from json_sft.ordered_serialization import serialize
from sml_v1.common import atomic_json
CORPUS=ROOT/'pretraining/data/extraction_001_foundation_20261013'
OUTPUT=ROOT/'pretraining/runs/extraction_001_foundation_20261013'
REPORT=ROOT/'reports/model-design/EXTRACTION-001_foundation_20261013'
PARENT=ROOT/'pretraining/runs/research_30b_176m_stage_a_continue_20261009/step_0132005_63f79113c3d2'
PARENT_SHA='5151567e266244c9e32fd6e01775192633abe227320269ecdefe28ce43e9ccc3'
PLAN=dict(seed=20261013,batch_size=32,microbatch=2,epochs=1,missing_presentations=512,peak_lr=1e-5,final_lr=2e-6,warmup_updates=10,grad_clip=1.,weight_decay=.01,betas=[.9,.95],eps=1e-8,eval_steps=[32,64,96,128],max_updates=128,padding_policy='micro_multiple128',adaptation='full')
SCHEMA={'type':'object','properties':{'answer':{'type':['string','null']}},'required':['answer'],'additionalProperties':False}
def rank(s):return hashlib.sha256(('extraction001:'+s).encode()).hexdigest()
def load(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def make_row(q,split):
 vals=q['answers']['text'];value=None if q['is_impossible'] else vals[0]
 return dict(id='extract001:'+q['id'],document=q['context'],instruction='Answer this question using only TEXT: '+q['question']+' Copy one exact source span; if the answer is absent, use null. Do not explain.',schema=SCHEMA,target={'answer':value},accepted_answers=list(dict.fromkeys(vals)) if value is not None else [None],_prompt_style='text-schema-last-v1',source='squad_absent' if value is None else ('quoref' if q.get('source_dataset')=='quoref' else 'squad_supported'),source_group=rank(norm(q['context'])),article=norm(q.get('article_key',q['title']).replace('_',' ')),split=split,provenance={'qid':q['id'],'dataset':q.get('source_dataset','squad2'),'original_split':q.get('source_split','train'),'answer_starts':q['answers']['answer_start'],'license':'CC-BY-4.0' if q.get('source_dataset')=='quoref' else 'CC-BY-SA-4.0'})
def pack(rows,split,tok):
 ids=[];mask=[];offsets=[0]
 for r in rows:
  a,b=serialize(tok,r);ids.append(a);mask.append(b);offsets.append(offsets[-1]+len(a))
 (CORPUS/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 np.save(CORPUS/(split+'.tokens.npy'),np.concatenate(ids));np.save(CORPUS/(split+'.mask.npy'),np.concatenate(mask));np.save(CORPUS/(split+'.offsets.npy'),np.asarray(offsets,np.int64))
 return dict(records=len(rows),sources=dict(Counter(r['source'] for r in rows)),articles=len(set(r['article'] for r in rows)),tokens=offsets[-1],max_length=max(map(len,ids)))
def build():
 if (CORPUS/'manifest.json').exists():return check(True)
 # Only source parsing/filter functions are reused; no historical checkpoint is read.
 from json_evidence_binding import data as source
 tok=load_tokenizer();raw=source.all_records();reviewed_nulls={q['id'] for case in source.mixed_admission()['selected'] for q in case['questions'] if q['is_impossible']};reserved={norm(x.replace('_',' ')) for x in source.permanent_evaluation_articles()};pools=defaultdict(list);reasons=Counter()
 rejectfile=REPORT/'review.json';rejected=set(json.loads(rejectfile.read_text()).get('rejected_qids',[])) if rejectfile.exists() else set()
 for q in raw.values():
  if q['id'] in rejected:reasons['manual rejection']+=1;continue
  issue=source.question_issue(q)
  if issue:reasons[issue]+=1;continue
  r=make_row(q,'candidate')
  if any(a is not None and len(a)>2 and norm(a) in norm(q['question']) for a in r['accepted_answers']):reasons['answer in question']+=1;continue
  # A one-span output must not be derived from multiple nonidentical TRAIN answers.
  if q.get('source_split','train')=='train' and len(set(r['accepted_answers']))>1:reasons['multiple TRAIN spans']+=1;continue
  try:
   a,b=serialize(tok,r)
   if len(a)>1024:raise ValueError('bounded sequence')
  except ValueError:reasons['token budget/roundtrip']+=1;continue
  original=q.get('source_split','train')
  if original=='train':
   if r['article'] in reserved:reasons['permanent evaluation article']+=1;continue
   if r['source']=='squad_absent' and q['id'] not in reviewed_nulls:reasons['unreviewed TRAIN null']+=1;continue
   split='train'
  else:split='development' if int(rank(r['article'])[:8],16)%2==0 else 'test'
  r['split']=split;pools[split,r['source']].append(r)
 quotas={'train':{'squad_supported':3072,'quoref':512,'squad_absent':88},'development':{'squad_supported':32,'quoref':32,'squad_absent':32},'test':{'squad_supported':64,'quoref':64,'squad_absent':64}}
 selected={};seen={};all_eval_articles={r['article'] for (s,_),rows in pools.items() if s!='train' for r in rows}
 for split,groups in quotas.items():
  rows=[];documents=set()
  for group,n in groups.items():
   chosen=[]
   for r in sorted(pools[split,group],key=lambda r:rank(r['id'])):
    if split=='train' and r['article'] in all_eval_articles:continue
    if r['source_group'] in documents:continue
    documents.add(r['source_group']);chosen.append(r)
    if len(chosen)==n:break
   if len(chosen)!=n:raise ValueError(f'Insufficient {split}/{group}: {len(chosen)}/{n}')
   rows+=chosen
  selected[split]=sorted(rows,key=lambda r:rank('order:'+r['id']))
 for a in selected:
  for b in selected:
   if a!=b and set(r['article'] for r in selected[a])&set(r['article'] for r in selected[b]):raise ValueError('Article leakage')
 CORPUS.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
 stats={s:pack(rows,s,tok) for s,rows in selected.items()}
 sourcepaths=[source.SOURCE_PATH,source.EVAL_SOURCE_PATH,*source.QUOREF_PATHS.values(),source.PERMANENT_RESERVATIONS,source.MIXED_REVIEW]
 if rejectfile.exists():sourcepaths.append(rejectfile)
 manifest=dict(format='extraction001-fixed-span-v1',plan=PLAN,parent=str(PARENT),parent_sha256=PARENT_SHA,splits=stats,excluded=dict(reasons),source_files={str(p.relative_to(ROOT)):file_sha(p) for p in sourcepaths},files={p.name:file_sha(p) for p in CORPUS.iterdir() if p.is_file()},scope='Published annotation fidelity and sampled semantic review; not perfect golds or pretraining decontamination. One original paragraph per record, no clipping, no generated QAs. Test is official DEV-derived, not official leaderboard.')
 atomic_json(CORPUS/'manifest.json',manifest)
 # Deterministic TRAIN-only semantic review packet; test labels do not inform training.
 review=[]
 for group in quotas['train']:
  review += sorted([r for r in selected['train'] if r['source']==group],key=lambda r:rank('review:'+r['id']))[:8]
 atomic_json(REPORT/'train-review-packet.json',review)
 return check(True)
def check(deep=False):
 m=json.loads((CORPUS/'manifest.json').read_text())
 for name,h in m['files'].items():
  if file_sha(CORPUS/name)!=h:raise ValueError('Changed corpus '+name)
 for name,h in m['source_files'].items():
  if file_sha(ROOT/name)!=h:raise ValueError('Changed source '+name)
 if m['plan']!=PLAN or m['parent_sha256']!=PARENT_SHA:raise ValueError('Changed plan')
 if deep:
  from sft.data import Corpus
  tok=load_tokenizer();groups={}
  for split in m['splits']:
   c=Corpus(CORPUS,split);rows=load(CORPUS/(split+'.jsonl'));groups[split]=set(r['article'] for r in rows)
   for i,r in enumerate(rows):
    ids,mask=serialize(tok,r);a,b=c.offsets[i:i+2]
    assert np.array_equal(ids,c.ids[a:b]) and np.array_equal(mask,c.mask[a:b])
    assert r['target']['answer'] is None or r['target']['answer'] in r['document']
  assert not groups['train']&groups['development'] and not groups['train']&groups['test'] and not groups['development']&groups['test']
 return m
if __name__=='__main__':print(json.dumps(build(),indent=2))
