"""SFT-064 single-turn grounded QA. Public original targets, no answer rewriting."""
import collections,hashlib,json,re
from pathlib import Path
import numpy as np
from extraction_001.data import ROOT,PARENT,PARENT_SHA,load
from sml_v1.tokenization import Tokenizer
from sft.data import serialize
from json_sft.corpus import file_sha
from sml_v1.common import atomic_json
CORPUS=ROOT/'pretraining/data/sft_064_grounded_qa_20261013'
OUTPUT=ROOT/'pretraining/runs/sft_064_grounded_qa_20261013'
REPORT=ROOT/'reports/model-design/SFT-064_grounded_qa_20261013'
TOK=ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
SQ=ROOT/'pretraining/data/json_036_sources_20261012/squad-train-v2.0.json'
DOLLY=ROOT/'pretraining/data/qa_candidates_20261013/dolly-databricks-dolly-15k.jsonl'
PLAN=dict(seed=20261013,batch_size=32,microbatch=2,epochs=1,peak_lr=1e-5,final_lr=2e-6,warmup_updates=20,grad_clip=1.,weight_decay=.01,betas=[.9,.95],eps=1e-8,eval_steps=[156,313,469,625,782],padding_policy='micro_multiple128')
PREFIX='Answer the question using only the passage. Give a direct answer without adding unsupported information. If the passage does not contain the answer, reply exactly: Not stated.'
def h(x):return hashlib.sha256(x.encode()).hexdigest()
def words(x):return re.findall(r'\w+',x.casefold())
def normalized(x):return ' '.join(words(x))
def split_of(group):
 n=int(h('qa063:'+group)[:8],16)%100
 return 'train' if n<80 else 'development' if n<90 else 'test'
def make(identifier,source,document,q,answers,group,**extra):
 return dict(id=identifier,component='qa',source=source,document=document,question=q,answers=answers,group=group,answer_absent=answers==['Not stated.'],messages=[dict(role='user',content=PREFIX+'\n\nPassage:\n'+document+'\n\nQuestion: '+q),dict(role='assistant',content=answers[0])],**extra)
def eligible(r,tok):
 if not 30<=len(r['document'].split())<=450:return False
 ids,_=serialize(tok,r['messages']);prompt,_=serialize(tok,r['messages'][:-1],generation=True)
 return len(ids)<=1024 and len(prompt)+160<=2048 and len(tok.encode(r['answers'][0]))<=150

def candidates():
 tok=Tokenizer(TOK);pools={s:[] for s in ('train','development','test')};sq=json.loads(SQ.read_text());stats=collections.Counter();passage_splits={};cross_grams=set()
 for article in sq['data']:
  group='squad-article:'+article['title'];split=split_of(group)
  for p in article['paragraphs']:
   doc=p['context'];key=normalized(doc);passage_splits.setdefault(key,set()).add(split)
   ws=words(doc)
   cross_grams.update(h(' '.join(ws[i:i+16]))[:24] for i in range(max(0,len(ws)-15)))
   for q in p['qas']:
    if q['id'] in {'5a7a49f217ab25001a8a047f','5a8a3e4f3b2508001a72a89f','5acf70f877cf76001a684e3f','5a0e220fd7c8500018864544'}:stats['squad_manual_reject']+=1;continue
    absent=q['is_impossible'];anns=q['answers']
    if absent:
     assert not anns;answers=['Not stated.']
    else:
     if not anns or any(doc[a['answer_start']:a['answer_start']+len(a['text'])]!=a['text'] for a in anns):stats['squad_bad_offset']+=1;continue
     answers=list(dict.fromkeys(a['text'] for a in anns))
    row=make('squad:'+q['id'],'squad2',doc,q['question'],answers,group,provenance=dict(dataset='SQuAD2.0',url='https://rajpurkar.github.io/SQuAD-explorer/',file_sha256=file_sha_cached_sq,original_split='train',title=article['title'],annotations=anns,license='CC-BY-SA-4.0',quality='Published human annotation; offset validation is not semantic certification'))
    if eligible(row,tok):pools[split].append(row)
    else:stats['squad_length_excluded']+=1
 # Exact duplicate contexts spanning article splits are excluded from every split.
 bad={k for k,v in passage_splits.items() if len(v)>1}
 for s in pools:pools[s]=[r for r in pools[s] if normalized(r['document']) not in bad]
 stats['cross_article_exact_contexts_excluded']=len(bad)
 # Dolly original responses only. Lexical checks are screening, NOT semantic proof.
 reject={14955,9686,4512,14743,7192,5004,6643,765,14731,7305,13111,6868,5721,11076,9695}
 seen=set()
 for i,r in enumerate(load(DOLLY)):
  if r['category']!='closed_qa':continue
  if i in reject:stats['dolly_manual_reject']+=1;continue
  doc,q,a=r['context'].strip(),r['instruction'].strip(),r['response'].strip()
  if not 25<=len(doc.split())<=200 or not 8<=len(a.split())<=90:stats['dolly_length']+=1;continue
  aw=set(words(a));evidence=set(words(doc)+words(q))
  if len(aw&evidence)/max(1,len(aw))<.9:stats['dolly_low_lexical_support']+=1;continue
  if set(re.findall(r'\b\d+(?:[.,]\d+)*\b',a))-set(re.findall(r'\b\d+(?:[.,]\d+)*\b',doc+' '+q)):stats['dolly_unsupported_number']+=1;continue
  if re.search(r'```|\b(?:today|currently|now|latest|as of|this year|your opinion|you think)\b',q+' '+a,re.I):stats['dolly_time_or_subjective']+=1;continue
  ws=words(doc)
  if any(h(' '.join(ws[j:j+16]))[:24] in cross_grams for j in range(max(0,len(ws)-15))):stats['dolly_squad_overlap']+=1;continue
  key=h(normalized(q)+'\n'+normalized(doc))
  if key in seen:stats['dolly_duplicate']+=1;continue
  seen.add(key);group='dolly-context:'+h(normalized(doc));split=split_of(group)
  row=make(f'dolly:{i}','dolly_closed_qa',doc,q,[a],group,provenance=dict(dataset='databricks/databricks-dolly-15k',revision='bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a',row=i,license='CC-BY-SA-3.0',quality='Original human response, length/lexical/numeric screened; only a disclosed sample receives semantic review, not all records'))
  if eligible(row,tok):pools[split].append(row)
  else:stats['dolly_packed_length']+=1
 return pools,dict(stats)
file_sha_cached_sq=file_sha(SQ)

def select(pools):
 sets={}
 for s in pools:
  allrows=sorted(pools[s],key=lambda r:h(r['id']));dolly=[r for r in allrows if r['source']=='dolly_closed_qa'][:400 if s=='train' else 8 if s=='development' else 16]
  pos=[r for r in allrows if r['source']=='squad2' and not r['answer_absent']];neg=[r for r in allrows if r['source']=='squad2' and r['answer_absent']]
  if s=='train':
   # Keep several contrasting questions per source, but cap dominance of a paragraph.
   old=load(ROOT/'pretraining/data/sft_063_grounded_qa_20261013/train.jsonl')
   chosen=[r for r in old if r['source']=='squad2' and not r['answer_absent']]
   negatives=[r for r in old if r['answer_absent']]
   per=collections.Counter(normalized(r['document']) for r in chosen)
   seen={(normalized(r['document']),normalized(r['question'])) for r in old}
   need=25000-4167-len(dolly)
   for r in pos:
    k=normalized(r['document']);dedup=(k,normalized(r['question']))
    if per[k]>=4 or dedup in seen:continue
    chosen.append(r);per[k]+=1;seen.add(dedup)
    if len(chosen)==need:break
   assert len(chosen)==need
   per=collections.Counter(normalized(r['document']) for r in negatives)
   for r in neg:
    k=normalized(r['document']);dedup=(k,normalized(r['question']))
    if per[k]>=2 or dedup in seen:continue
    negatives.append(r);per[k]+=1;seen.add(dedup)
    if len(negatives)==4167:break
   assert len(negatives)==4167
  else:
   # Explicit disjoint-answer question pairs test sensitivity to the question.
   groups=collections.defaultdict(list)
   for r in pos:groups[normalized(r['document'])].append(r)
   chosen=[];target=32 if s=='development' else 64
   for k,rs in sorted(groups.items(),key=lambda kv:h(kv[0])):
    first=None;second=None
    for a in rs:
     for b in rs:
      aa={normalized(x) for x in a['answers']};bb={normalized(x) for x in b['answers']}
      if a['id']!=b['id'] and normalized(a['question'])!=normalized(b['question']) and not aa&bb:
       first,second=a,b;break
     if first:break
    if not first:continue
    pid=s+':pair:'+h(k)[:16]
    chosen.extend([dict(first,pair_id=pid),dict(second,pair_id=pid)])
    if len(chosen)==2*target:break
   assert len(chosen)==2*target
   used={normalized(r['document']) for r in chosen};negatives=[]
   for r in neg:
    k=normalized(r['document'])
    if k in used:continue
    negatives.append(r);used.add(k)
    if len(negatives)==(16 if s=='development' else 32):break
  sets[s]=chosen+negatives+dolly
 return sets

def pack(rs,split,tok):
 ids=[];masks=[];offsets=[0]
 for r in rs:
  a,b=serialize(tok,r['messages']);assert len(a)<=1024;ids.append(a);masks.append(b);offsets.append(offsets[-1]+len(a))
 (CORPUS/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
 for suffix,v in [('tokens',np.concatenate(ids)),('mask',np.concatenate(masks)),('offsets',np.array(offsets,np.int64))]:np.save(CORPUS/f'{split}.{suffix}.npy',v)
 return dict(records=len(rs),sources=dict(collections.Counter(r['source'] for r in rs)),absent=sum(r['answer_absent'] for r in rs),tokens=offsets[-1],documents=len({normalized(r['document']) for r in rs}),source_groups=len({r['group'] for r in rs}))
def check(deep=False):
 m=json.loads((CORPUS/'manifest.json').read_text());assert m['plan']==PLAN
 for n,v in m['files'].items():assert file_sha(CORPUS/n)==v,n
 for n,v in m['sources'].items():assert file_sha(ROOT/n)==v,n
 assert file_sha(PARENT/'model.safetensors')==PARENT_SHA
 if deep:
  from sft.data import Corpus
  tok=Tokenizer(TOK)
  for split in m['splits']:
   c=Corpus(CORPUS,split)
   for i,r in enumerate(load(CORPUS/f'{split}.jsonl')):
    a,b=serialize(tok,r['messages']);start,end=c.offsets[i:i+2];assert np.array_equal(a,c.ids[start:end]) and np.array_equal(b,c.mask[start:end])
 return m

def build():
 assert not CORPUS.exists(),'Refusing overwrite'
 pools,stats=candidates();sets=select(pools)
 for split in ('development','test'):
  old=load(ROOT/f'pretraining/data/sft_063_grounded_qa_20261013/{split}.jsonl')
  assert sets[split]==old, 'Evaluation changed'
 assert len(sets['train'])==25000
 old_ids={r['id'] for r in load(ROOT/'pretraining/data/sft_063_grounded_qa_20261013/train.jsonl')}
 assert old_ids <= {r['id'] for r in sets['train']}, 'Original training rows not retained'
 for a in sets:
  for b in sets:
   if a==b:continue
   assert not {r['group'] for r in sets[a]}&{r['group'] for r in sets[b]}
   assert not {normalized(r['document']) for r in sets[a]}&{normalized(r['document']) for r in sets[b]}
   assert not {normalized(r['question'])+'\n'+normalized(r['document']) for r in sets[a]}&{normalized(r['question'])+'\n'+normalized(r['document']) for r in sets[b]}
 CORPUS.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True);tok=Tokenizer(TOK)
 splits={s:pack(rs,s,tok) for s,rs in sets.items()}
 sources=[SQ,DOLLY,ROOT/'pretraining/data/qa_candidates_20261013/dolly-provenance.json',ROOT/'pretraining/data/qa_candidates_20261013/dolly-README.md']
 atomic_json(CORPUS/'manifest.json',dict(plan=PLAN,parent=str(PARENT),parent_sha256=PARENT_SHA,splits=splits,screening_counts=stats,files={p.name:file_sha(p) for p in CORPUS.iterdir() if p.is_file()},sources={str(p.relative_to(ROOT)):file_sha(p) for p in sources},limitations='Original published human annotations, not perfect semantic truth. Dolly screened and sample-reviewed only. SQuAD article-disjoint within this run; former experiments/pretraining may have exposed articles, so not a pristine external benchmark. Exact shared paragraphs removed;16word Dolly/SQuAD overlap excluded; no exhaustive fuzzy or pretraining decontamination. QA-only; no entity-list/JSON/Python or long conversation targets. Full answers retained with no truncation or generated expansions.'))
 atomic_json(REPORT/'review-packet.json',{'dolly':[r for r in sets['train'] if r['source']=='dolly_closed_qa'][:40],'squad_answerable':[r for r in sets['train'] if r['source']=='squad2' and not r['answer_absent']][:12],'squad_absent':[r for r in sets['train'] if r['answer_absent']][:12]})
 return check(True)
if __name__=='__main__':print(json.dumps(build(),indent=2))
