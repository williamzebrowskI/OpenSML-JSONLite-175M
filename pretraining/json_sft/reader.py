"""Add short, single-question published reading examples without changing evals.

This is annotation-backed supervision, not individually certified natural gold.
Questions define schemas before the answer is read. Article/title grouping and
source near-duplicate checks preserve all existing held-out evaluation sources.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
import pyarrow.parquet as pq
from json_specialist.data import canonical,strict_json
from json_specialist.corpus import read_guard
from .corpus import ROOT,file_sha,rank,norm,load_tokenizer,near_cross_split
from .expansion import squad_row,REVISION
from .transfer import span_contract,validate as validate_labels
from .plain import check
from .serialization import serialize,PLAIN

def admission(q):
    if q['id'] in {'572773e45951b619008f8a3d','5727ce452ca10214002d96e0'}:return 'reviewed_role_or_incomplete_scope'
    if not 100<=len(q['context'])<=1200:return 'length'
    if not re.match(r'^(Who|What|When|Where|Which|How many|How much)\b',q['question']):return 'question_scope'
    if re.search(r"\b(?:not|never|didn.t|doesn.t|least|except)\b",q['question'],re.I):return 'complex_polarity'
    texts=list(q['answers']['text'])
    if texts and (len(set(texts))!=1 or len(texts[0].split())>12 or len(texts[0])>100):return 'answer_boundary_or_length'
    return None

def prepare(output,count=10000):
    base=ROOT/'pretraining/data/json_005_plain_envelope_v6';old=check(base)
    raw=ROOT/'pretraining/data/json_002_sources_20261010';receipt=strict_json((raw/'receipt.json').read_text())
    if receipt['revision']!=REVISION:raise ValueError('Changed source revision')
    path=raw/'train-00000-of-00001.parquet'
    if file_sha(path)!=receipt['files'][path.name]['sha256']:raise ValueError('Changed source bytes')
    grouped=defaultdict(list);rejected=Counter()
    for q in pq.read_table(path).to_pylist():
        if int(rank('squad-title:'+q['title'])[:8],16)%100<16:continue
        why=admission(q)
        if why:rejected[why]+=1;continue
        grouped[q['context']].append(q)
    tokenizer=load_tokenizer()
    rows=[strict_json(line) for split in ('train','development','test') for line in (base/(split+'.jsonl')).open()]
    seen={r['id'] for r in rows};added=[];nulls=0
    for context,questions in sorted(grouped.items(),key=lambda kv:rank('single:'+kv[0])):
        if len(added)>=count:break
        for q in sorted(questions,key=lambda q:rank('single:'+q['id'])):
            if not q['answers']['text'] and nulls>=count//5:continue
            row=span_contract(squad_row(context,[q],'train'))
            if row['id'] in seen:continue
            row['_prompt_style']=PLAIN;serialize(tokenizer,row)
            added.append(row);seen.add(row['id']);nulls+=not bool(q['answers']['text']);break
    rows.extend(added);rows,removed=near_cross_split(rows,drop=True)
    tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((base/'reserved_public_eval_sources.json').read_text()))
    output=Path(output)
    if output.exists():raise ValueError('Output exists')
    with tempfile.TemporaryDirectory(prefix='.json-reader-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']));arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    validate_labels(row,tokenizer,guard);a,b=serialize(tokenizer,row);arrays.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays);mask=np.concatenate(masks)
            for name,value in [('tokens',tokens),('mask',mask),('offsets',np.asarray(offsets,np.uint64))]:np.save(stage/(split+'.'+name+'.npy'),value,allow_pickle=False)
            stats[split]=dict(records=len(subset),distinct_documents=len({norm(r['document']) for r in subset}),sources=dict(Counter(r['source'] for r in subset)),total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json'):shutil.copyfile(base/name,stage/name)
        (stage/'admission-audit.json').write_text(json.dumps(dict(added_before_near_split=len(added),unanswerable_added=nulls,rejections=dict(rejected),near_split_exclusions=removed,scope='Published answer annotations replayed; no individual certification of all labels. One new question per passage; all existing dev/test triples unchanged.'),indent=2)+'\n')
        m=dict(old,data_revision='single-reader-v7',splits=stats,inputs=dict(base_manifest=file_sha(base/'manifest.json'),source_receipt=receipt),
            scope=old['scope']+['Added single-question short reading examples from training article groups only; exact spans, answer-boundary consensus and bounded unanswerable proportion.'],
            code={**old['code'],str(Path(__file__).relative_to(ROOT)):file_sha(__file__)},files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');stage.rename(output)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=10000);a=p.parse_args();print(json.dumps(prepare(a.output,a.count)['splits'],indent=2))
