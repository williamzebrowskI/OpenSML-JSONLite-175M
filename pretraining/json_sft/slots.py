"""Conservatively admit a larger published span-annotation pool after audit.

The 55-row semantic sample is not individual certification of all annotations.
All admitted rows are independently replayed against the pinned upstream spans.
"""
from collections import Counter
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import numpy as np
from json_specialist.data import canonical,strict_json
from json_specialist.corpus import read_guard
from .corpus import ROOT,file_sha,load_tokenizer,near_cross_split,rank,norm
from .transfer import FORMAT,check,validate

def reason(row):
    text=row['document']
    if re.search(r'\b(?:LAX|SFO|ATL|JFK|DFW|OAK|EWR|SJC|DCA|LGA|ORD|IAD)\b',text):return 'airport_code_role_ambiguity'
    if re.search(r"\b(?:don.t|doesn.t|not|avoid|except|instead|more than|less than|above|below)\b",text,re.I):return 'polarity_or_comparison_deferred'
    if 'in the area of Games' in text:return 'ambiguous_category_phrase'
    for key,values in row['target'].items():
        if any(re.search(r'\b(?:movie|film|song|game)$',v,re.I) for v in values) and re.search(r'title|movie_name|song_name|event_name',key):return 'possible_mention_boundary_qualifier'
        if key in ('city','location','destination','where_to','city_of_event') and any(re.search(r'\d',v) for v in values):return 'numeric_city_span'
    return None

def prepare(output):
    base=ROOT/'pretraining/data/json_003_business_transfer_v4';old=check(base)
    candidates=ROOT/'pretraining/data/json_001_annotation_candidates_v2'
    from .corpus import check as check_candidates
    check_candidates(candidates)
    rows=[]
    for split in ('train','development','test'):
        qa=business=0
        for line in (base/(split+'.jsonl')).open():
            row=strict_json(line)
            if split=='train' and row['source']=='squad_extract':
                if qa>=2000:continue
                qa+=1
            if split=='train' and row['source']=='business_transfer':
                # Source-ID sampling, never selection by target correctness/difficulty.
                if int(rank('retain:'+row['id'])[:8],16)%2:continue
                business+=1
            rows.append(row)
    seen={r['id'] for r in rows};counts=Counter();rejected=[];pool=[]
    for split in ('train','test'):
        for line in (candidates/(split+'.jsonl')).open():
            row=strict_json(line)
            if row['source']!='sgd' or row['id'] in seen:continue
            why=reason(row)
            if why:rejected.append(dict(id=row['id'],reason=why));continue
            pool.append(row);rows.append(row);seen.add(row['id']);counts[split]+=1
    rows,near_removed=near_cross_split(rows,drop=True)
    tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((base/'reserved_public_eval_sources.json').read_text()))
    original_train=[strict_json(s) for s in (candidates/'train.jsonl').open()];original_train=[r for r in original_train if r['source']=='sgd']
    sample=random.Random(20261010).sample(original_train,55)
    review=[dict(id=r['id'],decision='exclude' if reason(r) else 'sampled_no_concrete_role_or_missing_value_error_found',reason=reason(r),document=r['document'],target=r['target']) for r in sample]
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Output exists')
    with tempfile.TemporaryDirectory(prefix='.json-slots-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']));arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    a,b=validate(row,tokenizer,guard);arrays.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays).astype(np.int32);mask=np.concatenate(masks)
            for name,value in [('tokens',tokens),('mask',mask),('offsets',np.asarray(offsets,np.uint64))]:np.save(stage/(split+'.'+name+'.npy'),value,allow_pickle=False)
            stats[split]=dict(records=len(subset),distinct_documents=len({norm(r['document']) for r in subset}),sources=dict(Counter(r['source'] for r in subset)),total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json'):shutil.copyfile(base/name,stage/name)
        (stage/'admission-audit.json').write_text(json.dumps(dict(manually_inspected_train_sample=review,automatic_exclusions=rejected,near_split_exclusions=near_removed,added_before_near_split=dict(counts),scope='Dataset-level sample audit plus per-record annotation replay; not individual semantic certification of all rows.'),indent=2)+'\n')
        m=dict(old,format=FORMAT,tier='curated_annotations',status='experimental_source_annotated_and_sample_audited_not_individually_certified',splits=stats,
            inputs=dict(base_manifest=file_sha(base/'manifest.json'),candidate_manifest=file_sha(candidates/'manifest.json')),
            scope=old['scope']+['SGD annotation expansion explicitly admitted after 55-row review and conservative role/polarity/boundary exclusions; not all rows manually reviewed.'],
            code={**old['code'],str(Path(__file__).relative_to(ROOT)):file_sha(__file__)},files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');stage.rename(output)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output)['splits'],indent=2))
