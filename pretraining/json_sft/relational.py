"""Versioned relational-prose supplement, preserving previous frozen corpora."""
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from jsonschema import validators,FormatChecker
from json_specialist.corpus import read_guard,schema_shape
from json_specialist.data import canonical,strict_json
from .corpus import ROOT,file_sha,rank,norm,sha,near_cross_split,load_tokenizer
from .plain import check as check_previous
from .transfer import validate as validate_previous
from .serialization import serialize,PLAIN
from . import narrative
FORMAT='opensml-json-sft-relational-v8'

def validate(row,tokenizer,guard):
    if row['source']!='narrative_transfer':validate_previous(row,tokenizer,guard)
    else:
        if sha(norm(row['document'])) in guard['sources'] or (row['split']=='train' and sha(norm(row['document'])) in guard.get('external_sources',set())):raise ValueError('Reserved source')
        if sha(canonical(schema_shape(row['schema']))) in guard['schemas']:raise ValueError('Reserved schema')
        v=validators.validator_for(row['schema']);v.check_schema(row['schema']);v(row['schema'],format_checker=FormatChecker()).validate(row['target']);narrative.verify(row)
    if row.get('_prompt_style')!=PLAIN:raise ValueError('Wrong extraction style')
    return serialize(tokenizer,row)

def prepare(output,base,per_task=600):
    base=Path(base);old=check_previous(base);rows=[strict_json(s) for split in ('train','development','test') for s in (base/(split+'.jsonl')).open()]
    rows.extend(dict(r,_prompt_style=PLAIN) for r in narrative.records(per_task));rows,removed=near_cross_split(rows,drop=True)
    tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((base/'reserved_public_eval_sources.json').read_text()))
    output=Path(output)
    if output.exists():raise ValueError('Output exists')
    with tempfile.TemporaryDirectory(prefix='.json-relational-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']));arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    a,b=validate(row,tokenizer,guard);arrays.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays);mask=np.concatenate(masks)
            for name,value in [('tokens',tokens),('mask',mask),('offsets',np.asarray(offsets,np.uint64))]:np.save(stage/(split+'.'+name+'.npy'),value,allow_pickle=False)
            stats[split]=dict(records=len(subset),distinct_documents=len({norm(r['document']) for r in subset}),sources=dict(Counter(r['source'] for r in subset)),total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json'):shutil.copyfile(base/name,stage/name)
        (stage/'admission-audit.json').write_text(json.dumps(dict(near_split_exclusions=removed,scope='Added six finite relational prose families, separate source parsers; source facts are procedural, not natural-document golds.'),indent=2)+'\n')
        m=dict(old,format=FORMAT,data_revision='relational-v8',splits=stats,inputs=dict(base_manifest=file_sha(base/'manifest.json'),per_task=per_task),
            scope=old['scope']+['Finite relational prose supplement; train layouts 0–3, dev 4, test 5. Quoted value boundaries; arbitrary unquoted natural prose remains a separate generalization problem.'],
            code={**old['code'],**{str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),Path(narrative.__file__))}},files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');stage.rename(output)
    return m

def check(output,full=True):
    root=Path(output);m=strict_json((root/'manifest.json').read_text())
    if m['format']!=FORMAT:return check_previous(root,full=full)
    for name,digest in m['files'].items():
        if Path(name).name!=name or file_sha(root/name)!=digest:raise ValueError('Changed corpus '+name)
    for name,digest in m['code'].items():
        if not (ROOT/name).resolve().is_relative_to(ROOT/'pretraining') or file_sha(ROOT/name)!=digest:raise ValueError('Changed converter '+name)
    if file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json')!=m['tokenizer_sha256']:raise ValueError('Changed tokenizer')
    if full:
        tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((root/'reserved_public_eval_sources.json').read_text()));allrows=[]
        for split in ('train','development','test'):
            rows=[strict_json(s) for s in (root/(split+'.jsonl')).open()];tokens=np.load(root/(split+'.tokens.npy'),mmap_mode='r');mask=np.load(root/(split+'.mask.npy'),mmap_mode='r');offsets=np.load(root/(split+'.offsets.npy'),mmap_mode='r')
            if len(rows)!=m['splits'][split]['records'] or len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or len(mask)!=len(tokens):raise ValueError('Bad arrays')
            for i,row in enumerate(rows):
                if row['split']!=split:raise ValueError('Wrong split')
                x,y=validate(row,tokenizer,guard);a,b=map(int,offsets[i:i+2])
                if not np.array_equal(x,tokens[a:b]) or not np.array_equal(y,mask[a:b]):raise ValueError('Serialization drift')
                allrows.append(row)
        near_cross_split(allrows)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--per-task',type=int,default=600);a=p.parse_args();print(json.dumps(prepare(a.output,a.base,a.per_task)['splits'],indent=2))
