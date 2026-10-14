"""Controlled prompt-order ablation: retain precisely the V8 triples/splits."""
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from json_specialist.data import canonical,strict_json
from json_specialist.corpus import read_guard
from .corpus import ROOT,file_sha,load_tokenizer,near_cross_split,norm
from .relational import check as check_previous,validate as validate_previous
from .serialization import PLAIN
from . import ordered_serialization as serialization

FORMAT='opensml-json-sft-schema-last-v9'

def prepare(output,base):
    base=Path(base);old=check_previous(base);output=Path(output)
    if output.exists():raise ValueError('Output exists')
    tokenizer=load_tokenizer()
    with tempfile.TemporaryDirectory(prefix='.json-ordered-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            rows=[dict(strict_json(s),_prompt_style=serialization.SCHEMA_LAST) for s in (base/(split+'.jsonl')).open()]
            arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in rows:
                    a,b=serialization.serialize(tokenizer,row);arrays.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays);mask=np.concatenate(masks)
            for name,value in [('tokens',tokens),('mask',mask),('offsets',np.asarray(offsets,np.uint64))]:np.save(stage/(split+'.'+name+'.npy'),value,allow_pickle=False)
            stats[split]=dict(records=len(rows),distinct_documents=len({norm(r['document']) for r in rows}),sources=dict(Counter(r['source'] for r in rows)),total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json','admission-audit.json'):shutil.copyfile(base/name,stage/name)
        m=dict(old,format=FORMAT,data_revision='schema-last-v9',serialization=serialization.SCHEMA_LAST,splits=stats,
            inputs=dict(base_manifest=file_sha(base/'manifest.json'),base=str(base.resolve())),
            scope=old['scope']+['Identical V8 source/schema/target labels and order; source text precedes task/schema, placing field instructions nearer output.'],
            code={**old['code'],**{str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),Path(serialization.__file__))}},files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');stage.rename(output)
    return m

def check(output,full=True):
    root=Path(output);m=strict_json((root/'manifest.json').read_text())
    if m['format']!=FORMAT:return check_previous(root,full=full)
    for name,digest in m['files'].items():
        if Path(name).name!=name or file_sha(root/name)!=digest:raise ValueError('Changed corpus '+name)
    for name,digest in m['code'].items():
        if not (ROOT/name).resolve().is_relative_to(ROOT/'pretraining') or file_sha(ROOT/name)!=digest:raise ValueError('Changed converter '+name)
    if m['tokenizer_sha256']!=file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'):raise ValueError('Changed tokenizer')
    if full:
        base=Path(m['inputs']['base'])
        if file_sha(base/'manifest.json')!=m['inputs']['base_manifest']:raise ValueError('Base identity changed')
        check_previous(base)
        tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((root/'reserved_public_eval_sources.json').read_text()));allrows=[]
        for split in ('train','development','test'):
            rows=[strict_json(s) for s in (root/(split+'.jsonl')).open()];original=[strict_json(s) for s in (base/(split+'.jsonl')).open()]
            tokens=np.load(root/(split+'.tokens.npy'),mmap_mode='r');mask=np.load(root/(split+'.mask.npy'),mmap_mode='r');offsets=np.load(root/(split+'.offsets.npy'),mmap_mode='r')
            if len(rows)!=len(original) or len(rows)!=m['splits'][split]['records'] or len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or len(mask)!=len(tokens):raise ValueError('Bad arrays')
            for i,row in enumerate(rows):
                if row['split']!=split or row.get('_prompt_style')!=serialization.SCHEMA_LAST:raise ValueError('Wrong split/style')
                old=dict(row,_prompt_style=PLAIN)
                if canonical(old)!=canonical(original[i]):raise ValueError('Prompt-order ablation changed a record')
                validate_previous(old,tokenizer,guard);x,y=serialization.serialize(tokenizer,row);a,b=map(int,offsets[i:i+2])
                if not np.array_equal(x,tokens[a:b]) or not np.array_equal(y,mask[a:b]):raise ValueError('Serialization drift')
                allrows.append(row)
        near_cross_split(allrows)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(a.output,a.base)['splits'],indent=2))
