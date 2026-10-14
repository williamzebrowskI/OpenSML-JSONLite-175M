"""Balanced business extraction plus natural-passage supervision, version four."""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from json_specialist.corpus import read_guard
from json_specialist.data import canonical,strict_json,serialize
from jsonschema import validators,FormatChecker
from .corpus import ROOT,file_sha,load_tokenizer,near_cross_split,rank,sha,norm
from .expansion import check as check_expansion,validate as validate_expansion,squad_row
from . import business

FORMAT='opensml-json-sft-transfer-v4'

def span_contract(row):
    row=copy.deepcopy(row)
    def visit(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key=='description' and isinstance(item,str):value[key]=item.replace('the shortest answering span','an answering span')
                else:visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(row['schema']);return row

def validate(row,tokenizer,guard):
    if row['source']=='squad_extract':
        o=row['oracle'];original=squad_row(row['document'],o['annotations'],row['split'],o['variant'])
        validate_expansion(original,tokenizer,guard)
        expected=span_contract(original)
        for key in ('document','schema','target','instruction','evidence','source_group'):
            if canonical(row[key])!=canonical(expected[key]):raise ValueError('QA span contract mismatch')
        return serialize(tokenizer,row)
    if row['source']!='business_transfer':return validate_expansion(row,tokenizer,guard)
    if sha(norm(row['document'])) in guard['sources']:raise ValueError('Reserved source')
    if row['split']=='train' and sha(norm(row['document'])) in guard.get('external_sources',set()):raise ValueError('Published evaluation source')
    from json_specialist.corpus import schema_shape
    if sha(canonical(schema_shape(row['schema']))) in guard['schemas']:raise ValueError('Reserved schema')
    v=validators.validator_for(row['schema']);v.check_schema(row['schema']);v(row['schema'],format_checker=FormatChecker()).validate(row['target'])
    business.verify(row);return serialize(tokenizer,row)

def prepare(output,qa_count=4000,business_count=800):
    base=ROOT/'pretraining/data/json_002_expanded_v3';old=check_expansion(base)
    rows=[]
    for split in ('train','development','test'):
        qa=0
        for line in (base/(split+'.jsonl')).open():
            r=strict_json(line)
            if split=='train' and r['source']=='squad_extract':
                if qa>=qa_count:continue
                qa+=1
            rows.append(span_contract(r) if r['source']=='squad_extract' else r)
    rows.extend(business.records(business_count));rows,removed=near_cross_split(rows,drop=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Output exists')
    guard=read_guard();guard['external_sources']=set(strict_json((base/'reserved_public_eval_sources.json').read_text()));tokenizer=load_tokenizer()
    with tempfile.TemporaryDirectory(prefix='.json-transfer-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']))
            arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    a,b=validate(row,tokenizer,guard);arrays.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays).astype(np.int32);mask=np.concatenate(masks)
            np.save(stage/(split+'.tokens.npy'),tokens,allow_pickle=False);np.save(stage/(split+'.mask.npy'),mask,allow_pickle=False);np.save(stage/(split+'.offsets.npy'),np.asarray(offsets,np.uint64),allow_pickle=False)
            stats[split]=dict(records=len(subset),distinct_documents=len({norm(r['document']) for r in subset}),sources=dict(Counter(r['source'] for r in subset)),total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json'):shutil.copyfile(base/name,stage/name)
        (stage/'rejections.json').write_text(json.dumps(removed,indent=2)+'\n')
        m=dict(format=FORMAT,tier='reviewed_core',status='experimental_verified_controlled_plus_annotated_natural_not_release_certified',serialization='extract-envelope-output-json-v1',splits=stats,
            tokenizer_sha256=old['tokenizer_sha256'],inputs=dict(base_manifest=file_sha(base/'manifest.json'),qa_count=qa_count,business_per_family=business_count),
            scope=old['scope']+['Added finite business renderer grammar, all targets independently reconstructed from source; synthetic coverage is not a natural-data accuracy guarantee.'],
            code={**old['code'],**{str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),Path(business.__file__))}},files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');stage.rename(output)
    return m

def check(output,full=True):
    root=Path(output);m=strict_json((root/'manifest.json').read_text())
    if m['format']!=FORMAT:return check_expansion(root,full=full)
    for name,digest in m['files'].items():
        if Path(name).name!=name or file_sha(root/name)!=digest:raise ValueError('Changed corpus '+name)
    for name,digest in m['code'].items():
        if not (ROOT/name).resolve().is_relative_to(ROOT/'pretraining') or file_sha(ROOT/name)!=digest:raise ValueError('Changed converter '+name)
    if file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json')!=m['tokenizer_sha256']:raise ValueError('Changed tokenizer')
    if full:
        tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((root/'reserved_public_eval_sources.json').read_text()));rows=[]
        for split in ('train','development','test'):
            items=[strict_json(s) for s in (root/(split+'.jsonl')).open()];tokens=np.load(root/(split+'.tokens.npy'),mmap_mode='r');mask=np.load(root/(split+'.mask.npy'),mmap_mode='r');offsets=np.load(root/(split+'.offsets.npy'),mmap_mode='r')
            if len(items)!=m['splits'][split]['records'] or len(offsets)!=len(items)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or len(mask)!=len(tokens):raise ValueError('Bad arrays')
            for i,row in enumerate(items):
                if row['split']!=split:raise ValueError('Wrong split')
                a,b=map(int,offsets[i:i+2]);x,y=validate(row,tokenizer,guard)
                if not np.array_equal(tokens[a:b],x) or not np.array_equal(mask[a:b],y):raise ValueError('Serialization drift')
                rows.append(row)
        near_cross_split(rows)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--qa-count',type=int,default=4000);p.add_argument('--business-count',type=int,default=800);p.add_argument('--check',action='store_true');a=p.parse_args()
    m=check(a.output) if a.check else prepare(a.output,a.qa_count,a.business_count);print(json.dumps(m['splits'],indent=2))
