"""Balanced extension with reviewed native structures and unquoted prose.

Public examples remain source/schema/gold triples. Views increase schema-task
coverage, not unique source counts. No benchmark prompts or labels are added.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from jsonschema import validators,FormatChecker
from json_specialist.corpus import read_guard,schema_shape
from json_specialist.data import canonical,strict_json
from .corpus import ROOT,file_sha,rank,sha,norm,load_tokenizer,near_cross_split
from .ordered import check as check_previous
from .transfer import validate as validate_base
from .ordered_serialization import serialize,SCHEMA_LAST
from .serialization import PLAIN
from . import unquoted

FORMAT='opensml-json-sft-precise-v10'
UPSTREAM_SHA='c07b20b7e88afc9cde0cf8c92e50ad150861ba2e35f4c82eb85999aba7182f4a'
RAW=ROOT/'pretraining/data/json_native_sources_20261010/hermes-singleturn.json'
REVIEW=ROOT/'reports/model-design/JSON-008_sft-corpus-20261010/native-nested-review.json'

def upstream():
    if file_sha(RAW)!=UPSTREAM_SHA:raise ValueError('Native upstream bytes changed')
    return {r['id']:r for r in strict_json(RAW.read_text())}

def close_schema(value):
    value=copy.deepcopy(value)
    def visit(node):
        if not isinstance(node,dict):return
        if '$ref' in node:raise ValueError('Native references not admitted')
        if node.get('type')=='object':
            node['additionalProperties']=False
            for child in node['properties'].values():visit(child)
        if node.get('type')=='array':visit(node['items'])
    visit(value);return value

def native_view(schema,target,mode):
    schema=close_schema(schema);keys=sorted(schema['properties']);mapping={k:k for k in keys}
    if mode=='subset':
        keys=keys[:max(1,len(keys)//2)];schema['properties']={k:schema['properties'][k] for k in keys};schema['required']=[k for k in schema.get('required',[]) if k in keys]
    elif mode=='alias':
        mapping={k:'field_'+str(i+1) for i,k in enumerate(keys)}
        for k in keys:
            field=schema['properties'][k];field['description']=field.get('description','')+' Extract the source '+field.get('title',k)+' field ('+k+').'
        schema['properties']={mapping[k]:schema['properties'][k] for k in keys};schema['required']=[mapping[k] for k in schema.get('required',[])]
    elif mode not in ('original','nested'):raise ValueError('Unknown native view')
    answer={mapping[k]:copy.deepcopy(target[k]) for k in keys if k in target}
    if mode=='nested':
        schema=dict(type='object',properties={'record':schema},required=['record'],additionalProperties=False);answer={'record':answer}
    return schema,answer

def native_row(review,original,split,mode):
    turns=original['conversations'];schema=strict_json(re.search(r'<schema>\s*(.*?)\s*</schema>',turns[0]['value'],re.S).group(1))
    document=next(t['value'] for t in turns if t['from']=='human');target=strict_json(next(t['value'] for t in turns if t['from']=='gpt'))
    if review['decision']!='accept_pending_admission' or [document,schema,target]!=[review['document'],review['schema'],review['target']]:raise ValueError('Native review/upstream mismatch')
    for ev in review['evidence']:
        value=target
        for key in ev['path']:value=value[key]
        if value!=ev['value'] or document[ev['start']:ev['end']]!=value:raise ValueError('Native evidence changed')
    new_schema,answer=native_view(schema,target,mode)
    return dict(id='native-nested:'+review['id']+':'+mode,source='native_nested',source_group='hermes:'+sha(canonical(schema_shape(schema))),split=split,
        document=document,schema=new_schema,target=answer,
        instruction='Extract the fields requested by this schema from the supplied source. Preserve complete explicit values, case, source order and entity associations. Return only JSON.',
        oracle=dict(source_id=review['id'],view=mode),provenance=dict(dataset='NousResearch/hermes-function-calling-v1',revision='dae3e1d28cfbcf4b915c04ea1e072030529b4bda',source_file_sha256=UPSTREAM_SHA,declared_license='Apache-2.0',attribution='NousResearch Hermes function-calling authors',quality='Individually source-reviewed synthetic native schema/text/JSON; not independent human or external factual certification.',conversion='Input schema closed recursively; original, alias, subset or outer nesting derives solely from supplied schema. Targets project unchanged original labels.'))

def validate(row,tokenizer,guard,reviews,originals):
    source=row['source']
    if source not in ('native_nested','unquoted_transfer'):validate_base(row,tokenizer,guard)
    else:
        if sha(norm(row['document'])) in guard['sources'] or (row['split']=='train' and sha(norm(row['document'])) in guard.get('external_sources',set())):raise ValueError('Reserved source')
        if sha(canonical(schema_shape(row['schema']))) in guard['schemas']:raise ValueError('Reserved schema')
        validators.validator_for(row['schema'])(row['schema'],format_checker=FormatChecker()).validate(row['target'])
        if source=='unquoted_transfer':unquoted.verify(row)
        else:
            o=row['oracle'];expected=native_row(reviews[o['source_id']],originals[o['source_id']],row['split'],o['view'])
            for key in ('id','source','source_group','document','schema','instruction','target','provenance'):
                if canonical(row[key])!=canonical(expected[key]):raise ValueError('Native conversion differs: '+key)
    if row.get('_prompt_style') not in (PLAIN,SCHEMA_LAST):raise ValueError('Unsupported extraction style')
    return serialize(tokenizer,row)

def prepare(output,style=PLAIN,per_task=1500):
    base=ROOT/'pretraining/data/json_004_slots_balanced_v5';old=check_previous(base);output=Path(output)
    if output.exists():raise ValueError('Output exists')
    receipt=strict_json(REVIEW.read_text());reviews={r['id']:r for r in receipt['records'] if r['decision']=='accept_pending_admission'};originals=upstream()
    rows=[strict_json(s) for split in ('train','development','test') for s in (base/(split+'.jsonl')).open()]
    native=sorted(reviews,key=lambda k:rank('native-nested:'+k));assign={key:'development' if i<6 else 'test' if i<12 else 'train' for i,key in enumerate(native)}
    existing={r['source_group']:r['split'] for r in rows}
    for key in native:
        split=assign[key];r=native_row(reviews[key],originals[key],split,'original');split=existing.get(r['source_group'],split)
        for mode in (('original','alias','nested','subset') if split=='train' else ('original',)):
            rows.append(native_row(reviews[key],originals[key],split,mode))
    rows.extend(unquoted.records(per_task));rows=[dict(r,_prompt_style=style) for r in rows];rows,removed=near_cross_split(rows,drop=True)
    tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((base/'reserved_public_eval_sources.json').read_text()))
    with tempfile.TemporaryDirectory(prefix='.json-precise-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']));arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    a,b=validate(row,tokenizer,guard,reviews,originals);arrays.append(a);masks.append(b);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays);mask=np.concatenate(masks)
            for name,value in [('tokens',tokens),('mask',mask),('offsets',np.asarray(offsets,np.uint64))]:np.save(stage/(split+'.'+name+'.npy'),value,allow_pickle=False)
            stats[split]=dict(records=len(subset),distinct_documents=len({norm(r['document']) for r in subset}),sources=dict(Counter(r['source'] for r in subset)),total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json'):shutil.copyfile(base/name,stage/name)
        shutil.copyfile(REVIEW,stage/'native-review.json');(stage/'admission-audit.json').write_text(json.dumps(dict(near_split_exclusions=removed,native_source_documents=len(native),native_views_are_not_new_documents=True),indent=2)+'\n')
        m=dict(old,format=FORMAT,data_revision='precise-v10',serialization=style,splits=stats,inputs=dict(base_manifest=file_sha(base/'manifest.json'),native_upstream=str(RAW),native_upstream_sha256=UPSTREAM_SHA,per_task=per_task),
            scope=old['scope']+['Adds source-reviewed nested/array Hermes records, keeping native documents/families together; views do not increase source counts.','Adds unquoted finite relationship layouts with independent source parsers. Existing held-out labels unchanged; new native and prose holdouts reserved before training.'],
            code={**old['code'],**{str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__),Path(unquoted.__file__),ROOT/'pretraining/json_sft/ordered_serialization.py',ROOT/'pretraining/json_sft/narrative.py')}},files={p.name:file_sha(p) for p in stage.iterdir()})
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
        tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((root/'reserved_public_eval_sources.json').read_text()));reviews={r['id']:r for r in strict_json((root/'native-review.json').read_text())['records']};originals=upstream();allrows=[]
        for split in ('train','development','test'):
            rows=[strict_json(s) for s in (root/(split+'.jsonl')).open()];tokens=np.load(root/(split+'.tokens.npy'),mmap_mode='r');mask=np.load(root/(split+'.mask.npy'),mmap_mode='r');offsets=np.load(root/(split+'.offsets.npy'),mmap_mode='r')
            if len(rows)!=m['splits'][split]['records'] or len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or len(mask)!=len(tokens):raise ValueError('Bad arrays')
            for i,row in enumerate(rows):
                if row['split']!=split or row.get('_prompt_style')!=m['serialization']:raise ValueError('Wrong split/style')
                x,y=validate(row,tokenizer,guard,reviews,originals);a,b=map(int,offsets[i:i+2])
                if not np.array_equal(x,tokens[a:b]) or not np.array_equal(y,mask[a:b]):raise ValueError('Serialization drift')
                allrows.append(row)
        near_cross_split(allrows)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--style',choices=(PLAIN,SCHEMA_LAST),default=PLAIN);p.add_argument('--per-task',type=int,default=1500);a=p.parse_args();print(json.dumps(prepare(a.output,a.style,a.per_task)['splits'],indent=2))
