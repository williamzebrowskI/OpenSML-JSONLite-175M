"""Versioned extraction expansion: source annotations, never model-generated golds.

Published QA is adapted into supplied-schema extraction, not conversational QA.
Schemas are chosen using question text/ID before looking at answer annotations.
The original reviewed core and its held-out sources remain immutable.
"""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import random
import shutil
import tempfile
import numpy as np
import pyarrow.parquet as pq
from jsonschema import FormatChecker, validators
from json_specialist.data import canonical, serialize, strict_json
from json_specialist.corpus import read_guard, schema_shape
from .corpus import ROOT, check as check_core, file_sha, load_tokenizer, near_cross_split, norm, rank, replay as replay_core, sha

FORMAT='opensml-json-sft-expanded-v3'
REVISION='3ffb306f725f7d2ce8394bc1873b24868140c412'

def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

def project(original, mode):
    """Schema shape derives from input schema only; values are projected later."""
    properties=copy.deepcopy(original['schema']['properties'])
    for key,field in properties.items():
        if not field.get('description'):field['description']='Extract the '+field.get('title',key)+' (source field '+key+').'
    keys=sorted(properties)
    aliases={key:('field_'+str(i+1) if mode=='alias' else key) for i,key in enumerate(keys)}
    if mode=='array':
        properties={k:dict(type='array',description=v.get('description',k)+' Return a one-element array for a stated value; [] if absent.',items=dict(type='string')) for k,v in properties.items()}
    elif mode=='first':
        properties={k:dict(type=['string','null'],description=v.get('description',k).replace('Use [] when absent.','')+' Return the first explicitly mentioned value as a string, or null if absent.') for k,v in properties.items()}
    schema=obj({aliases[k]:properties[k] for k in keys})
    if mode=='nested':schema=obj({'record':schema})
    target=copy.deepcopy(original['target'])
    if mode=='array':target={k:[v] for k,v in target.items()}
    elif mode=='first':target={k:(v[0] if v else None) for k,v in target.items()}
    target={aliases[k]:v for k,v in target.items()}
    if mode=='nested':target={'record':target}
    return schema,target

def squad_row(context, annotations, split, variant=0):
    key=':'.join(q['id'] for q in annotations)+':'+str(variant)
    mode=('scalar','array','nested')[int(rank(key)[:8],16)%3]
    names=('answer','result','value','detail','text','extracted')
    properties={}; paths=[]
    # No answers are consulted in schema construction.
    for i,q in enumerate(annotations):
        name=names[(int(rank(q['id'])[:8],16)+i)%len(names)]
        while name in properties:name+='_'+str(i+1)
        desc=q['question']+' Copy the shortest answering span verbatim from this passage only; do not use outside knowledge.'
        field=(dict(type='array',items=dict(type='string'),description=desc+' Use [] if the passage does not answer this question.')
               if mode=='array' else dict(type=['string','null'],description=desc+' Use null if the passage does not answer this question.'))
        properties[name]=field;paths.append(name)
    schema=obj(properties)
    if mode=='nested':schema=obj({'fields':schema})
    target={}
    evidence=[]
    for name,q in zip(paths,annotations):
        answers=q['answers']; texts=list(answers['text']); starts=list(answers['answer_start'])
        if len(texts)!=len(starts):raise ValueError('Answer annotation length')
        for text,start in zip(texts,starts):
            if not text.strip() or context[int(start):int(start)+len(text)]!=text:raise ValueError('Answer span mismatch')
        value=texts[0] if texts else None
        if value and (len(value)>160 or len(value.split())>24):raise ValueError('Long answer deferred')
        target[name]=([value] if value is not None else []) if mode=='array' else value
        if value is not None:evidence.append(dict(path=name,start=int(starts[0]),end=int(starts[0])+len(value),text=value))
    if mode=='nested':target={'fields':target}
    return dict(id='squad-json:'+key,document=context,schema=schema,target=target,
        instruction='Fill every required field using its question and the supplied passage. Preserve exact source text, capitalization, and punctuation. Return only JSON; use the schema\'s null or empty-array convention for unanswered questions.',
        source='squad_extract',source_group='squad-title:'+annotations[0]['title'],split=split,
        provenance=dict(dataset='rajpurkar/squad_v2',revision=REVISION,license='CC-BY-SA-4.0',attribution='SQuAD authors and contributing Wikipedia authors',
            conversion='Question-defined schema, exact human answer spans or published unanswerable label; no target-derived schema.',
            quality='Human annotations with full span replay and sampled semantic review, not every label re-certified.'),
        oracle=dict(annotations=annotations,variant=variant),evidence=evidence)

def validate(row,tokenizer,guard):
    if sha(norm(row['document'])) in guard['sources']:raise ValueError('Reserved evaluation source')
    if row['split']=='train' and sha(norm(row['document'])) in guard.get('external_sources',set()):raise ValueError('Published evaluation source')
    if sha(canonical(schema_shape(row['schema']))) in guard['schemas']:raise ValueError('Reserved evaluation schema')
    schema=row['schema'];v=validators.validator_for(schema);v.check_schema(schema)
    v(schema,format_checker=FormatChecker()).validate(row['target'])
    if row['source']=='squad_extract':
        rebuilt=squad_row(row['document'],row['oracle']['annotations'],row['split'],row['oracle']['variant'])
        for name in ('id','document','schema','target','instruction','evidence','source_group'):
            if canonical(row[name])!=canonical(rebuilt[name]):raise ValueError('QA conversion mismatch: '+name)
    elif row['source'] in ('public_projected','slots_projected'):
        original=row['original_record'];replay_core(original)
        if original['split']!=row['split'] or original['document']!=row['document']:raise ValueError('Projection source changed')
        schema,target=project(original,row['projection'])
        if canonical([row['schema'],row['target']])!=canonical([schema,target]):raise ValueError('Projection changed')
    else:replay_core(row)
    return serialize(tokenizer,row)

def write(output,rows,core,inputs):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Corpus already exists')
    rows,removed=near_cross_split(rows,drop=True)
    tokenizer=load_tokenizer();guard=read_guard()
    guard['external_sources']=set(strict_json((core/'reserved_public_eval_sources.json').read_text()))
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate IDs')
    with tempfile.TemporaryDirectory(prefix='.json-expanded-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']))
            arrays=[];masks=[];offsets=[0]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    a,m=validate(row,tokenizer,guard);arrays.append(a);masks.append(m);offsets.append(offsets[-1]+len(a));f.write(canonical(row)+'\n')
            tokens=np.concatenate(arrays).astype(np.int32);mask=np.concatenate(masks)
            np.save(stage/(split+'.tokens.npy'),tokens,allow_pickle=False);np.save(stage/(split+'.mask.npy'),mask,allow_pickle=False)
            np.save(stage/(split+'.offsets.npy'),np.asarray(offsets,np.uint64),allow_pickle=False)
            stats[split]=dict(records=len(subset),distinct_documents=len({norm(r['document']) for r in subset}),sources=dict(Counter(r['source'] for r in subset)),
                total_tokens=len(tokens),target_eos_tokens=int(mask.sum()),prompt_tokens=int((~mask).sum()),max_total_tokens=max(map(len,arrays)))
        for name in ('extraction-smoke-10.json','jsonschemabench-structural-smoke-10.json','reserved_public_eval_sources.json'):shutil.copyfile(core/name,stage/name)
        (stage/'rejections.json').write_text(json.dumps(removed,indent=2)+'\n')
        m=dict(format=FORMAT,tier='reviewed_core',status='experimental_annotated_expansion_not_release_certified',seed=20261010,serialization='extract-envelope-output-json-v1',
            splits=stats,inputs=inputs,tokenizer_sha256=file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'),
            scope=['Original held-out business and slot examples unchanged. SQuAD title groups disjoint across splits.',
                'SQuAD is adapted grounded extraction supervision, not native text/schema/JSON and not external factual QA.',
                'All targets replay annotations/source spans. Sampled semantic review cannot certify all published labels.',
                'Schema projections increase task coverage, not unique-source counts. No benchmark or smoke targets trained.'],
            code={str(Path(__file__).relative_to(ROOT)):file_sha(__file__),**strict_json((core/'manifest.json').read_text())['code']},
            files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');stage.rename(output)
    return m

def prepare(output,qa_count=8000):
    core=ROOT/'pretraining/data/json_001_curated_v2';check_core(core)
    raw=ROOT/'pretraining/data/json_002_sources_20261010';receipt=strict_json((raw/'receipt.json').read_text())
    if receipt['revision']!=REVISION:raise ValueError('Source revision changed')
    for name,info in receipt['files'].items():
        if file_sha(raw/name)!=info['sha256']:raise ValueError('Changed source bytes')
    rows=[]
    for split in ('train','development','test'):
        rows.extend(strict_json(line) for line in (core/(split+'.jsonl')).open())
    for original in list(rows):
        if original['split']!='train':continue
        if original['source']=='reviewed_public':modes=('alias','array','nested')
        elif original['source']=='sgd':modes=('alias','first','nested')
        else:continue
        for mode in modes:
            schema,target=project(original,mode)
            rows.append(dict(original,id=original['id']+':'+mode,schema=schema,target=target,
                instruction='Extract every required field from the supplied source using the schema descriptions. Follow the requested structure and missing-value convention exactly. Do not add outside facts.',
                source='public_projected' if original['source']=='reviewed_public' else 'slots_projected',original_record=original,projection=mode))
    grouped=defaultdict(list)
    for q in pq.read_table(raw/'train-00000-of-00001.parquet').to_pylist():
        if not 100<=len(q['context'])<=2200:continue
        grouped[(q['title'],q['context'])].append(q)
    tokenizer=load_tokenizer();guard=read_guard();counts=Counter();rejected=Counter()
    for (title,context),questions in sorted(grouped.items(),key=lambda kv:rank(kv[0][1])):
        bucket=int(rank('squad-title:'+title)[:8],16)%100
        split='development' if bucket<8 else 'test' if bucket<16 else 'train'
        cap=qa_count if split=='train' else 150
        if counts[split]>=cap:continue
        questions=sorted(questions,key=lambda q:rank(q['id']))
        # 1–3 independent questions from one passage; no duplicate context in another split.
        for start in range(0,len(questions),3):
            if counts[split]>=cap:break
            batch=questions[start:start+3]
            try:
                row=squad_row(context,batch,split);validate(row,tokenizer,guard)
            except ValueError as e:rejected[str(e).split('\n')[0]]+=1;continue
            rows.append(row);counts[split]+=1
    return write(output,rows,core,dict(core_manifest=file_sha(core/'manifest.json'),squad_receipt=receipt,qa_requested=qa_count,qa_counts=dict(counts),qa_rejections=dict(rejected)))

def check(output,full=True):
    root=Path(output);m=strict_json((root/'manifest.json').read_text())
    if m['format']!=FORMAT:return check_core(root,full=full)
    for name,digest in m['files'].items():
        if Path(name).name!=name or file_sha(root/name)!=digest:raise ValueError('Changed corpus file '+name)
    for name,digest in m['code'].items():
        if not (ROOT/name).resolve().is_relative_to(ROOT/'pretraining') or file_sha(ROOT/name)!=digest:raise ValueError('Converter changed '+name)
    if file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json')!=m['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    if full:
        tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((root/'reserved_public_eval_sources.json').read_text()));rows=[]
        for split in ('train','development','test'):
            tokens=np.load(root/(split+'.tokens.npy'),mmap_mode='r');mask=np.load(root/(split+'.mask.npy'),mmap_mode='r');offsets=np.load(root/(split+'.offsets.npy'),mmap_mode='r')
            items=[strict_json(s) for s in (root/(split+'.jsonl')).open()]
            if len(items)!=m['splits'][split]['records'] or len(offsets)!=len(items)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or len(mask)!=len(tokens):raise ValueError('Bad arrays')
            for i,row in enumerate(items):
                if row['split']!=split:raise ValueError('Wrong split')
                a,b=map(int,offsets[i:i+2]);x,y=validate(row,tokenizer,guard)
                if not np.array_equal(x,tokens[a:b]) or not np.array_equal(y,mask[a:b]):raise ValueError('Serialization changed')
                rows.append(row)
        near_cross_split(rows)
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--qa-count',type=int,default=8000);p.add_argument('--check',action='store_true');a=p.parse_args()
    m=check(a.output) if a.check else prepare(a.output,a.qa_count);print(json.dumps(m['splits'],indent=2))
