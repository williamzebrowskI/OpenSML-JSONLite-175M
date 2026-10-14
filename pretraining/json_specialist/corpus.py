"""CPU-only preparation and validation of a bounded experimental corpus."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import tempfile
import os
import numpy as np
from jsonschema import validators, FormatChecker
from tokenizers import Tokenizer
from .data import canonical, strict_json, serialize
from .synthetic import records
from .verify import verify_controlled

REPO=Path(__file__).resolve().parents[2]
SEEDS=Path(__file__).resolve().parent/'seeds'
TOKENIZER=REPO/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'
TOKENIZER_SHA='e095f0413a11fa91fbb58f37d00e4c6bcda59beb89b28f6f680db1436a7e8891'
SPLITS=('train','development','test')
FORMATS=FormatChecker()

def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(text):return ' '.join(text.casefold().split())
def schema_shape(x):
    if isinstance(x,list):return [schema_shape(y) for y in x]
    if not isinstance(x,dict):return x
    out={}
    for key,value in x.items():
        if key in ('title','description','$comment','examples','default','$schema'):continue
        if key in ('properties','patternProperties','$defs','definitions') and isinstance(value,dict):
            out[key]={name:schema_shape(s) for name,s in value.items()}
        else:out[key]=schema_shape(value)
    return out

def nodes(x):
    if isinstance(x,dict):
        yield x
        for y in x.values():yield from nodes(y)
    elif isinstance(x,list):
        for y in x:yield from nodes(y)

def validate_record(record, tokenizer, guard):
    if not record['document'].strip() or not record['instruction'].strip():raise ValueError('Empty source/instruction')
    if digest(norm(record['document'])) in guard['sources']:raise ValueError('Reserved evaluation source')
    if digest(canonical(schema_shape(record['schema']))) in guard['schemas']:raise ValueError('Reserved evaluation schema')
    schema=record['schema']
    if schema.get('type')!='object' or not schema.get('properties'):raise ValueError('Expected meaningful object schema')
    known_drafts={c.META_SCHEMA['$id'] if '$id' in c.META_SCHEMA else c.META_SCHEMA['id'] for c in (validators.Draft4Validator,validators.Draft6Validator,validators.Draft7Validator,validators.Draft201909Validator,validators.Draft202012Validator)}
    if '$schema' in schema and schema['$schema'] not in known_drafts:raise ValueError('Unresolved schema dialect')
    for node in nodes(schema):
        if '$ref' in node and not node['$ref'].startswith('#'):raise ValueError('External references are not supported')
        if node.get('format') and node['format'] not in FORMATS.checkers:raise ValueError('Unknown schema format')
        if len(node.get('properties',{}))>8:raise ValueError('Schema too wide')
    v=validators.validator_for(schema);v.check_schema(schema)
    v(schema,format_checker=FORMATS).validate(record['target'])
    ids,mask=serialize(tokenizer,record)
    if record['provenance']['kind']=='procedural_synthetic':evidence=verify_controlled(record)
    else:
        if record['review']['decision']!='admit_experimental_seed':raise ValueError('Unreviewed public target')
        evidence=record['review']['evidence']
        if {e['path'] for e in evidence}!={'$.'+k for k in record['target']}:raise ValueError('Incomplete evidence')
        for e in evidence:
            key=e['path'][2:]
            if record['document'][e['start']:e['end']]!=e['value'] or record['target'][key]!=e['value']:
                raise ValueError('Public evidence/target mismatch')
    return ids,mask,evidence

def read_guard():
    data=strict_json((SEEDS/'reserved_evaluation_hashes_v1.json').read_text())
    return {'sources':set().union(*(set(v) for v in data['reserved_source_hashes'].values())),
            'schemas':set(data['reserved_schema_shape_hashes'])}

def check_separation(rows):
    groups=defaultdict(set);sources=defaultdict(set);triples=set();docs={}
    for row in rows:
        split=row['provenance']['split'];provenance=row['provenance']
        for key in ('source_family','template_family'):
            groups[(key,provenance[key])].add(split)
        text=norm(row['document']);sources[digest(text)].add(split)
        triple=canonical([text,row['schema'],row['instruction'],row['target']])
        if triple in triples:raise ValueError('Duplicate normalized training triple')
        triples.add(triple);docs[digest(text)]=(split,text)
    if any(len(s)>1 for s in list(groups.values())+list(sources.values())):raise ValueError('Cross-split source/template overlap')
    # Word five-shingle Jaccard over actual text, not proof of semantic independence.
    inverted=defaultdict(list);shingles={};near=[]
    for key,(split,text) in docs.items():
        tokens=re.findall(r'\w+',text)
        current=set(tuple(tokens[i:i+5]) for i in range(max(1,len(tokens)-4)))
        candidates=Counter(k for s in current for k in inverted[s] if docs[k][0]!=split)
        for other,intersection in candidates.items():
            union=len(current)+len(shingles[other])-intersection
            score=intersection/union if union else 1
            if score>=.85:near.append({'a':other,'b':key,'jaccard':score})
        shingles[key]=current
        for s in current:inverted[s].append(key)
    if near:raise ValueError(f'Cross-split near duplicates at Jaccard >= .85: {len(near)}')
    return {'unique_documents':len(docs),'cross_split_exact_sources':0,'cross_split_source_or_template_families':0,
            'cross_split_word5_jaccard_at_least_0_85':0,'near_duplicate_scope':'Actual-text word five-shingle Jaccard only; controlled grammar remains shared across layouts.'}

def load_tokenizer():
    if file_digest(TOKENIZER)!=TOKENIZER_SHA:raise ValueError('Frozen tokenizer changed')
    t=Tokenizer.from_file(str(TOKENIZER));t.no_padding();t.no_truncation();t.encode_special_tokens=True
    return t

def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Output exists; validate with --check instead of overwriting')
    rows=list(records())
    rows += [strict_json(line) for line in (SEEDS/'hermes_reviewed_v1.jsonl').read_text().splitlines()]
    separation=check_separation(rows);tokenizer=load_tokenizer();guard=read_guard()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.json-corpus-',dir=output.parent) as temporary:
        stage=Path(temporary);stats={};checks=[]
        for split in SPLITS:
            subset=[r for r in rows if r['provenance']['split']==split]
            allids=[];allmask=[];offsets=[0];written=[];input_total=0;target_total=0;max_prompt=0;max_answer=0
            for row in subset:
                ids,mask,evidence=validate_record(row,tokenizer,guard)
                item=dict(row);item['validation']={'status':'passed_finite_contract_checks','evidence':evidence}
                item['validation']['record_sha256']=digest(canonical(row))
                written.append(canonical(item))
                allids.extend(ids.tolist());allmask.extend(mask.tolist());offsets.append(len(allids))
                answer=int(mask.sum());prefix=len(ids)-answer;input_total+=prefix;target_total+=answer
                max_prompt=max(max_prompt,prefix);max_answer=max(max_answer,answer)
            (stage/(split+'.jsonl')).write_text('\n'.join(written)+'\n')
            np.save(stage/(split+'.tokens.npy'),np.asarray(allids,np.int32),allow_pickle=False)
            np.save(stage/(split+'.mask.npy'),np.asarray(allmask,np.bool_),allow_pickle=False)
            np.save(stage/(split+'.offsets.npy'),np.asarray(offsets,np.uint64),allow_pickle=False)
            stats[split]={'records':len(subset),'prompt_tokens':input_total,'target_eos_tokens':target_total,
                          'nonpadding_tokens':input_total+target_total,'max_prompt_tokens':max_prompt,'max_target_eos_tokens':max_answer,
                          'sources':dict(Counter(r['provenance']['source'] for r in subset)),
                          'task_families':dict(Counter(r['task_family'] for r in subset)),
                          'features':dict(Counter(f for r in subset for f in r['provenance'].get('features',[])))}
        # Unreviewed candidates are deliberately absent from these model-ready splits.
        accepted={r['id'] for r in rows};excluded=set()
        review=REPO/'reports/model-design/JSON-005_additions-audit-20261010/consolidated_manual_review.json'
        for r in strict_json(review.read_text())['records']:
            if r['decision']!='plausible':excluded.add(r['dataset']+':'+r['id'])
        for r in strict_json((SEEDS/'public_selection_review_v1.json').read_text())['records']:
            if r['decision']=='reject':excluded.add('hermes:'+r['id'])
        queue_counts=Counter()
        rawroot=Path('/tmp/opensml-json-consolidated-quality-20261010')
        with (stage/'review_queue.jsonl').open('w') as f:
            for source in ('hermes','scrapegraph','extract0'):
                path=rawroot/(source+'.candidates.jsonl')
                if not path.exists():raise ValueError('Review-queue source missing: '+str(path))
                for line in path.open():
                    r=strict_json(line);key=source+':'+r['id']
                    if key in accepted or key in excluded:continue
                    f.write(canonical({'status':'pending_semantic_review_not_training_data','source':source,'candidate':r})+'\n')
                    queue_counts[source]+=1
        manifest={'format':'opensml-json-corpus-v1','status':'experimental_starter_corpus_not_release_certified',
                  'training_started':False,'training_launcher_ready':False,'seed':20261010,
                  'splits':stats,'separation':separation,'review_queue':dict(queue_counts),
                  'tokenizer_sha256':TOKENIZER_SHA,'eos_id':1,'serialization':'extract-envelope-output-json-v1',
                  'default_schema_draft':'2020-12 when absent; reject unknown explicit dialects',
                  'validator_formats':sorted(FORMATS.checkers),
                  'data_limitations':['2400 controlled rows represent 1200 documents and 48 task/layout combinations, with two schema projections per document.',
                                      'Shared finite controlled grammar; layout-held-out performance does not establish natural-document generalization.',
                                      '22 public seeds have one semantic reviewer plus schema/evidence checks, not independent human certification.',
                                      'Reserved benchmark exclusions use exact source hashes and 577 Github_easy schema shapes, not full-benchmark or pretraining decontamination.',
                                      'No public source in the pending review queue has been admitted automatically.'],
                  'inputs':{str(p.relative_to(REPO)):file_digest(p) for p in sorted(SEEDS.glob('*')) if p.is_file()},
                  'code':{str(p.relative_to(REPO)):file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                  'files':{p.name:file_digest(p) for p in sorted(stage.iterdir())}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        os.rename(stage,output)
    return manifest

def check(output):
    root=Path(output);manifest=strict_json((root/'manifest.json').read_text());t=load_tokenizer();guard=read_guard();allrows=[]
    for name,sha in manifest['files'].items():
        if file_digest(root/name)!=sha:raise ValueError('Corpus artifact changed: '+name)
    for name,sha in {**manifest['inputs'],**manifest['code']}.items():
        if file_digest(REPO/name)!=sha:raise ValueError('Corpus code/seed changed; build a new version: '+name)
    for split in SPLITS:
        token=np.load(root/(split+'.tokens.npy'),allow_pickle=False);mask=np.load(root/(split+'.mask.npy'),allow_pickle=False)
        offsets=np.load(root/(split+'.offsets.npy'),allow_pickle=False)
        rows=[strict_json(line) for line in (root/(split+'.jsonl')).read_text().splitlines()]
        if len(offsets)!=len(rows)+1 or offsets[0]!=0 or offsets[-1]!=len(token) or len(mask)!=len(token):raise ValueError('Invalid token arrays')
        for index,row in enumerate(rows):
            if row['provenance']['split']!=split:raise ValueError('Wrong split')
            validation=row.pop('validation')
            if digest(canonical(row))!=validation['record_sha256']:raise ValueError('Record was altered')
            ids,expected_mask,evidence=validate_record(row,t,guard)
            a,b=map(int,offsets[index:index+2])
            if not np.array_equal(token[a:b],ids) or not np.array_equal(mask[a:b],expected_mask):raise ValueError('Serialization drift')
            if canonical(evidence)!=canonical(validation['evidence']):raise ValueError('Evidence drift')
            allrows.append(row)
    check_separation(allrows)
    return manifest
