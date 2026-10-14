"""Curate supplied-schema extraction from pinned annotations and reviewed seeds.

Annotation replay establishes conversion fidelity, not perfect semantic golds.
No model, optimizer, checkpoint or training launcher is imported here.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import random
import re
import shutil
import tempfile
import zipfile
import numpy as np
import pyarrow.parquet as pq
from jsonschema import FormatChecker, validators, ValidationError
from json_specialist.data import canonical, strict_json, serialize
from json_specialist.corpus import load_tokenizer, read_guard, schema_shape, norm
from json_specialist.verify import verify_controlled

ROOT = Path(__file__).resolve().parents[2]
RAW_DEFAULT = ROOT/'pretraining/data/json_001_sources_20261010'
SEED = 20261010
FEW_REV = '205f3e9c9f3577ea2561d43f2f62dc249ab92d5b'
SGD_REV = 'e852981ae34990f4358979625854259302feaa78'
CAPS = {'train': {'fewnerd': 0, 'sgd': 3000},
        'development': {'fewnerd': 0, 'sgd': 150}, 'test': {'fewnerd': 0, 'sgd': 150}}
FORMAT = 'opensml-json-sft-corpus-v2'
CHECKER = FormatChecker()

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rank(value):
    return sha(str(SEED) + ':' + value)

def nodes(x):
    if isinstance(x, dict):
        yield x
        for v in x.values(): yield from nodes(v)
    elif isinstance(x, list):
        for v in x: yield from nodes(v)

def fewnerd_record(row, index, labels, split):
    tokens, tags = row['tokens'], row['fine_ner_tags']
    if len(tokens) != len(tags) or len(tags) != len(row['ner_tags']):
        raise ValueError('annotation_length')
    # Schema choices depend on the input ID/seed, never on target values.
    available = ['location', 'organization', 'person', 'product']
    requested = sorted(random.Random(rank('few:' + split + ':' + str(index))).sample(available, 3))
    document = ' '.join(tokens)
    if not 40 <= len(document) <= 1800: raise ValueError('source_length')
    entities, evidence = [], []
    offsets = np.cumsum([0] + [len(t)+1 for t in tokens]).tolist()
    start = 0
    while start < len(tokens):
        tag = tags[start]
        if tag not in labels: raise ValueError('unknown_label')
        end = start+1
        while end < len(tokens) and tags[end] == tag: end += 1
        coarse = labels[tag].split('-')[0]
        if coarse in requested:
            text = ' '.join(tokens[start:end])
            # Known adjective/location and markup ambiguity: exclude whole row.
            if coarse == 'location' and text.casefold() in {
                'french','english','german','spanish','italian','american','british','japanese','chinese','dutch'}:
                raise ValueError('ambiguous_demonym_location')
            if re.search(r'&\s*amp\s*;|<|>', text): raise ValueError('markup_entity')
            entities.append({'text': text, 'type': coarse})
            evidence.append({'start': int(offsets[start]), 'end': int(offsets[end-1]+len(tokens[end-1])),
                             'text': text, 'type': coarse, 'fine_tag': tag, 'token_start': start, 'token_end': end})
        start = end
    if not entities or len(entities) > 12: raise ValueError('no_or_too_many_entities')
    schema = {'type':'object','properties':{'entities':{'type':'array','items':{
        'type':'object','properties':{'text':{'type':'string','description':'Exact source mention, preserving supplied spacing.'},
        'type':{'type':'string','enum':requested}},'required':['text','type'],'additionalProperties':False}}},
        'required':['entities'],'additionalProperties':False}
    return {'id':f'fewnerd:{split}:{row["id"]}:{index}', 'document':document, 'schema':schema,
        'instruction':'Extract named entity mentions of the requested types in source order. Copy each complete mention exactly; repeat mentions when repeated. Return only types listed in the schema. Do not infer unstated names.',
        'target':{'entities':entities},'source':'fewnerd','source_group':sha(norm(document)), 'split':split,
        'provenance':{'dataset':'DFKI-SLT/few-nerd','revision':FEW_REV,'original_split':{'development':'validation'}.get(split,split),
            'original_id':row['id'],'row_index':index,'license':'CC-BY-SA-4.0','attribution':'Few-NERD dataset authors',
            'conversion':'Published fine IO-tag maximal runs; coarse requested types selected from input ID before answers; token-joined source spacing.',
            'annotation_quality':'Human-authored published labels, mechanically replayed; not individually re-certified.'},
        'evidence':evidence, 'oracle':{'tokens':tokens,'fine_tags':tags,'labels':{str(k):v for k,v in labels.items()},'row_id':row['id'],'row_index':index}}

def sgd_record(dialogue, turn_index, frame, service, split, member):
    if turn_index != 0: raise ValueError('not_self_contained_first_request')
    if service['service_name'].split('_')[0] in {'Flights','Buses','RentalCars'}:
        raise ValueError('deferred_ambiguous_travel_service')
    utterance = dialogue['turns'][turn_index]['utterance']
    if re.search(r'\b(at least|at most|or higher|or lower|minimum|maximum)\b',utterance,re.I):
        raise ValueError('deferred_comparison_operator')
    if not 35 <= len(utterance) <= 1000: raise ValueError('source_length')
    if re.search(r'\b(same as|as before|previous|earlier|that one|those ones|instead|change it|change that)\b', utterance, re.I):
        raise ValueError('possible_missing_context')
    possible = sorted(s['name'] for s in service['slots'] if not s['is_categorical'])
    if not possible: raise ValueError('no_span_slots')
    key = dialogue['dialogue_id'] + ':' + str(turn_index) + ':' + service['service_name']
    rng = random.Random(rank('sgd:' + key))
    requested = sorted(rng.sample(possible, min(6, len(possible))))
    properties = {}
    for slot in service['slots']:
        if slot['name'] in requested:
            properties[slot['name']] = {'type':'array','description':slot['description']+' Only explicitly mentioned values in this utterance; copy verbatim, no normalization. Use [] when absent.', 'items':{'type':'string'}}
    schema = {'type':'object','properties':properties,'required':requested,'additionalProperties':False}
    target = {k:[] for k in requested}
    evidence = []
    for span in sorted(frame.get('slots', []), key=lambda s:s.get('start', -1)):
        if span['slot'] not in requested: continue
        a,b = span.get('start'),span.get('exclusive_end')
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(utterance):
            raise ValueError('invalid_span')
        value = utterance[a:b]
        if not value.strip(): raise ValueError('empty_span')
        target[span['slot']].append(value)
        evidence.append({'path':span['slot'],'start':a,'end':b,'text':value})
    if not evidence: raise ValueError('no_selected_explicit_span')
    return {'id':'sgd:'+split+':'+key, 'document':utterance, 'schema':schema,
        'instruction':'Extract only explicitly mentioned '+service['description']+' slot values from this single utterance. Copy exact source wording in mention order. An absent field is an empty array. Do not infer prior dialogue, normalize dates, or supply requested-but-unstated values.',
        'target':target,'source':'sgd','source_group':'sgd:'+split+':'+dialogue['dialogue_id'],'split':split,
        'provenance':{'dataset':'google-research-datasets/schema_guided_dstc8','upstream_revision':SGD_REV,
            'original_split':{'development':'dev'}.get(split,split),'original_member':member,'dialogue_id':dialogue['dialogue_id'],
            'turn_index':turn_index,'service':service['service_name'],'license':'CC-BY-SA-4.0','attribution':'Google Schema-Guided Dialogue dataset authors',
            'conversion':'First USER utterance spans only; flights/buses/rental-car services deferred; no cumulative state, canonical aliases, future turns or system/service results.',
            'annotation_quality':'Published simulator/span annotations over crowdworker paraphrases, mechanically replayed; not individually re-certified.'},
        'evidence':evidence,'oracle':{'frame_spans':frame.get('slots',[]),'service':service,'dialogue_id':dialogue['dialogue_id'],'turn_index':turn_index}}

def replay(row):
    source = row['source']
    if source == 'fewnerd':
        o = row['oracle']
        original = {'tokens':o['tokens'],'fine_ner_tags':o['fine_tags'],'ner_tags':o['fine_tags'],'id':o['row_id']}
        expected = fewnerd_record(original,o['row_index'],{int(k):v for k,v in o['labels'].items()},row['split'])
        if canonical([row['document'],row['schema'],row['instruction'],row['target'],row['evidence']]) != canonical([expected[k] for k in ['document','schema','instruction','target','evidence']]):
            raise ValueError('Few-NERD conversion mismatch')
    elif source == 'sgd':
        o = row['oracle']; turns=[{} for _ in range(o['turn_index'])]+[{'utterance':row['document']}]
        d={'dialogue_id':o['dialogue_id'],'turns':turns}
        expected=sgd_record(d,o['turn_index'],{'slots':o['frame_spans']},o['service'],row['split'],row['provenance']['original_member'])
        if canonical([row[k] for k in ['schema','instruction','target','evidence']]) != canonical([expected[k] for k in ['schema','instruction','target','evidence']]):
            raise ValueError('SGD conversion mismatch')
    elif source == 'controlled':
        verify_controlled(row['original_record'])
        if canonical([row[k] for k in ['document','schema','instruction','target']]) != canonical([row['original_record'][k] for k in ['document','schema','instruction','target']]):
            raise ValueError('Controlled adaptation mismatch')
    elif source == 'reviewed_public':
        original = row['original_record']
        if canonical([row[k] for k in ['document','schema','instruction','target']]) != canonical([original[k] for k in ['document','schema','instruction','target']]):
            raise ValueError('Reviewed public adaptation mismatch')
        for ev in original['review']['evidence']:
            if row['document'][ev['start']:ev['end']] != ev['value'] or row['target'][ev['path'][2:]] != ev['value']:
                raise ValueError('Reviewed public evidence mismatch')
    else: raise ValueError('Unknown source')

def validate(row, tokenizer, guard):
    if sha(norm(row['document'])) in guard['sources']: raise ValueError('reserved_source')
    if row['split']=='train' and sha(norm(row['document'])) in guard.get('external_sources',set()):raise ValueError('published_eval_source')
    if sha(canonical(schema_shape(row['schema']))) in guard['schemas']: raise ValueError('reserved_schema')
    schema=row['schema']
    if schema.get('type')!='object' or not schema.get('properties'):raise ValueError('empty_schema')
    for node in nodes(schema):
        if len(node.get('properties',{}))>8: raise ValueError('schema_width')
        if '$ref' in node: raise ValueError('reference_not_supported')
        if node.get('format') and node['format'] not in CHECKER.checkers:raise ValueError('unknown_format')
    v=validators.validator_for(schema,default=None) if '$schema' in schema else validators.Draft202012Validator
    if v is None:raise ValueError('unknown_draft')
    v.check_schema(schema);v(schema,format_checker=CHECKER).validate(row['target'])
    replay(row)
    return serialize(tokenizer,row)

def load_candidates(raw):
    counters=defaultdict(Counter); candidates=defaultdict(list); inputs={};reserved=set()
    pins=strict_json((ROOT/'reports/model-design/JSON-008_sft-corpus-20261010/source-files.json').read_text())
    for name,digest in pins['sha256'].items():
        if Path(name).name!=name or file_sha(raw/name)!=digest:raise ValueError('Pinned source bytes differ: '+name)
    inputs.update(pins['sha256'])
    meta=strict_json((raw/'fewnerd.meta.json').read_text())
    if meta['sha']!=FEW_REV:raise ValueError('Few-NERD revision changed')
    info=next(x for x in meta['cardData']['dataset_info'] if x['config_name']=='supervised')
    labels={int(k):v for k,v in next(x for x in info['features'] if x['name']=='fine_ner_tags')['sequence']['class_label']['names'].items()}
    for split,original in [('train','train'),('development','validation'),('test','test')]:
        path=raw/f'fewnerd-{original}.parquet';inputs[path.name]=file_sha(path)
        rows=pq.read_table(path).to_pylist()
        for index,row in enumerate(rows):
            if split!='train':reserved.add(sha(norm(' '.join(row['tokens']))))
            counters['fewnerd:'+split]['input_rows']+=1
            if CAPS[split]['fewnerd']==0:
                counters['fewnerd:'+split]['deferred_semantic_quality']+=1
                continue
            try:candidates[split].append(fewnerd_record(row,index,labels,split))
            except ValueError as e:counters['fewnerd:'+split][str(e)]+=1
    archive=raw/'sgd-original.zip';inputs[archive.name]=file_sha(archive)
    with zipfile.ZipFile(archive) as z:
        base=z.namelist()[0].split('/')[0]
        if not base.endswith(SGD_REV):raise ValueError('SGD archive revision changed')
        for split,original in [('train','train'),('development','dev'),('test','test')]:
            services={s['service_name']:s for s in strict_json(z.read(base+'/'+original+'/schema.json').decode())}
            files=sorted(n for n in z.namelist() if n.startswith(base+'/'+original+'/dialogues_') and n.endswith('.json'))
            for member in files:
                for d in strict_json(z.read(member).decode()):
                    counters['sgd:'+split]['input_dialogues']+=1
                    options=[]
                    for ti,turn in enumerate(d['turns']):
                        if turn['speaker']!='USER':continue
                        if split!='train':reserved.add(sha(norm(turn['utterance'])))
                        for frame in turn['frames']:
                            try:options.append(sgd_record(d,ti,frame,services[frame['service']],split,member))
                            except ValueError as e:counters['sgd:'+split][str(e)]+=1
                    # One utterance per dialogue prevents prefixes manufacturing scale.
                    if options:candidates[split].append(min(options,key=lambda r:rank(r['id'])))
    return candidates,counters,inputs,reserved

def near_cross_split(rows, drop=False):
    # Greedy holdout-priority matching; no golds enter similarity calculation.
    inverted=defaultdict(list);sets={};seen={};removed=[];kept=[]
    for row in sorted(rows,key=lambda r:({'test':0,'development':1,'train':2}[r['split']],rank(r['id']))):
        text=norm(row['document']);key=sha(text)
        words=re.findall(r'\w+',text)
        shingles=set(tuple(words[i:i+5]) for i in range(max(1,len(words)-4)))
        conflicts=[]
        if key in seen and seen[key]!=row['split']:conflicts.append(key)
        for other,intersection in Counter(k for s in shingles for k in inverted[s] if seen[k]!=row['split']).items():
            union=len(shingles)+len(sets[other])-intersection
            if union and intersection/union>=.85:conflicts.append(other)
        if conflicts:
            if not drop:raise ValueError('Cross-split source similarity')
            removed.append({'id':row['id'],'split':row['split'],'reason':'source_exact_or_word5_jaccard_ge_0.85'})
            continue
        if key not in seen:
            seen[key]=row['split'];sets[key]=shingles
            for s in shingles:inverted[s].append(key)
        kept.append(row)
    groups=defaultdict(set)
    for row in kept:groups[row['source_group']].add(row['split'])
    if any(len(s)>1 for s in groups.values()):raise ValueError('Cross-split source group')
    return kept,removed

def prepare(output, raw=RAW_DEFAULT, exclusions=None, *, reviewed_only=True):
    output=Path(output).resolve();raw=Path(raw)
    if output.exists():raise ValueError('Output exists; use --check or a new version')
    excluded=set(strict_json(Path(exclusions).read_text()).get('excluded_ids',[])) if exclusions else set()
    review_path=ROOT/'reports/model-design/JSON-008_sft-corpus-20261010/semantic-review.json'
    reviewed_ids={r['id'] for r in strict_json(review_path.read_text())['narrowed_review'] if r['decision']=='sampled_no_concrete_target_error_found'}
    tokenizer=load_tokenizer();guard=read_guard()
    candidates,counters,inputs,reserved=load_candidates(raw)
    inputs['semantic_review']=file_sha(review_path)
    if exclusions:inputs['semantic_review_exclusions']=file_sha(exclusions)
    guard['external_sources']=reserved
    rows=[];rejects=[];seen=set()
    # Published evaluation sources have priority over train, even for exact copies.
    for split in ('test','development','train'):
        counts=Counter()
        for row in sorted(candidates[split],key=lambda r:rank(r['id'])):
            if counts[row['source']]>=CAPS[split][row['source']]:continue
            if reviewed_only and row['source']=='sgd' and row['id'] not in reviewed_ids:continue
            try:
                if row['id'] in excluded:raise ValueError('semantic_review_exclusion')
                source=sha(norm(row['document']))
                if source in seen:raise ValueError('duplicate_source')
                validate(row,tokenizer,guard)
                seen.add(source);rows.append(row);counts[row['source']]+=1
            except (ValueError,ValidationError) as e:
                counters[row['source']+':'+split]['candidate_rejected_'+type(e).__name__+':'+str(e).split('\n')[0][:100]]+=1
    del candidates
    starter=ROOT/'pretraining/data/json_001_starter_v1'
    # Reuse finite source-parser-verified controlled examples; preserve layout split.
    for split in ('train','development','test'):
        path=starter/(split+'.jsonl');inputs['starter/'+path.name]=file_sha(path)
        for line in path.open():
            old=strict_json(line);old.pop('validation',None)
            if old['provenance']['kind']!='procedural_synthetic':continue
            rows.append({'id':old['id'],'document':old['document'],'schema':old['schema'],'instruction':old['instruction'],
                'target':old['target'],'source':'controlled','source_group':old['provenance']['source_family'],'split':split,
                'provenance':old['provenance'],'original_record':old})
    pool=ROOT/'reports/model-design/JSON-006_public-curation-20261010/accepted_public_pool.jsonl';inputs['reviewed_public_pool']=file_sha(pool)
    for line in pool.open():
        old=strict_json(line);family=old['provenance']['source_family']
        bucket=int(rank(family)[:8],16)%100
        split='test' if bucket<5 else 'development' if bucket<10 else 'train'
        rows.append({'id':old['id'],'document':old['document'],'schema':old['schema'],'instruction':old['instruction'],
            'target':old['target'],'source':'reviewed_public','source_group':family,'split':split,
            'provenance':old['provenance'],'original_record':old})
    rows,removed=near_cross_split(rows,drop=True);rejects.extend(removed)
    ids=[r['id'] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate ID')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.json-sft-',dir=output.parent) as td:
        stage=Path(td);stats={}
        for split in ('train','development','test'):
            subset=sorted((r for r in rows if r['split']==split),key=lambda r:rank(r['id']))
            arrays=[];masks=[];offsets=[0];prefixes=answers=0;lengths=[]
            with (stage/(split+'.jsonl')).open('w') as f:
                for row in subset:
                    a,m=validate(row,tokenizer,guard);arrays.append(a);masks.append(m);offsets.append(offsets[-1]+len(a))
                    prefixes+=int((~m).sum());answers+=int(m.sum());lengths.append(len(a))
                    f.write(canonical(row)+'\n')
            np.save(stage/(split+'.tokens.npy'),np.concatenate(arrays).astype(np.int32),allow_pickle=False)
            np.save(stage/(split+'.mask.npy'),np.concatenate(masks),allow_pickle=False)
            np.save(stage/(split+'.offsets.npy'),np.asarray(offsets,np.uint64),allow_pickle=False)
            stats[split]={'records':len(subset),'sources':dict(Counter(r['source'] for r in subset)),
                          'prompt_tokens':prefixes,'target_eos_tokens':answers,'total_tokens':prefixes+answers,'max_total_tokens':max(lengths)}
        fixtures=ROOT/'reports/model-design/JSON-001-evaluation-fixtures-20261009'
        shutil.copyfile(fixtures/'extraction-smoke-10.json',stage/'extraction-smoke-10.json')
        shutil.copyfile(fixtures/'jsonschemabench-structural-smoke-10.json',stage/'jsonschemabench-structural-smoke-10.json')
        (stage/'reserved_public_eval_sources.json').write_text(json.dumps(sorted(reserved))+'\n')
        (stage/'rejections.json').write_text(json.dumps({'counters':counters,'source_similarity_exclusions':rejects,'semantic_excluded_ids':sorted(excluded)},indent=2)+'\n')
        manifest={'format':FORMAT,'status':'experimental_reviewed_core_not_release_certified' if reviewed_only else 'candidate_annotation_expansion_not_individually_reviewed',
            'tier':'reviewed_core' if reviewed_only else 'annotation_candidates','parent':None,'training_started':False,
            'seed':SEED,'serialization':'extract-envelope-output-json-v1','eos_id':1,'splits':stats,'inputs':inputs,
            'source_revisions':{'fewnerd':FEW_REV,'sgd_upstream':SGD_REV},'tokenizer_sha256':file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'),
            'scope':['Few-NERD excluded after semantic review; SGD adapted from published ontology, not native JSON Schema triples.',
                     'Public annotation labels are supervision, not independent semantic certification.',
                     'One selected source sentence/utterance per source group; no dialogue-state aliases or future context.',
                     'Source exact and cross-split word-five-shingle Jaccard >=0.85 excluded among selected records.',
                     'No Wikipedia article IDs available: sentence separation does not establish article or pretraining decontamination.',
                     'Shared ontology/schema families across published splits; external schema smoke panel remains separate.',
                     'Original reviewed public seeds remain individually reviewed; rejected synthetic/benchmark pools excluded.'],
            'code':{str(p.relative_to(ROOT)):file_sha(p) for p in [Path(__file__),*sorted((ROOT/'pretraining/json_specialist').glob('*.py'))]},
            'files':{p.name:file_sha(p) for p in sorted(stage.iterdir())}}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        stage.rename(output)
    return manifest

def check(output, full=True):
    root=Path(output);m=strict_json((root/'manifest.json').read_text())
    if m['format']!=FORMAT:raise ValueError('Unknown corpus format')
    for name,digest in m['files'].items():
        if Path(name).name!=name or file_sha(root/name)!=digest:raise ValueError('Changed corpus file: '+name)
    if m['tokenizer_sha256']!=file_sha(ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'):raise ValueError('Tokenizer changed')
    for name,digest in m['code'].items():
        if not (ROOT/name).resolve().is_relative_to(ROOT/'pretraining') or file_sha(ROOT/name)!=digest:raise ValueError('Converter code changed: '+name)
    if full:
        tokenizer=load_tokenizer();guard=read_guard();guard['external_sources']=set(strict_json((root/'reserved_public_eval_sources.json').read_text()));rows=[]
        for split in ('train','development','test'):
            tokens=np.load(root/(split+'.tokens.npy'),mmap_mode='r');masks=np.load(root/(split+'.mask.npy'),mmap_mode='r');offsets=np.load(root/(split+'.offsets.npy'),mmap_mode='r')
            items=[strict_json(line) for line in (root/(split+'.jsonl')).open()]
            if len(items)!=m['splits'][split]['records'] or len(offsets)!=len(items)+1 or offsets[0]!=0 or offsets[-1]!=len(tokens) or len(tokens)!=len(masks):raise ValueError('Bad corpus arrays')
            for i,row in enumerate(items):
                if row['split']!=split:raise ValueError('Wrong split')
                a,b=map(int,offsets[i:i+2]);expected,mask=validate(row,tokenizer,guard)
                if not np.array_equal(tokens[a:b],expected) or not np.array_equal(masks[a:b],mask):raise ValueError('Output-only serialization drift')
                rows.append(row)
            if sum(len(row['document']) for row in items)==0:raise ValueError('Empty source split')
        near_cross_split(rows)
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate record IDs')
    return m
