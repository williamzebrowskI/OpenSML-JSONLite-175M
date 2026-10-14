"""SFT-021 matched-budget contrast experiment, complete pairs and frozen probes."""
from collections import Counter,defaultdict
import json
import math
import random
import shutil
import uuid
from datasets import Dataset
from sml_v1.common import atomic_json,file_sha256,read_json
from sml_v1.tokenization import Tokenizer
from .curate import local,verify,write_split,blocked_questions
from .conversation import prior_exclusions,split_group
from .reliable_chat import cached,encode
from .quality import digest,normalized,social_prompt,check_rules
from .seeds.reviewed_021 import authored_rows
from .behavior_v8 import cases,rollouts,python_checks
from .bounded_python import run_function
import ast


def select_pairs(rows,quotas,seed):
    selected=[]
    for bucket,quota in quotas.items():
        candidates=[r for r in rows if r['bucket']==bucket]
        units=defaultdict(list)
        bypair=defaultdict(list)
        for r in candidates:
            if r.get('pair_id'):bypair[r['pair_id']].append(r)
            else:units[r['group']].append([r])
        for pair_id,pair in bypair.items():
            if len(pair)!=2 or {r['contrast_side'] for r in pair}!={0,1} or len({r['group'] for r in pair})!=1:
                raise ValueError('Incomplete or mixed contrast pair: '+pair_id)
            units[pair[0]['group']].append(sorted(pair,key=lambda r:r['contrast_side']))
        for group in units:units[group].sort(key=lambda unit:digest(str(seed)+json.dumps(unit,sort_keys=True,default=str)))
        groups=sorted(units,key=lambda g:digest(str(seed)+g));count=0
        while count<quota:
            progress=False
            for group in groups:
                if not units[group]:continue
                unit=units[group].pop()
                if count+len(unit)>quota:raise ValueError('Quota would break a pair')
                selected.extend(unit);count+=len(unit);progress=True
                if count==quota:break
            if not progress:raise ValueError(f'Insufficient {bucket} rows: {count}/{quota}')
    return selected


def validate_target(row):
    target=row['messages'][-1]['content']
    if row.get('rules') and not check_rules(target,row['rules']):raise ValueError('gold_contract')
    if row.get('functional_assertions'):
        if not all(python_checks(target,row['functional_assertions'],**row['python_contract']).values()):raise ValueError('gold_python')
    if row.get('trace_contract'):
        c=row['trace_contract']
        value=run_function(ast.parse(c['code']),c['function'],ast.literal_eval(c['args_literal']))
        from .behavior_v8 import typed_equal
        if not typed_equal(value,ast.literal_eval(c['expected_literal'])) or not typed_equal(ast.literal_eval(target),value):raise ValueError('gold_trace')


def prepare(config):
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
        manifest=verify(dest)
        if manifest['plan_sha256']!=file_sha256(config):raise ValueError('Prepared plan changed')
        return manifest
    tokenizer=Tokenizer(local(plan['tokenizer']));rng=random.Random(plan['seed'])
    blocked,bench=blocked_questions(local('pretraining/data/.sft_hf_cache'))
    oldgroups,oldprompts,oldreceipts=prior_exclusions(include_all=True);blocked.update(oldprompts)
    from .behavior import cases as historical_cases
    for suite in ('chat-relevance-v2','conversation-v3','conversation-v4','conversation-v5','instruction-v6','instruction-v7','instruction-v8'):
        for split in ('dev','test'):
            for case in historical_cases(split,suite):
                blocked.update(digest(m['content']) for m in case['messages'] if m['role']=='user' and not social_prompt(m['content']))
    blocked.update(digest(p) for split in ('dev','test') for case in rollouts(split) for p in case['prompts'])
    # Freeze user-requested fresh diagnostics. Their literal questions and all
    # Python contracts are excluded; they never choose an in-run checkpoint.
    fresh_path=local(plan['frozen_regression_protocol'])
    fresh=read_json(fresh_path)
    blocked.update(digest(p) for case in fresh['cases'] for p in case['prompts'])
    arcgrams=set()
    for cfg in ('ARC-Easy','ARC-Challenge'):
        folder=local('pretraining/data/.sft_hf_cache/allenai___ai2_arc')/cfg/'0.0.0'/'210d026faf9955653af8916fad021475a3f00453'
        for path in folder.glob('*.arrow'):
            for question in Dataset.from_file(str(path))['question']:
                words=normalized(question).split();arcgrams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    candidates=authored_rows();receipts=[]
    src=next(s for s in plan['sources'] if s['component']=='chat');raw,receipt=cached(src);receipts.append(dict(source=src,**receipt))
    for item in read_json(local('pretraining/sft/seeds/hf_reviewed_019.json'))['items']:
        messages=[m for m in raw[item['row']]['messages'] if m['role']!='system']
        if messages[:len(item['messages'])]!=item['messages']:raise ValueError('Reviewed HF prefix changed')
        candidates.append(dict(bucket='hf_reviewed',component='chat',group=item['group'],messages=item['messages'],pair_id=None,provenance=dict(src,row=item['row'],review=item['review'],transformation='Reviewed intact source exchanges; no clipping or rewriting')))
    src=next(s for s in plan['sources'] if s['component']=='qa');raw,receipt=cached(src);receipts.append(dict(source=src,**receipt))
    indices=list(range(len(raw)));rng.shuffle(indices);per_title=Counter()
    for index in indices:
        row=raw[index];group='wikipedia:'+row['title'];answers=row['answers']['text'];context=row['context']
        if group in oldgroups or not 30<=len(context.split())<=180 or len(row['question'].split())>30 or per_title[group]>=12:continue
        if answers:
            target=answers[0];start=row['answers']['answer_start'][0]
            if context[start:start+len(target)]!=target or len(target.split())>12 or not target.strip():continue
        else:target='Not stated in the passage'
        # Source spans stay intact; arbitrary conversion into free-standing
        # world knowledge would silently remove their evidence and is avoided.
        candidates.append(dict(bucket='passage',component='qa',group=group,pair_id=None,
            messages=[dict(role='user',content='Use only this passage. Return the answer alone; if it is missing, reply "Not stated in the passage".\n\nPassage: '+context+'\n\nQuestion: '+row['question']),dict(role='assistant',content=target)],
            answers=answers or [target],answerable=bool(answers),raw_questions=[row['question']],title=row['title'],
            provenance=dict(src,row=index,id=row['id'],source_url='https://en.wikipedia.org/wiki/'+row['title'],transformation='Original passage/question and verified literal gold answer, or source unanswerable annotation')))
        per_title[group]+=1
        if sum(per_title.values())>=4000:break
    pools=defaultdict(list);rejected=Counter();seen=set()
    # Historical semantic exclusions also cover renamed equivalents.
    aliases={'head_slice':'take_first','tail_slice':'take_last','first':'take_first','last':'take_last','divisible':'multiple'}
    for row in candidates:
        try:
            group=row['group'];bucket=row['bucket']
            if group in oldgroups:raise ValueError('historical_reserved_group')
            if bucket=='python' and any(group.endswith('python-'+family) and 'instruction-python-family:'+alias in oldgroups for family,alias in aliases.items()):raise ValueError('historical_python_alias')
            prompts=row.get('raw_questions',[m['content'] for m in row['messages'] if m['role']=='user'])
            if any(digest(p) in blocked for p in prompts if not social_prompt(p)):raise ValueError('reserved_or_benchmark_prompt')
            for prompt in prompts:
                words=normalized(prompt).split()
                if any(' '.join(words[i:i+8]) in arcgrams for i in range(len(words)-7)):raise ValueError('arc_eight_word_overlap')
            validate_target(row)
            key=digest(json.dumps(row['messages'],sort_keys=True))
            if key in seen:raise ValueError('duplicate')
            split='train' if bucket=='hf_reviewed' else plan.get('group_split_overrides',{}).get(group,split_group(group,plan['seed'],plan['heldout_group_percent']))
            messages,ids,mask=encode(row['messages'],tokenizer,plan,row['component'],group)
            pools[split].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask));seen.add(key)
        except ValueError as error:rejected[row['bucket']+':'+str(error)]+=1
    # Reject both sides when screening rejects either; no one-sided contrast
    # enters train or held-out sets. Never duplicate scarce data to fill a quota.
    for split,rows in pools.items():
        pairs=Counter(r['pair_id'] for r in rows if r.get('pair_id'))
        pools[split]=[r for r in rows if not r.get('pair_id') or pairs[r['pair_id']]==2]
    pools['train']=select_pairs(pools['train'],plan['train_bucket_quotas'],plan['seed'])
    if len(pools['train'])!=2735:raise ValueError('Matched 86-update budget changed')
    # Bound holdouts while retaining complete pairs and distinct groups.
    for split in ('dev','test'):
        quotas={bucket:min(cap,sum(r['bucket']==bucket for r in pools[split])) for bucket,cap in plan['heldout_caps'].items()}
        quotas={k:v-(v%2 if k in ('contrast_text','python','dialogue') else 0) for k,v in quotas.items()}
        pools[split]=select_pairs(pools[split],quotas,plan['seed']+1)
    groups=[{r['group'] for r in pools[s]} for s in ('train','dev','test')]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Group split leakage')
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        # Contrast dev/test units are chosen before model evaluation, frozen in
        # the corpus, and kept separate from the previous user-query panel.
        panels={}
        for split in ('dev','test'):
            panels[split]=select_pairs(pools[split],{'contrast_text':16,'python':16,'dialogue':8},plan['seed']+21)
            panels[split]=[{k:v for k,v in row.items() if not k.startswith('_')} for row in panels[split]]
        atomic_json(staging/'contrast_panels.json',panels)
        atomic_json(staging/'frozen_regression_protocol.json',fresh)
        counts={};reviews=[]
        for split in ('train','dev','test'):
            rows=pools[split];rng.shuffle(rows);components=Counter(r['component'] for r in rows)
            if any(components[c]<8 for c in ('chat','constraints','python','qa')):raise ValueError('Insufficient component coverage: '+str((split,components)))
            counts[split]=dict(conversations=len(rows),semantic_groups=len({r['group'] for r in rows}),components=dict(components),buckets=dict(Counter(r['bucket'] for r in rows)),
                input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in components},
                multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows),contrast_pairs=len({r['pair_id'] for r in rows if r.get('pair_id')}),families_by_bucket={b:len({r['group'] for r in rows if r['bucket']==b}) for b in {r['bucket'] for r in rows}})
            reviews.extend({k:v for k,v in r.items() if not k.startswith('_')} for r in rows[:30])
            write_split(staging,split,rows)
        atomic_json(staging/'review_examples.json',reviews)
        atomic_json(staging/'curation.json',dict(counts=counts,rejected=dict(rejected),source_receipts=receipts,previous_holdout_exclusions=oldreceipts,benchmark_exclusions=bench,
            seed_files={p:file_sha256(local(p)) for p in plan['seed_files']},frozen_regression_sha256=file_sha256(fresh_path),fresh_python_semantic_exclusions=['count_positive','clamp','lookup/get_default','lengths','offset_values/add-one','first_positive'],
            limitations=['Finite authored scenes and code families remain correlated; counts are not independent skills.','Literal SQuAD spans check alignment, not all source factual claims or absent-answer labels.','No ARC data trains or selects this run; exact/eight-word screening is not complete semantic decontamination.','Contrast panels test held-out scenario/function groups; historical train families may appear in current holdouts.','Frozen SFT-020 fresh probes are excluded from train and checkpoint ranking; use them after the run for a paired regression review.']))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        manifest=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=counts,files=files,seed=plan['seed'],serialization='explicit-ids-assistant-mask-v1',epochs=plan['epochs'],final_quality_version='contrast-instructions-v1',total_steps=math.ceil(len(pools['train'])/64)*plan['epochs'])
        atomic_json(staging/'manifest.json',manifest);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(counts,indent=2),flush=True);return manifest
