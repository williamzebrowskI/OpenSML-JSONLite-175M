"""SFT-020: balanced short instructions, contextual QA and tested Python.

Authored family variants are finite, not independent data. HF source answers
are either an explicitly reviewed intact whitelist or literal SQuAD spans with
clear passage-only instructions. No blanket trust in length or source tests.
"""
import ast
from collections import Counter,defaultdict
import json
import math
import random
import re
import shutil
import uuid
from datasets import Dataset
from sml_v1.common import atomic_json,file_sha256,read_json
from sml_v1.tokenization import Tokenizer
from .curate import local,verify,write_split,blocked_questions
from .conversation import split_group,prior_exclusions
from .quality import digest,normalized,social_prompt,check_rules
from .reliable_chat import cached,encode
from .balanced import pair
from .behavior_v8 import cases,rollouts,python_checks
from .seeds.reviewed_019 import TEXT,PYTHON

PROVENANCE=dict(repo='local/reviewed-instructions-019',revision='reviewed-instructions-v1',split='authored',license='CC0-1.0',
    method='Explicit reviewed seed contracts and deterministic family variants; not independent teacher responses')


from .seeds.reviewed_020 import authored_rows


def select_balanced(rows, quotas, seed):
    """Round-robin by semantic family and task stratum; no oversampling."""
    chosen=[]
    for bucket, quota in quotas.items():
        pool=[r for r in rows if r['bucket']==bucket]
        strata=defaultdict(list)
        for row in pool:
            strata[(row.get('selection_stratum','default'),row['group'])].append(row)
        for key, group in strata.items():
            group.sort(key=lambda r:digest(str(seed)+json.dumps(r['messages'],sort_keys=True)))
        keys=sorted(strata,key=lambda k:digest(str(seed)+repr(k)))
        accepted=[]
        # Stratify first on query position / intent, then on family.
        bykind=defaultdict(list)
        for key in keys:bykind[key[0].split(':')[0] if bucket=='social' else key[0]].append(key)
        kinds=sorted(bykind)
        offsets=Counter();style_offsets=Counter()
        while len(accepted)<quota:
            progress=False
            for kind in kinds:
                available=[k for k in bykind[kind] if strata[k]]
                if not available:continue
                styles=sorted({k[0] for k in available})
                style=styles[style_offsets[kind]%len(styles)];style_offsets[kind]+=1
                available=[k for k in available if k[0]==style]
                key=available[offsets[(kind,style)]%len(available)];offsets[(kind,style)]+=1
                accepted.append(strata[key].pop());progress=True
                if len(accepted)==quota:break
            if not progress:raise ValueError(f'Quota not available: {bucket} {len(accepted)}/{quota}')
        chosen.extend(accepted)
    if len(chosen)!=2735:raise ValueError('Matched 86-update budget requires 2735 histories')
    return chosen


def prepare(config):
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
        manifest=verify(dest)
        if manifest['plan_sha256']!=file_sha256(config):raise ValueError('Prepared plan changed')
        return manifest
    tokenizer=Tokenizer(local(plan['tokenizer']));rng=random.Random(plan['seed'])
    blocked,bench=blocked_questions(local('pretraining/data/.sft_hf_cache'))
    oldgroups,oldprompts,oldreceipts=prior_exclusions(include_all=True);blocked.update(oldprompts)
    from .behavior import cases as diagnostic_cases
    from .behavior_v5 import rollouts as r5
    from .behavior_v6 import rollouts as r6
    for suite in ('chat-relevance-v2','conversation-v3','conversation-v4','conversation-v5','instruction-v6','instruction-v7','instruction-v8'):
        for split in ('dev','test'):
            for case in diagnostic_cases(split,suite):blocked.update(digest(m['content']) for m in case['messages'] if m['role']=='user' and not social_prompt(m['content']))
    for module in (rollouts,r5,r6):
        blocked.update(digest(q) for split in ('dev','test') for case in module(split) for q in case['prompts'])
    # Previous local Python family holdouts must remain excluded semantically,
    # even when a newer corpus changes the spelling of its family IDs.
    heldout_family_aliases={'first':'take_first','divisible':'multiple','last':'take_last','reverse':'reverse_list','head_slice':'take_first','tail_slice':'take_last'}
    excluded_python={family for family,alias in heldout_family_aliases.items() if 'instruction-python-family:'+alias in oldgroups}
    arcgrams=set()
    for cfg in ('ARC-Easy','ARC-Challenge'):
        folder=local('pretraining/data/.sft_hf_cache/allenai___ai2_arc')/cfg/'0.0.0'/'210d026faf9955653af8916fad021475a3f00453'
        for p in folder.glob('*.arrow'):
            for q in Dataset.from_file(str(p))['question']:
                w=normalized(q).split();arcgrams.update(' '.join(w[i:i+8]) for i in range(len(w)-7))
    candidates=authored_rows();source_receipts=[]
    text=next(s for s in plan['sources'] if s['component']=='chat');raw,receipt=cached(text);source_receipts.append(dict(source=text,**receipt))
    whitelist=read_json(local('pretraining/sft/seeds/hf_reviewed_019.json'))
    for item in whitelist['items']:
        source_messages=raw[item['row']]['messages'];body=[m for m in source_messages if m['role']!='system']
        if body[:len(item['messages'])]!=item['messages']:raise ValueError('Reviewed HF prefix changed')
        candidates.append(dict(bucket='hf_reviewed',component='chat',group=item['group'],messages=item['messages'],provenance=dict(text,row=item['row'],review=item['review'],transformation='Reviewed intact initial exchanges; no answer rewriting')))
    qa=next(s for s in plan['sources'] if s['component']=='qa');raw,receipt=cached(qa);source_receipts.append(dict(source=qa,**receipt))
    # Seeded title-grouped sampling. Literal answer evidence is checked for all
    # selected spans; it is not a universal factual validity certificate.
    indices=list(range(len(raw)));rng.shuffle(indices)
    title_counts=Counter()
    for index in indices:
        r=raw[index];context=r['context'];question=r['question'];answers=r['answers']['text']
        if 'wikipedia:'+r['title'] in oldgroups:continue
        if not 30<=len(context.split())<=180 or len(question.split())>30 or title_counts[r['title']]>=14:continue
        if answers:
            answer=answers[0];start=r['answers']['answer_start'][0]
            if context[start:start+len(answer)]!=answer or len(answer.split())>12 or not answer.strip():continue
            target=answer
        else:target='Not stated in the passage'
        candidates.append(dict(bucket='passage',component='qa',group='wikipedia:'+r['title'],
            messages=pair('Use only this passage. Return the answer alone; if it is missing, reply "Not stated in the passage".\n\nPassage: '+context+'\n\nQuestion: '+question,target),
            answers=answers if answers else ['Not stated in the passage'],answerable=bool(answers),title=r['title'],raw_questions=[question],
            provenance=dict(qa,row=index,id=r['id'],title=r['title'],source_url='https://en.wikipedia.org/wiki/'+r['title'],transformation='Unmodified passage and question; verified literal gold span; explicit absent-answer convention')))
        title_counts[r['title']]+=1
        if sum(title_counts.values())>=plan['grounded_source_cap']*4:break
    # Deterministic stratification over entire reviewed families, so a small
    # random draw cannot leave clarification or social coverage out of a split.
    strata=defaultdict(set)
    for row in candidates:
        if not row['provenance'].get('repo','').startswith('local/') or row['group'] in oldgroups:continue
        group=row['group'];family=group.split(':',1)[1]
        if family.startswith('python-') and family[7:] in excluded_python:continue
        strata[family.split('-',1)[0]].add(group)
    assignments={}
    for stratum,groups in strata.items():
        groups=sorted(groups,key=lambda g:digest(str(plan['seed'])+g))
        # Three exact-format families all belong to training; independent
        # diagnostic values cover those operations, not relabelled train rows.
        n=0 if len(groups)<6 else max(4 if stratum=='python' else 2,round(len(groups)*plan['heldout_group_percent']/100))
        for i,group in enumerate(groups):assignments[group]='dev' if i<n else 'test' if i<2*n else 'train'
    rows_by_split=defaultdict(list);rejected=Counter();seen=set();passage_counts=Counter()
    for row in candidates:
        bucket=row['bucket'];group=row['group'];messages=row['messages']
        try:
            if group in oldgroups:raise ValueError('previous_holdout_group')
            if bucket=='python' and group.rsplit('-',1)[-1] in excluded_python:raise ValueError('previous_python_family_holdout')
            users=row.get('raw_questions',[m['content'] for m in messages if m['role']=='user'])
            if any(digest(u) in blocked for u in users if not social_prompt(u)):raise ValueError('benchmark_or_diagnostic_prompt')
            for u in users:
                w=normalized(u).split()
                if any(' '.join(w[i:i+8]) in arcgrams for i in range(len(w)-7)):raise ValueError('arc_eight_word_overlap')
            if bucket!='python' and any(len(m['content'].split())>45 for m in messages if m['role']=='assistant'):raise ValueError('long_text_target')
            if row.get('rules') and not check_rules(messages[-1]['content'],row['rules']):raise ValueError('gold_format_contract')
            if bucket=='python':
                code_answer=next(m['content'] for m in messages if m['role']=='assistant')
                if not all(python_checks(code_answer,row['functional_assertions'],**row['python_contract']).values()):raise ValueError('gold_python_contract')
            key=digest(json.dumps(messages,sort_keys=True))
            if key in seen:raise ValueError('duplicate')
            split=assignments.get(group,split_group(group,plan['seed']+20,plan['heldout_group_percent']))
            if bucket=='hf_reviewed':split='train'
            if bucket=='passage':
                cap=plan['grounded_source_cap'] if split=='train' else 120
                if passage_counts[split]>=cap:raise ValueError('passage_cap')
                passage_counts[split]+=1
            messages,ids,mask=encode(messages,tokenizer,plan,row['component'],group)
            rows_by_split[split].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask));seen.add(key)
        except ValueError as e:rejected[bucket+':'+str(e)]+=1
    rows_by_split['train']=select_balanced(rows_by_split['train'],plan['train_bucket_quotas'],plan['seed'])
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        counts={};review=[]
        required={'concise','clarify','followup','grounded','missing','format','python','social','passage'}
        for split,rows in rows_by_split.items():
            if required-{r['bucket'] for r in rows}:raise ValueError('Realized coverage missing: '+str((split,required-{r['bucket'] for r in rows})))
        for split in ('train','dev','test'):
            rows=rows_by_split[split];rng.shuffle(rows);components=Counter(r['component'] for r in rows)
            if any(components[c]<8 for c in ('chat','constraints','python','qa')):raise ValueError('Insufficient held-out coverage: '+str((split,components)))
            counts[split]=dict(conversations=len(rows),semantic_groups=len({r['group'] for r in rows}),components=dict(components),buckets=dict(Counter(r['bucket'] for r in rows)),
                input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),
                assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in components},multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows),
                families_by_bucket={b:len({r['group'] for r in rows if r['bucket']==b}) for b in {r['bucket'] for r in rows}})
            for bucket in counts[split]['buckets']:review.extend({k:v for k,v in r.items() if not k.startswith('_')} for r in [r for r in rows if r['bucket']==bucket][:8])
            write_split(staging,split,rows)
        groups=[{r['group'] for r in rows_by_split[s]} for s in ('train','dev','test')]
        if any(groups[i]&groups[j] for i in range(3) for j in range(i)):raise ValueError('Family split leakage')
        atomic_json(staging/'review_examples.json',review)
        atomic_json(staging/'curation.json',dict(counts=counts,rejected=dict(rejected),source_receipts=source_receipts,previous_holdout_exclusions=oldreceipts,benchmark_exclusions=bench,
            seed_files={p:file_sha256(local(p)) for p in plan['seed_files']},previous_python_family_exclusions=sorted(excluded_python),
            limitations=['Reviewed seed families and value/name variants are correlated, not independent knowledge.','Passage span verification checks evidence alignment, not every Wikipedia fact or unanswerable annotation.','Exact/eight-word screening does not establish complete semantic benchmark decontamination.','Instruction-v8 changes social heuristics, so compare against saved SFT-019 responses rescored with v8; historical v7 results remain unchanged. No ARC examples train or select this run.']))
        atomic_json(staging/'diagnostic_protocol.json',dict(suite='instruction-v8',selection='dev only',dev=cases('dev'),test=cases('test'),dev_rollouts=rollouts('dev'),test_rollouts=rollouts('test')))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        manifest=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=counts,files=files,seed=plan['seed'],
            serialization='explicit-ids-assistant-mask-v1',epochs=plan['epochs'],final_quality_version='balanced-instructions-v2',total_steps=math.ceil(counts['train']['conversations']/64)*plan['epochs'])
        atomic_json(staging/'manifest.json',manifest);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(counts,indent=2),flush=True);return manifest
