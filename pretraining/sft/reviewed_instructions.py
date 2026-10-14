"""SFT-019: reviewed short instructions, contextual QA and tested Python.

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
from .behavior_v7 import cases,rollouts,python_checks
from .seeds.reviewed_019 import TEXT,PYTHON

PROVENANCE=dict(repo='local/reviewed-instructions-019',revision='reviewed-instructions-v1',split='authored',license='CC0-1.0',
    method='Explicit reviewed seed contracts and deterministic family variants; not independent teacher responses')


def authored_rows():
    rows=[]
    def add(bucket,family,messages,**extra):
        rows.append(dict(bucket=bucket,component='python' if bucket=='python' else 'constraints' if bucket=='format' else 'chat',
                         group='reviewed-019:'+family,messages=messages,provenance=PROVENANCE,**extra))
    for topic,q,answer,short in TEXT:
        add('concise','knowledge-'+topic,pair(q,answer),review_contract='Both sentences checked for direct relevance; no extra unsupported claims.')
        add('concise','knowledge-'+topic,pair(q+' Give a brief answer.',short),review_contract='Complete concise answer to the same question.')
        add('format','knowledge-'+topic,pair(q+' Respond in exactly two sentences.',answer),rules=[dict(kind='count',args=['exactly','2','sentences'])])
        add('followup','knowledge-'+topic,pair(q,answer)+pair('Can you give me a shorter version of that answer?',short),review_contract='Shortening refers to the actual prior explanation.')
    openings=[('Hey, are you there?','Yes, I am here. What would you like to talk about?'),
        ('Can we talk?','Of course. What is on your mind?'),('Good afternoon!','Hello! How can I help you?'),
        ('Just saying hi.','Hi! What would you like to chat about?'),('Is anyone here to help?','I am here to help. What do you need?'),
        ('I have a quick question.','Go ahead. What would you like to ask?'),('Hi, I am new here.','Welcome! How can I help you?'),
        ('Hello, how are things?','I am here and ready to chat. What is on your mind?')]
    needs=[('understand a word','I can explain its meaning and give an example. Which word?'),
        ('write Python','I can help with simple Python code. What should it do?'),('draft a note','Who is the note for, and what should it say?'),
        ('understand an idea','Which idea would you like me to explain?'),('plan a small task','What task do you have in mind?'),
        ('summarize some text','Please share the text you want summarized.'),('check my code','Please paste the code and any error message.'),
        ('brainstorm ideas','What topic would you like ideas about?')]
    for i,(opening,greeting) in enumerate(openings):
        for j,(need,reply) in enumerate(needs):
            add('social',f'social-intent-{j}',pair(opening,greeting)+pair(['Which kinds of everyday tasks can you assist with?','What is the range of help you can offer?','Could you briefly list your useful skills?','What can we work on together in this chat?','Tell me about the tasks you can assist with.','Which questions could I bring to this conversation?','How could you help with writing or programming?','What sorts of help are available here?'][i],
                'I can answer questions, explain ideas, help with writing, and work through simple Python code.')+
                pair('I would like to '+need+'.',reply),review_contract='No biography, assumed tutor subject, or invented user situation.')
    for i,(obj,details) in enumerate([
        ('paragraph','paragraph'),('Python script','code and the full error message'),('recipe','recipe and what you want to change'),
        ('email draft','draft and the intended recipient'),('list','list and the order you want'),('short story','story and the kind of feedback you want'),
        ('instructions','instructions and which part is confusing'),('table','table and the question you want answered'),
        ('plan','plan and the goal'),('question','question'),('sentence','sentence'),('code snippet','code snippet and what you expected'),
        ('summary','original text and your summary'),('message','message and the tone you want'),('outline','outline and its topic'),
        ('schedule','schedule and the tasks it needs to include'),('shopping list','shopping list'),('function','function and the test that fails'),
        ('text','text'),('description','description and what needs clarification')]):
        for form in range(5):
            prompt=[f'Help me improve my {obj}; I have not shared it.',f'Can you check a {obj} for me?',
                f'I need feedback on my {obj}.',f'I want help with a {obj}, but it is not in this message.',
                f'Could you explain my {obj}? I will send it next.'][form]
            answer=f'Please share the {details}, and I can help.'
            add('clarify','missing-'+obj,pair(prompt,answer),review_contract='Ask for actual missing input; do not invent a review.')
            add('clarify','missing-'+obj,pair(openings[form%len(openings)][0],openings[form%len(openings)][1])+pair(prompt,answer),review_contract='Missing-input request after a genuine social exchange.')
    # Different record schemas, update operations and reference types; all
    # variants of a schema/operation share a split, irrespective of values.
    for schema in ('room','shelf','locker','desk','seat','gate','bin','drawer','office','rack','cabinet','box'):
        for v in range(12):
            a=11+v;b=40+v;c=70+v;name='Mila' if v%2 else 'Dara';other='Ben' if v%2 else 'Luca'
            fact=f'{name} uses {schema} {a}; {other} uses {schema} {b}.'
            add('grounded','record-'+schema,pair(f'Use only this record: {fact} Which {schema} does {name} use? Reply with the number alone.',str(a)),rules=[dict(kind='exact',args=[str(a)])])
            add('followup','record-'+schema,pair(f'Remember this record: {fact}',f'{name} uses {schema} {a}; {other} uses {schema} {b}.')+
                pair(f'Update: {name} now uses {schema} {c}.',f'Updated: {name} now uses {schema} {c}.')+
                pair(f'Give just the latest {schema} number for {name}.',str(c)),rules=[dict(kind='exact',args=[str(c)])])
            add('missing','record-'+schema,pair(f'Record: {fact} What is {name}’s phone number? If it is absent, say "Not in the record".', 'Not in the record'),rules=[dict(kind='exact',args=['Not in the record'])])
    for category,items in [('fruit',['apple','pear','plum']),('materials',['wood','glass','steel']),('colors',['red','blue','green']),
        ('tools',['hammer','saw','drill']),('clothes',['coat','shirt','hat']),('plants',['oak','pine','birch']),('pets',['cat','dog','rabbit']),
        ('shapes',['circle','square','triangle']),('places',['park','school','library']),('drinks',['tea','coffee','water']),('seasons',['spring','summer','winter']),
        ('days',['Monday','Tuesday','Friday']),('months',['March','June','July']),('rooms',['kitchen','bedroom','hall']),('foods',['rice','bread','pasta'])]:
        for v in range(8):
            ordered=items[v%3:]+items[:v%3];remain=[ordered[0],ordered[2]];joined=', '.join(remain)
            add('followup','list-family-'+category,pair('Keep this list in order: '+', '.join(ordered)+'.','I have noted the list.')+
                pair('Remove '+ordered[1]+' from it.','The remaining items are '+joined+'.')+
                pair('Give the remaining items separated by a comma and a space, with no extra text.',joined),rules=[dict(kind='exact',args=[joined])])
            add('format','list-family-'+category,pair('List '+', '.join(ordered)+' in that order, using exactly three bullet points.','\n'.join('- '+x for x in ordered)),rules=[dict(kind='count',args=['exactly','3','bullet points'])])
    # Actual supplied text for editing and summarization, rather than a naked
    # greeting that launches unsolicited rewriting.
    for i,(name,obj,place) in enumerate([(n,o,p) for n,o,p in zip(['Iris','Noah','Zoe','Tariq','Rosa','Lina','Omar','Eva','Jules','Ada','Theo','Mina'],
        ['book','bag','note','cup','coat','map','pen','box','key','letter','folder','photo'],['table','shelf','desk','bench','chair','counter','drawer','cabinet','rack','office','hall','room'])]):
        plural=obj+'es' if obj=='box' else obj+'s'
        for v in range(6):
            source=f'{name} brought a {obj}. {name} left the {obj} on the {place}. Then {name} went outside.'
            add('summary','summary-'+obj,pair('Summarize in one sentence: '+source,f'{name} left a {obj} on the {place} before going outside.'),rules=[dict(kind='count',args=['exactly','1','sentences'])])
            add('format','edit-'+obj,pair(f'Fix the subject-verb agreement and return only the corrected sentence: The {plural} is on the {place}.',f'The {plural} are on the {place}.'),rules=[dict(kind='exact',args=[f'The {plural} are on the {place}.'])])
    for family,args,body,meaning,inputs,expected,explanation in PYTHON:
        for v in range(6):
            fn=f'{family}_{v+1}';code=f'def {fn}({args}):\n'+''.join('    '+line+'\n' for line in body.splitlines())
            tests=[dict(function=fn,args_literal=repr(a),expected_literal=repr(e)) for a,e in zip(inputs,expected)]
            # Validate literal gold against all independent tests before using
            # its derived follow-up answers; expected values were supplied above.
            assert all(python_checks('```python\n'+code+'```',tests,function=fn).values()),family
            for mode in ('named','unnamed','followup','docstring'):
                requested=None if mode=='unnamed' else fn
                instruction=(f'Write a Python function to {meaning}.' if requested is None else f'Write {fn}({args}) in Python to {meaning}.')
                doc=mode=='docstring';target=code
                if doc:
                    instruction+=' Include a docstring.'
                    target=code.split('\n',1)[0]+'\n    """'+meaning[0].upper()+meaning[1:]+'."""\n'+code.split('\n',1)[1]
                instruction+=' Return only a Python code block.';answer='```python\n'+target.rstrip()+'\n```'
                messages=pair(instruction,answer)
                if mode=='followup':
                    a=inputs[1];e=expected[1]
                    messages+=pair(f'What is the result of {fn}('+', '.join(repr(x) for x in a)+')? Give only the Python value.',repr(e))
                    messages+=pair('Explain briefly how that function works.',explanation)
                add('python','python-'+family,messages,functional_assertions=tests,python_contract=dict(function=requested,code_only=True,docstring=doc),
                    review_contract='Reviewed prose, explicit domain, four edge/ordinary inputs, correct call result and grounded explanation.')
    # Exact IDs and case controls vary values, but all of each transformation
    # share one family to avoid pretending token changes are independent tasks.
    for family in ('copy-id','lower-label','upper-label'):
        for v in range(40):
            token=f'TAG_{1000+v}' if family=='copy-id' else ['Birch','Cobalt','Daisy','Hazel','Jade','Maple','Pearl','Ruby'][v%8]+str(v)
            prompt=f'Copy only this identifier: {token}' if family=='copy-id' else f'Return only {token} in '+('lowercase.' if family=='lower-label' else 'uppercase.')
            answer=token if family=='copy-id' else token.lower() if family=='lower-label' else token.upper()
            add('format','format-'+family,pair(prompt,answer),rules=[dict(kind='exact',args=[answer])])
    # Remove construction duplicates rather than inventing additional coverage.
    unique={digest(json.dumps(r['messages'],sort_keys=True)):r for r in rows}
    return list(unique.values())


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
    for suite in ('chat-relevance-v2','conversation-v3','conversation-v4','conversation-v5','instruction-v6','instruction-v7'):
        for split in ('dev','test'):
            for case in diagnostic_cases(split,suite):blocked.update(digest(m['content']) for m in case['messages'] if m['role']=='user' and not social_prompt(m['content']))
    for module in (rollouts,r5,r6):
        blocked.update(digest(q) for split in ('dev','test') for case in module(split) for q in case['prompts'])
    # Previous local Python family holdouts must remain excluded semantically,
    # even when a newer corpus changes the spelling of its family IDs.
    heldout_family_aliases={'first':'take_first','divisible':'multiple','last':'take_last','reverse':'reverse_list'}
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
        if sum(title_counts.values())>=plan['grounded_source_cap']*2:break
    # Deterministic stratification over entire reviewed families, so a small
    # random draw cannot leave clarification or social coverage out of a split.
    strata=defaultdict(set)
    for row in candidates:
        if row['provenance'].get('repo')!=PROVENANCE['repo']:continue
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
            split=assignments.get(group,split_group(group,plan['seed'],plan['heldout_group_percent']))
            if bucket=='hf_reviewed':split='train'
            if bucket=='passage':
                cap=plan['grounded_source_cap'] if split=='train' else 120
                if passage_counts[split]>=cap:raise ValueError('passage_cap')
                passage_counts[split]+=1
            messages,ids,mask=encode(messages,tokenizer,plan,row['component'],group)
            rows_by_split[split].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask));seen.add(key)
        except ValueError as e:rejected[bucket+':'+str(e)]+=1
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
            limitations=['Reviewed seed families and value/name variants are correlated, not independent knowledge.','Passage span verification checks evidence alignment, not every Wikipedia fact or unanswerable annotation.','Exact/eight-word screening does not establish complete semantic benchmark decontamination.','New diagnostic scores are not directly comparable to historical v6 scores; no ARC data trains or selects this run.']))
        atomic_json(staging/'diagnostic_protocol.json',dict(suite='instruction-v7',selection='dev only',dev=cases('dev'),test=cases('test'),dev_rollouts=rollouts('dev'),test_rollouts=rollouts('test')))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        manifest=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=counts,files=files,seed=plan['seed'],
            serialization='explicit-ids-assistant-mask-v1',epochs=plan['epochs'],final_quality_version='reviewed-instructions-v1',total_steps=math.ceil(counts['train']['conversations']/64)*plan['epochs'])
        atomic_json(staging/'manifest.json',manifest);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(counts,indent=2),flush=True);return manifest
