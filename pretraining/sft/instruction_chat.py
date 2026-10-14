"""SFT-018: complete text instruction exchanges and checked beginner Python.

No science-choice corpus or plain-QA serialization. Each selected source history
is an intact initial set of exchanges, never a cut assistant answer. Realized
counts are reported; scarce Python data never shrinks text or creates copies.
"""
from collections import Counter,defaultdict
import ast
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
from .quality import digest,normalized,repeated,social_prompt,constraint_rules,check_rules
from .conversation import prior_exclusions,split_group,curriculum
from .short_chat import focused_curriculum
from .reliable_chat import cached,encode,python_row,correction_curriculum
from .bounded_python import verify_target,run_function
from .behavior_v6 import cases,rollouts
from .balanced import pair


def text_prefix(messages,tokenizer,plan):
    """Keep complete initial exchanges; reject specialized supplied systems."""
    messages=[dict(m) for m in messages];prefix=[];offset=0
    if messages and messages[0]['role']=='system':
        system=messages[0]['content']
        if len(system.split())>45 or re.search(r'\b(?:rewrit|teacher|tutor|persona|detailed|long answer|step.by.step|expert|role|translate)',system,re.I):raise ValueError('specialized_system')
        prefix.append(messages[0]);offset=1
    for i in range(offset,min(len(messages)-1,offset+2*plan['max_dialogue_turns']),2):
        user,answer=messages[i:i+2]
        if user['role']!='user' or answer['role']!='assistant':raise ValueError('nonalternating')
        if len(user['content'].split())>plan['user_max_words'] or len(answer['content'].split())>plan['text_reply_max_words'] or len(tokenizer.encode(answer['content']))>plan['assistant_max_tokens']:break
        prefix.extend((user,answer))
    if not any(m['role']=='assistant' for m in prefix):raise ValueError('no_complete_short_exchange')
    return prefix


def python_curriculum():
    """Finite authored functions with offline checks and actual-code follow-ups.

    Constants/names are variants, not independent tasks; all variants of a
    function family share one holdout group. Outputs use the bounded interpreter.
    """
    rows=[]
    recipes=[
        ('scale','n','return n * C','multiply its numeric input by C',[[0],[2],[-3]],'The multiplication scales the input by C.'),
        ('offset','n','return n + C','add C to its numeric input',[[0],[2],[-3]],'Adding C shifts the input by that amount.'),
        ('difference','n','return n - C','subtract C from its numeric input',[[0],[7],[-3]],'The function subtracts C from the input.'),
        ('above','n','return n > C','test whether its numeric input is greater than C',[[0],[30],[-3]],'The comparison returns True or False.'),
        ('below','n','return n < C','test whether its numeric input is less than C',[[0],[30],[-3]],'The comparison returns True or False.'),
        ('multiple','n','return n % C == 0','test whether its integer input is divisible by C',[[0],[12],[-7]],'A zero remainder means exact divisibility.'),
        ('filter_high','numbers','return [n for n in numbers if n > C]','keep the numbers strictly greater than C in their original order',[[[]],[[0,3,20]],[[-2,30]]],'The comprehension keeps only numbers that satisfy the condition.'),
        ('filter_low','numbers','return [n for n in numbers if n < C]','keep the numbers strictly less than C in their original order',[[[]],[[0,3,20]],[[-2,30]]],'The comprehension keeps only numbers that satisfy the condition.'),
        ('add_each','numbers','return [n + C for n in numbers]','add C to each number and return a new list',[[[]],[[0,3,20]],[[-2,30]]],'It builds a new list without changing the input list.'),
        ('scale_each','numbers','return [n * C for n in numbers]','multiply each number by C and return a new list',[[[]],[[0,3,20]],[[-2,30]]],'It builds a new list without changing the input list.'),
        ('take_first','items','return items[:C]','return the first C items as a new list, or all items if fewer exist',[[[]],[[1,2,3,4,5]],[[9]]],'Slicing also works when the list is shorter than the requested count.'),
        ('take_last','items','return items[-C:]','return the last C items as a new list, or all items if fewer exist',[[[]],[[1,2,3,4,5]],[[9]]],'Negative slicing starts from the end of the list.'),
    ]
    basic=[
        ('text_lower','text','return text.lower()','return the text in lowercase',[[''],['Ab Cd'],['XYZ']],'lower() changes uppercase letters to lowercase.'),
        ('text_upper','text','return text.upper()','return the text in uppercase',[[''],['Ab Cd'],['xyz']],'upper() changes lowercase letters to uppercase.'),
        ('trim_text','text','return text.strip()','remove whitespace at both ends of the text',[[''],['  hello  '],['a b']],'strip() preserves whitespace inside the text.'),
        ('reverse_items','items','return items[::-1]','return a reversed copy of the input list',[[[]],[[1,2,3]],[['a','b']]],'The negative slice step reverses the order.'),
        ('sum_items','numbers','return sum(numbers)','return the sum of the numbers, or zero for an empty list',[[[]],[[1,2,3]],[[-5,2]]],'sum() adds every number; an empty list totals zero.'),
        ('positive_only','numbers','return [n for n in numbers if n > 0]','keep only strictly positive numbers in their original order',[[[]],[[-2,0,4]],[[-1,-3]]],'The condition excludes both zero and negative numbers.'),
        ('count_entries','items','return len(items)','return the number of items in a list',[[[]],[[1,2,3]],[['a']]],'len() counts the entries in the list.'),
        ('join_parts','parts','return "-".join(parts)','join a list of strings with hyphens',[[[]],[['a','b']],[['z']]],'join() inserts the separator between strings.'),
    ]
    for family,args,body,task,inputs,explanation in recipes+basic:
        for variant,c in enumerate((2,3,4,5,6,7,8,9)):
            fn=f'{family}_{c}';code=f'def {fn}({args}):\n    '+body.replace('C',str(c))
            # Read the original task anew for each constant.
            original=next(t[3] for t in recipes+basic if t[0]==family)
            explanation0=next(t[5] for t in recipes+basic if t[0]==family)
            tests=[dict(function=fn,args=x,expected=run_function(ast.parse(code),fn,x)) for x in inputs]
            response='```python\n'+code+'\n```'
            if verify_target('',response,tests=tests,strict=True) is None:raise ValueError('Authored function failed')
            prompt=f'Write a Python function {fn}({args}) to '+original.replace('C',str(c))+'. Return only a Python code block.'
            follow=inputs[1];call=fn+'('+', '.join(repr(x) for x in follow)+')'
            result=tests[1]['expected']
            messages=pair(prompt,response)+pair(f'What does {call} return? Give only the Python value.',repr(result))+pair('Why does the function return that result?',explanation0.replace('C',str(c)))
            rows.append(dict(component='python',bucket='python_local',group='instruction-python-family:'+family,messages=messages,
                functional_assertions=tests,foundation_kind=family,provenance=dict(repo='local/instruction-python-v1',revision='instruction-python-v1',split='authored',license='CC0-1.0',method='Finite 20 function families, eight variants each; bounded-interpreter outputs; grouped by family; no downloaded-code execution')))
    return rows


def prepare(config):
    plan=read_json(config);dest=local(plan['prepared'])
    if dest.exists():
        result=verify(dest)
        if result['plan_sha256']!=file_sha256(config):raise ValueError('Prepared configuration changed')
        return result
    tokenizer=Tokenizer(local(plan['tokenizer']));rng=random.Random(plan['seed'])
    blocked,bench=blocked_questions(local('pretraining/data/.sft_hf_cache'))
    oldgroups,oldprompts,oldreceipts=prior_exclusions(include_all=True);blocked.update(oldprompts)
    from .behavior import cases as historical_cases
    for suite in ('chat-relevance-v2','conversation-v3','conversation-v4','conversation-v5','instruction-v6'):
        for split in ('dev','test'):
            for case in historical_cases(split,suite):
                blocked.update(digest(m['content']) for m in case['messages'] if m['role']=='user' and not social_prompt(m['content']))
    for split in ('dev','test'):
        blocked.update(digest(q) for case in rollouts(split) for q in case['prompts'])
    # Include actual-history diagnostics inspected before this decision.
    inspected=local('reports/sft/SFT-017-recreate-014-20261008/diagnostic-protocol.json')
    if inspected.exists():
        def exclude(obj):
            if isinstance(obj,dict):
                if obj.get('role')=='user' and not social_prompt(obj.get('content','')):blocked.add(digest(obj['content']))
                for k,v in obj.items():
                    if k=='prompts':blocked.update(digest(q) for q in v)
                    else:exclude(v)
            elif isinstance(obj,list):
                for v in obj:exclude(v)
        exclude(read_json(inspected));oldreceipts.append(dict(path=str(inspected),sha256=file_sha256(inspected)))
    arcgrams=set()
    for cfg in ('ARC-Easy','ARC-Challenge'):
        folder=local('pretraining/data/.sft_hf_cache/allenai___ai2_arc')/cfg/'0.0.0'/'210d026faf9955653af8916fad021475a3f00453'
        for path in folder.glob('*.arrow'):
            for q in Dataset.from_file(str(path))['question']:
                words=normalized(q).split();arcgrams.update(' '.join(words[i:i+8]) for i in range(len(words)-7))
    pools=defaultdict(lambda:defaultdict(list));rejects=Counter();seen=set();prompt_groups={};source_receipts=[]
    def accept(row):
        bucket=row['bucket'];component=row['component']
        try:
            group=row['group'];messages=row['messages']
            if group in oldgroups:raise ValueError('previous_holdout_group')
            if any(digest(m['content']) in blocked for m in messages if m['role']=='user' and not social_prompt(m['content'])):raise ValueError('heldout_or_benchmark_prompt')
            users=' '.join(m['content'] for m in messages if m['role']=='user');answers=' '.join(m['content'] for m in messages if m['role']=='assistant')
            words=normalized(users).split()
            if any(' '.join(words[i:i+8]) in arcgrams for i in range(len(words)-7)):raise ValueError('arc_eight_word_overlap')
            if re.search(r'\b(?:role.?play|pretend|act as|impersonate|you are (?:a|an|the)|diagnos\w*|prescrib\w*|legal advice|investment advice|latest|current president|today.s date|near me)\b',users,re.I):raise ValueError('persona_or_specialist')
            if component!='python' and re.search(r'\b(?:statistical|statistics|standard (?:deviation|error)|confidence interval|skewness|calculus|polynomial|coefficients|derivative|integral|investment|stocks|insurance|custody|visitation|legal|lawyer|rights act|medication|disease|cancer|treatment|allerg\w*|currently (?:popular|president|trending|available)|right now|this season|trending|wind-resistant|nutrition labels)\b',users+' '+answers,re.I):raise ValueError('specialist_or_stale_target')
            if re.search(r'\b(?:begin by having the user|you can now assist|let the user know|provide you their input|your editing services)\b',users,re.I):raise ValueError('meta_task_persona')
            if bucket=='magpie' and re.search(r'\b(?:assist (?:the )?user|assisting the user|do you understand what|in this interaction|coding assistance|ready to assist|response did not appear|data analysis|data visualizations|strategy session|you are to provide)\b',users,re.I):raise ValueError('meta_readiness_without_task')
            if re.search(r'\b(?:play along|play as|assign you a character|what character|as a character|specialized AI assistant|instructions you have received|how you will assist|function in editing)\b',users,re.I):raise ValueError('roleplay_or_meta_assignment')
            if bucket=='magpie' and re.search(r'\b(?:edit\w*|proofread|rewrit\w*|rephras\w*)\b',users,re.I) and not re.search(r'[\"\']|:\s*\S|following (?:sentence|text|paragraph)',users,re.I):raise ValueError('editing_without_supplied_text')
            if component!='python' and re.search(r'\b(?:confidence|correlation|median|skew\w*|gross profit|gross margin|data ?set|calculate|equation|probability|supplements?|vitamins?|symptoms?|antibiotic|infection|bleeding|pharmacy|stock prices?|court|doctor|time traveler)\b',users+' '+answers,re.I):raise ValueError('outside_basic_text_scope')
            if re.search(r"(?:you are|you're) (?:a|an) (?:literary|creative|professional|personal|math|English)|\b(?:we have|we sell|our store|our menu|our restaurant)\b",users+' '+answers,re.I):raise ValueError('specialized_or_shop_persona')
            if re.search(r"(?:you are|you're) (?:a|an|the) |\b(?:expert at|my passion|for centuries|my knowledge cutoff|i was trained|massive corpus|my training data)\b",users+' '+answers,re.I):raise ValueError('role_assignment_or_unsupported_identity')
            if component!='python' and (len(re.findall(r'\b\d+\b',users))>=4 or re.search(r'\b(?:quantifier|model.theoretic|nitrogen assimilation|sales data|first two sentences|your previous (?:message|answer)|last message)\b',users,re.I)):raise ValueError('complex_or_missing_source_context')
            if re.search(r'\b(?:my name is|i live in|i was born|my wife|my husband|you can help me|i am (?:alex|sarah)|my hometown)\b',answers,re.I):raise ValueError('invented_identity_or_role_reversal')
            if re.search(r'\b(?:revised version|corrected sentence|your revised sentence)\b',answers,re.I) and not re.search(r'\b(?:rewrite|revise|edit|grammar|correct|proofread)\b',users,re.I):raise ValueError('unrequested_editing')
            if component!='python' and ('```' in users+answers or re.search(r'\b(?:javascript|typescript|SQL|C\+\+|C#|Ruby|Python|programming|implement|algorithm)\b',users,re.I)):raise ValueError('code_requires_verified_bucket')
            key=digest(json.dumps(messages,sort_keys=True))
            if key in seen:raise ValueError('duplicate')
            prompt_keys={digest(m['content']) for m in messages if m['role']=='user' and len(m['content'].split())>=10}
            if any(k in prompt_groups and prompt_groups[k]!=group for k in prompt_keys):raise ValueError('shared_substantive_prompt_group')
            split=split_group(group,plan['seed'],plan['heldout_group_percent'])
            cap=plan['bucket_caps'][bucket] if split=='train' else plan['heldout_per_bucket']
            if len(pools[bucket][split])>=cap:raise ValueError('cap')
            messages,ids,mask=encode(messages,tokenizer,plan,component,group)
            pools[bucket][split].append(dict(row,messages=messages,split=split,_ids=ids,_mask=mask));seen.add(key)
            prompt_groups.update({k:group for k in prompt_keys})
        except (ValueError,SyntaxError) as e:rejects[bucket+':'+str(e)]+=1
    text=next(s for s in plan['sources'] if s['component']=='chat')
    raw,receipt=cached(text);source_receipts.append(dict(source=text,**receipt))
    indices=list(range(len(raw)));rng.shuffle(indices)
    review_exclusions={(x['repo'],x.get('row')) for x in plan.get('review_exclusions',[])}
    allowed={'everyday-conversations':'everyday','smol-magpie-ultra-short':'magpie','smol-contraints':'constraints'}
    for index in indices:
        r=raw[index];bucket=allowed.get(r['source'])
        if not bucket:continue
        if (text['repo'],index) in review_exclusions:rejects[bucket+':manual_review']+=1;continue
        if all(len(pools[bucket][s])>=(plan['bucket_caps'][bucket] if s=='train' else plan['heldout_per_bucket']) for s in ('train','dev','test')):continue
        try:
            messages=text_prefix(r['messages'],tokenizer,plan)
            first=next((m['content'] for m in messages if m['role']=='user' and not social_prompt(m['content'])),messages[0]['content'])
            row=dict(component='constraints' if bucket=='constraints' else 'chat',bucket=bucket,group='instruction-first:'+digest(first),messages=messages,
                provenance=dict(text,row=index,source_subset=r['source'],transformation='Complete initial exchanges up to configured turn/length budget; source system preserved when compatible; no shortened assistant answer'))
            if bucket=='constraints':
                if len([m for m in messages if m['role']=='user'])!=1:raise ValueError('constraint_turn_count')
                rules=constraint_rules(first)
                if not rules or not check_rules(messages[-1]['content'],rules):raise ValueError('unsupported_or_failed_constraint')
                row['rules']=rules
            accept(row)
        except ValueError as e:rejects[bucket+':'+str(e)]+=1
    foundation=[*focused_curriculum(),*correction_curriculum()];rng.shuffle(foundation)
    for row in foundation:
        row=dict(row,component='chat',bucket='foundation');row.pop('rules',None);accept(row)
    python=next(s for s in plan['sources'] if s['component']=='python')
    raw,receipt=cached(python);source_receipts.append(dict(source=python,**receipt))
    indices=list(range(len(raw)));rng.shuffle(indices)
    for index in indices:
        if (python['repo'],index) in review_exclusions:rejects['python_hf:manual_review']+=1;continue
        try:
            row=python_row(raw[index],index,python);prompt=row['messages'][0]['content']
            if len(prompt.split())>100 or re.search(r'\b(?:file path|operating system|line separators|branch.free|branching|URL|http|unicode|split_pattern|HTTP|round\w*|nearest|epsilon|decimal precision|SSH|XML|HTML|escaping|escape_xml|neural network|formula|geolocation|type inference|unpacking|invalid inputs?|input.*invalid|alias|same object|stub|placeholder|intervals?)\b',prompt,re.I):raise ValueError('ambiguous_or_specialist_python')
            code=row['messages'][1]['content'].split('```python\n',1)[1].split('```',1)[0]
            tree=ast.parse(code)
            if any(isinstance(n,(ast.Raise,ast.Try,ast.While,ast.Import,ast.ImportFrom,ast.Global,ast.Nonlocal)) for n in ast.walk(tree)):raise ValueError('unsupported_branch_or_side_effect')
            if re.search(r'\b(?:tuple|two strings)\b',prompt,re.I) and re.search(r'\b(?:returns?|output)\b.*\b(?:tuple|two strings)\b',prompt,re.I) and not all(isinstance(ast.literal_eval(t['expected_literal']),tuple) for t in row['functional_assertions']):raise ValueError('return_type_contract_conflict')
            accept(dict(row,bucket='python_hf'))
        except (ValueError,SyntaxError,TypeError) as e:rejects['python_hf:'+str(e)]+=1
    for row in python_curriculum():accept(row)
    rows_by_split=defaultdict(list);counts={}
    for bucket,pool in pools.items():
        for split,rows in pool.items():rows_by_split[split].extend(rows)
    if len(rows_by_split['train'])<2000:raise ValueError('Too little clean instruction data')
    staging=dest.with_name('.'+dest.name+'-'+uuid.uuid4().hex);staging.mkdir(parents=True)
    try:
        reviews=[]
        for split in ('train','dev','test'):
            rows=rows_by_split[split];rng.shuffle(rows)
            components=Counter(r['component'] for r in rows)
            if any(components[c]<8 for c in ('chat','constraints','python')):raise ValueError('Too few '+split+' components: '+str(components))
            counts[split]=dict(conversations=len(rows),semantic_groups=len({r['group'] for r in rows}),components=dict(components),buckets=dict(Counter(r['bucket'] for r in rows)),
                input_tokens=sum(len(r['_ids'])-1 for r in rows),assistant_tokens=sum(int(r['_mask'].sum()) for r in rows),
                assistant_tokens_by_component={c:sum(int(r['_mask'].sum()) for r in rows if r['component']==c) for c in components},
                multi_turn=sum(sum(m['role']=='user' for m in r['messages'])>1 for r in rows))
            for bucket in plan['bucket_caps']:
                reviews.extend({k:v for k,v in r.items() if not k.startswith('_')} for r in [r for r in rows if r['bucket']==bucket][:12])
            write_split(staging,split,rows)
        atomic_json(staging/'review_examples.json',reviews)
        atomic_json(staging/'curation.json',dict(counts=counts,rejected=dict(rejects),source_receipts=source_receipts,previous_holdout_exclusions=oldreceipts,benchmark_exclusions=bench,
            requested_bucket_caps=plan['bucket_caps'],limitations=['Text targets structurally filtered and spot reviewed, not a universal truth certificate.','Python source tests cover supplied examples; unsupported/ambiguous tasks rejected.','Authored Python variants share family groups and are not independent tasks.','Complete initial dialogue prefixes preserve all retained context; longer later exchanges omitted.','Exact/eight-word exclusion is incomplete semantic decontamination; no ARC training targets or ARC selection.']))
        atomic_json(staging/'diagnostic_protocol.json',dict(suite='instruction-v6',selection='dev only',dev=cases('dev'),test=cases('test'),dev_rollouts=rollouts('dev'),test_rollouts=rollouts('test')))
        files={p.name:dict(sha256=file_sha256(p),bytes=p.stat().st_size) for p in staging.iterdir()}
        manifest=dict(format='opensml-sft-corpus-v1',plan_sha256=file_sha256(config),tokenizer=tokenizer.fingerprint,sources=plan['sources'],counts=counts,files=files,seed=plan['seed'],
            serialization='explicit-ids-assistant-mask-v1',epochs=plan['epochs'],final_quality_version='instruction-chat-v1',total_steps=math.ceil(counts['train']['conversations']/64)*plan['epochs'])
        atomic_json(staging/'manifest.json',manifest);staging.rename(dest)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps(counts,indent=2),flush=True);return manifest
