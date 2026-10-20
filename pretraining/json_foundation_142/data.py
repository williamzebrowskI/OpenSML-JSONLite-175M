"""Explicit source/fact/schema coverage. Benchmark answers are never examples."""
import copy,json,random,hashlib
from collections import Counter,defaultdict
from jsonschema import Draft202012Validator
from .common import *
from json_learnability_141 import data as A
from .runtime import Encoding
obj=A.obj;field=A.field;string=A.string;array=A.array
ATOMIC=dict(A.SKILLS)
COMPOSITE={
 'typed_profile':'Names, booleans, nullable contact and boolean arrays together',
 'numeric_bundle':'Integer, decimal, negative value, numeric string and numeric array together',
 'optional_bundle':'Named people, required nullable room and omitted/present optional alias',
 'lists_bundle':'Separate complete people/place/product entities plus missing venue',
 'nested_delivery':'Multi-field nested recipient and address objects',
 'record_allocations':'Populated associated name/role/quantity records',
 'parallel_teams':'Two nested team membership arrays and project title',
 'exact_span_bundle':'Unicode name, complete title, escaped path and unnormalized date/time',
 'role_routes':'Sender/recipient and origin/destination, positional reversals and distractors',
 'correction_bundle':'Final person/place/time across canceled, tentative and corrected states',
 'constraints_bundle':'Const, enum, code pattern and bounded integer together',
 'local_refs_bundle':'Local referenced contact objects and lists of referenced contacts',
 'composition_bundle':'Root oneOf email/sms branches with matching populated values',
 'conditional_bundle':'Conditional required and forbidden fields, pickup/delivery',
 'maps_bundle':'Dynamic-key counts, numeric matrices and heterogeneous tuples',
 'mixed_inventory':'Populated associated product/quantity/price/boolean/nullable records',
}
POOLS={
 'train':dict(first=['Ada','Bryn','Cleo','Dara','Enzo','Faye','Galen','Hana','Iman','Jules','Kira','Luca','Mina','Nico','Oren','Pia'],last=['Ames','Baird','Cove','Dale','Esher','Frost','Grove','Hale','Ivory','Jade','Kern','Lowe','Moss','Nash','Oak','Pike'],city=['Ghent','Lund','Dijon','Trieste','Graz','Turku','Linz','Bath','Bonn','Olomouc','Padua','León']),
 'development':dict(first=['Rhea','Seth','Talia','Ulric','Veda','Wren','Yara','Zain'],last=['Quill','Roux','Stone','Thorne','Upton','Vane','Wells','York'],city=['Roskilde','Vannes','Tartu','Modena','Uppsala','Namur']),
 'reserved':dict(first=['Arielle','Bodhi','Chiara','Devon','Elara','Florian','Giada','Hector'],last=['Arden','Belmont','Caspar','Delmar','Eldon','Finch','Grey','Hart'],city=['Aalborg','Besançon','Klaipėda','Perugia','Sligo','Umeå'])}


def person(p,j):return p['first'][j%len(p['first'])]+' '+p['last'][(j//len(p['first']))%len(p['last'])]
def replace_strings(v,mapping):
    if isinstance(v,str):
        for a,b in sorted(mapping.items(),key=lambda x:-len(x[0])):v=v.replace(a,b)
        return v
    if isinstance(v,list):return [replace_strings(x,mapping) for x in v]
    if isinstance(v,dict):return {k:replace_strings(x,mapping) for k,x in v.items()}
    return v


def atomic(skill,split,i):
    # Reuse the tested single-feature grammar, with fresh source indices and pools.
    j={'train':50,'development':110,'reserved':150}[split]+i
    text,schema,target,tags=A.scenario(skill,'train',j);n=j+1;p=POOLS[split]
    olda=['Anika','Bastien','Celia'][n%3]+' Linden';oldb=['Dario','Emil','Farah'][n%3]+' Corven'
    mapping={olda:person(p,i),oldb:person(p,i+19),f'Brno District {n}':p['city'][i%len(p['city'])],f'Parma Quarter {n}':p['city'][(i+1)%len(p['city'])],f'Nov. the {n}th at 7 pm':f'November {(i%28)+1} at 7 pm'}
    text=replace_strings(text,mapping);schema=replace_strings(schema,mapping);target=replace_strings(target,mapping)
    # Remove the old log marker and use new split-specific prose envelopes.
    body=text.split('. ',1)[1]
    wrappers={'train':['A clerk wrote: ','According to the submitted note, ','The report reads: ','Please use these recorded facts: '],
              'development':['A fresh request states: ','The message supplied for processing says: '],
              'reserved':['Here is a new submission: ','The latest intake note records: ']}
    body=wrappers[split][i%len(wrappers[split])]+body
    return body,schema,target,tags


def facts(split,i):
    p=POOLS[split];base={'train':0,'development':200,'reserved':400}[split]+i
    return dict(a=person(p,i),b=person(p,i+19),c=person(p,i+37),city=p['city'][i%len(p['city'])],other=p['city'][(i+1)%len(p['city'])],
        country={'Ghent':'Belgium','Lund':'Sweden','Dijon':'France','Trieste':'Italy','Graz':'Austria','Turku':'Finland','Linz':'Austria','Bath':'United Kingdom','Bonn':'Germany','Olomouc':'Czechia','Padua':'Italy','León':'Spain','Roskilde':'Denmark','Vannes':'France','Tartu':'Estonia','Modena':'Italy','Uppsala':'Sweden','Namur':'Belgium','Aalborg':'Denmark','Besançon':'France','Klaipėda':'Lithuania','Perugia':'Italy','Sligo':'Ireland','Umeå':'Sweden'}[p['city'][i%len(p['city'])]],title=f'The Silver Orchard Beyond Chapter {base+1}',code=f'ZX-{base+10:03d}',
        quantity=i%10,price=round(1.25+(i%17)*.5,2),delta=-round(.5+(i%7)*.25,2),time=f'{14+i%7}:30',oldtime=f'{8+i%4}:00',index=base+1)


def composite(family,split,i):
    f=facts(split,i);a,b,c=f['a'],f['b'],f['c'];city,other=f['city'],f['other'];k=f['index'];frame=i%4
    if family=='typed_profile':
        active=bool(i%2);archived=bool((i//2)%2);phone=None if i%3==0 else f'040-5{k:03d}';checks=[active,archived,False]
        target=dict(holder=a,active=active,archived=archived,phone=phone,checks=checks)
        schema=obj(dict(holder=string('Full account holder name'),active=field('boolean','Explicit active flag'),archived=field('boolean','Explicit archived flag'),phone=field(['string','null'],'Stated phone, or null when absent'),checks=array(field('boolean','One explicit check result'),'All check flags in stated order, including repeated flags')))
        phone_clause='No phone is recorded.' if phone is None else f'The phone is {phone}.'
        clauses=[f'The account belongs to {a}. Its active flag is {str(active).lower()} and its archived flag is {str(archived).lower()}.',f'For {a}, archived is explicitly {str(archived).lower()}; active is explicitly {str(active).lower()}.',f'{a} has active={str(active).lower()}, archived={str(archived).lower()}.',f'Holder {a}: the active status reads {str(active).lower()}, while archived reads {str(archived).lower()}.']
        text=clauses[frame]+' '+phone_clause+' Recorded checks, in order: '+', '.join(str(x).lower() for x in checks)+'.'
    elif family=='numeric_bundle':
        code=f'{k:05d}';target=dict(quantity=f['quantity'],unit_price=f['price'],change=f['delta'],code=code,readings=[f['price'],f['delta'],0])
        schema=obj(dict(quantity=field('integer','Approved item count'),unit_price=field('number','Price per item'),change=field('number','Signed recorded change'),code=string('Only the batch code, preserving leading zeroes'),readings=array(field('number','One recorded reading'),'All readings in order')))
        parts=[f"Batch code {code} contains {f['quantity']} units at {f['price']} per unit.",f"For code {code}, the unit price is {f['price']} and quantity is {f['quantity']}.",f"Approve {f['quantity']} items for batch {code}; price each is {f['price']}.",f"The batch identified by {code} has quantity {f['quantity']}, with unit price {f['price']}."]
        text=parts[frame]+f" The recorded change is {f['delta']}. Readings: {f['price']}, {f['delta']}, 0."
    elif family=='optional_bundle':
        guests=[] if i%4==0 else [a,b][:1+i%2];room=None if i%2==0 else f'Suite {k}';target=dict(guests=guests,room=room)
        schema=obj(dict(guests=array(string('Guest name'),'Registered guests only, in source order'),room=field(['string','null'],'Assigned room, otherwise null'),alias=string('Alias only when explicitly stated')),required=['guests','room'])
        if i%3:target['alias']=f'Star{k}'
        guest='No guests are registered.' if not guests else 'Registered guests: '+' and '.join(guests)+'.'
        roomline='No room has been assigned.' if room is None else f'The assigned room is {room}.'
        text=[guest+' '+roomline,roomline+' '+guest,f'For this registration, {guest} {roomline}',f'Accommodation details: {roomline} {guest}'][frame]
        text+=' '+(f"The explicit alias is {target['alias']}." if 'alias'in target else 'No alias is supplied.')+f' {c} is the organizer, not a registered guest.'
    elif family=='lists_bundle':
        count=[0,1,2,3,4,5,6,8][i%8];domain=['speakers','cities','products'][i%3];p=POOLS[split]
        names=[person(p,i+j*3) for j in range(count)] if domain=='speakers' else [p['city'][(i+j)%len(p['city'])] for j in range(count)] if domain=='cities' else [f'{x} {k}' for x in ['Amber Relay','Cobalt Sensor','Silver Gauge','Copper Valve','Ivory Probe','Cedar Cable','Violet Meter','Jade Panel'][:count]]
        target={domain:names,'venue':None};schema=obj({domain:array(string('One complete matching entity'),f'All explicitly listed {domain}, each as a separate item in source order'), 'venue':field(['string','null'],'Named venue or null if no venue is supplied')})
        value='; '.join(names);text=(f'No {domain} have been selected.' if count==0 else [f'The selected {domain} are {value}.',f'Include these {domain}: {value}.',f'The request lists {value} as its {domain}.',f'For {domain}, keep this order: {value}.'][frame])+' No venue is named.'
    elif family=='nested_delivery':
        address=obj(dict(city=string('Recipient city'),country=string('Recipient country')));contact=obj(dict(name=string('Full recipient name'),address=address));target=dict(delivery=dict(recipient=dict(name=a,address=dict(city=city,country=f['country']))));schema=obj(dict(delivery=obj(dict(recipient=contact))))
        text=[f"Send the parcel to {a}, who receives it in {city}, {f['country']}.",f"The delivery recipient is {a}. The receiving address is in {city}, {f['country']}.",f"Recipient {a} is based in {city} in {f['country']}; deliver there.",f"For delivery, the contact is {a}; recipient city: {city}; recipient country: {f['country']}."][frame]+f' The sender is {b}, based in {other}.'
    elif family in ['record_allocations','mixed_inventory']:
        count=1+i%4;p=POOLS[split];records=[];clauses=[]
        for j in range(count):
            name=person(p,i+j*3);quantity=(i+j)%7;role=['editor','designer','reviewer','translator'][j];item=f"{['Amber Relay','Cobalt Sensor','Cedar Panel','Ivory Gauge'][j]} {k}"
            if family=='record_allocations':
                v=dict(name=name,role=role,tasks=quantity);line=[f'{name} is the {role} with {quantity} tasks',f'Assign the {role} role to {name}, with task count {quantity}',f'{name} has {quantity} tasks as {role}',f'The {role} is {name}; tasks assigned: {quantity}'][frame]
            else:
                flag=bool((i+j)%2);ref=None if i%2==0 else f'R{k}-{j}';v=dict(product=item,quantity=quantity,price=f['price'],approved=flag,reference=ref)
                line=f"{item}: quantity {quantity}, unit price {f['price']}, approved {str(flag).lower()}"+('; no reference was provided' if ref is None else f'; reference {ref}')
            records.append(v);clauses.append(line)
        if family=='record_allocations':
            item_schema=obj(dict(name=string('Full name for this person'),role=string('Role of that same person'),tasks=field('integer','Task count for that person')));root='people'
        else:
            item_schema=obj(dict(product=string('Complete product name'),quantity=field('integer','Quantity for that product'),price=field('number','Unit price for that product'),approved=field('boolean','Explicit approval flag'),reference=field(['string','null'],'That product reference, null if absent')));root='items'
        target={root:records};schema=obj({root:array(item_schema,'One record per listed entry, preserving associations and source order')});text='. '.join(clauses)+'.'
    elif family=='parallel_teams':
        design=[a,b] if i%2 else [a];review=[c];target=dict(design=dict(members=design),review=dict(members=review),project=f['title'])
        schema=obj(dict(design=obj(dict(members=array(string('Member name'),'Design-team members only'))),review=obj(dict(members=array(string('Member name'),'Review-team members only'))),project=string('Entire project title')))
        d=' and '.join(design);v=' and '.join(review)
        text=[f"Project {f['title']}. Design team: {d}. Review team: {v}.",f"The review team is {v}; the design team is {d}. They work on {f['title']}.",f"For {f['title']}, {d} do design, while {v} do review.",f"Project title: {f['title']}. Review membership: {v}. Design membership: {d}."][frame]
    elif family=='exact_span_bundle':
        label=['Zoë','Élodie','Søren','İpek'][i%4]+' '+a.split()[-1];title=f['title']+' — "annotated edition"';folder=f'C:\\projects\\set-{k}';when=['next Wednesday',f'November {(i%28)+1} at noon',f'2027-06-{i%28+1:02d}', 'the first Monday in July'][i%4]
        target=dict(label=label,title=title,folder=folder,when=when);schema=obj(dict(label=string('Entire contact label exactly as stated'),title=string('Complete title between the title delimiters'),folder=string('Exact folder path, preserving backslashes'),when=string('Complete schedule expression, without normalization')))
        lines=[f'Label: {label}.',f'Title begins [{title}] title ends.',f'Folder path: {folder}.',f'Schedule: {when}.'];text=' '.join(lines[frame:]+lines[:frame])
    elif family=='role_routes':
        target=dict(sender=a,recipient=b,origin=city,destination=other);schema=obj(dict(sender=string('Person sending the parcel'),recipient=string('Person receiving the parcel'),origin=string('Sender city'),destination=string('Recipient city')))
        text=[f'{a} from {city} sends the parcel to {b} in {other}.',f'The parcel is for {b}, in {other}; {a} sent it from {city}.',f'Recipient: {b}, {other}. Sender: {a}, {city}.',f'From {city}, {a} dispatches a parcel whose recipient is {b} at {other}.'][frame]+f' {c} is only the courier.'
    elif family=='correction_bundle':
        target=dict(person=a,city=other,time=f['time']);schema=obj(dict(person=string('Person whose booking is described'),city=string('Final confirmed destination'),time=string('Final confirmed time, copied exactly')))
        text=[f"{a} first booked {city} at {f['oldtime']}, then replaced both with {other} at {f['time']}.",f"The confirmed booking for {a} is {other} at {f['time']}. The earlier {city} booking at {f['oldtime']} was canceled.",f"For {a}, change the time from {f['oldtime']} to {f['time']} and the city from {city} to {other}.",f"{a} considered {city} at {f['oldtime']}. Final confirmation: {other}, {f['time']}. Earlier proposals are superseded."][frame]
    elif family=='constraints_bundle':
        priority=['low','medium','high'][i%3];target=dict(kind='order',code=f['code'],priority=priority,quantity=f['quantity']);schema=obj(dict(kind=string('Required record kind',const='order'),code=string('Full order code',pattern='^ZX-[0-9]{3}$',minLength=6,maxLength=6),priority=string('Explicit priority',enum=['low','medium','high']),quantity=field('integer','Approved quantity',minimum=0,maximum=9)))
        text=[f"Order {f['code']}: approved quantity {f['quantity']}; priority {priority}.",f"The priority is {priority} for order {f['code']}, which has approved quantity {f['quantity']}.",f"Approve {f['quantity']} units for order {f['code']} with {priority} priority.",f"An order with code {f['code']} is marked {priority}; quantity approved: {f['quantity']}."][frame]
    elif family=='local_refs_bundle':
        contact=obj(dict(name=string('Full contact name'),city=string('City for that contact')))
        if i%2:target=dict(contacts=[dict(name=a,city=city),dict(name=b,city=other)]);schema=obj(dict(contacts=array({'$ref':'#/$defs/contact'},'Contacts in stated order')),**{'$defs':dict(contact=contact)})
        else:target=dict(primary=dict(name=a,city=city),backup=dict(name=b,city=other));schema=obj(dict(primary={'$ref':'#/$defs/contact'},backup={'$ref':'#/$defs/contact'}),**{'$defs':dict(contact=contact)})
        text=f'Primary contact: {a} in {city}. Backup contact: {b} in {other}.' if i%2==0 else f'Contacts, in order: {a} from {city}; {b} from {other}.'
    elif family=='composition_bundle':
        email=bool(i%2);contact=obj(dict(kind=dict(const='email'),address=string('Email address')));sms=obj(dict(kind=dict(const='sms'),phone=string('Phone number')))
        branch=contact if email else sms;target=dict(kind='email',address=f'contact{k}@example.net') if email else dict(kind='sms',phone=f'040-{k:04d}')
        schema=dict(type='object',oneOf=[contact,sms]);text=f"The contact kind is email and its address is contact{k}@example.net." if email else f'The contact kind is sms and its phone number is 040-{k:04d}.'
    elif family=='conditional_bundle':
        mode='pickup' if i%2==0 else 'delivery';target=dict(mode=mode,**({'location':f'Depot {k}'} if mode=='pickup' else {'address':f'{k} Pine Road'}))
        schema=obj(dict(mode=string('Explicit fulfillment mode',enum=['pickup','delivery']),location=string('Pickup location'),address=string('Delivery address')),required=['mode'],**{'if':dict(properties=dict(mode=dict(const='pickup')),required=['mode']),'then':dict(required=['location'],**{'not':dict(required=['address'])}),'else':dict(required=['address'],**{'not':dict(required=['location'])})})
        text=f'Fulfillment is pickup at Depot {k}.' if mode=='pickup' else f'The fulfillment mode is delivery to {k} Pine Road.'
    elif family=='maps_bundle':
        mode=i%3
        if mode==0:target=dict(counts={f'part{k}':i%7,f'part{k+1}':i%7+1});schema=obj(dict(counts=dict(type='object',description='Map each named part to its integer count',additionalProperties=dict(type='integer'))));text=f'Inventory: part{k} has count {i%7}; part{k+1} has count {i%7+1}.'
        elif mode==1:target=dict(matrix=[[i%7,0],[2,i%7+1]]);schema=obj(dict(matrix=array(array(field('integer','One numeric cell'),'Cells in row order'),'Rows of the matrix')));text=f'Matrix row one: {i%7}, 0. Row two: 2, {i%7+1}.'
        else:target=dict(record=[f'unit{k}',i%7,bool(i%2)]);schema=obj(dict(record=dict(type='array',description='Label, count, enabled flag in that order',prefixItems=[dict(type='string'),dict(type='integer'),dict(type='boolean')],items=False,minItems=3)));text=f'The label is unit{k}, the count is {i%7}, and enabled is {str(bool(i%2)).lower()}.'
    else:raise ValueError(family)
    # Split-specific envelopes; family grammar is deliberately shared and disclosed.
    prefixes={'train':['The submitted message says: ','A coordinator reports: ','Please record this request: ','The source note states: '], 'development':['A new intake contains: ','A separate message reads: '], 'reserved':['A later report supplies: ','New evidence was submitted: ']}
    return prefixes[split][i%len(prefixes[split])]+text,schema,target,[family,'combined_skills']


def root_names(s):
    result=set(s.get('properties',{}))
    for k in ['oneOf','anyOf','allOf']:
        for v in s.get(k,[]):result.update(root_names(v))
    return result


def rename_root(schema,target):
    schema=copy.deepcopy(schema)
    style=int(hashlib.sha256(dumps(schema).encode()).hexdigest()[:8],16)%4
    key=lambda j:['field_'+chr(97+j),'slot_'+str(j+1),'value_'+str(j+1),'entry_'+chr(97+j)][style]
    mapping={k:key(j) for j,k in enumerate(sorted(root_names(schema)))}
    def walk(s):
        if 'properties'in s:s['properties']={mapping[k]:v for k,v in reversed(list(s['properties'].items()))}
        if 'required'in s:s['required']=[mapping[k] for k in reversed(s['required'])]
        for k in ['if','then','else','not']:
            if k in s:walk(s[k])
        for k in ['oneOf','anyOf','allOf']:
            for v in s.get(k,[]):walk(v)
    walk(schema);return schema,{mapping[k]:v for k,v in target.items()}


def views(schema,target,n):
    renamed,other=rename_root(schema,target)
    if n==0:return copy.deepcopy(schema),copy.deepcopy(target),'canonical'
    if n==1:return renamed,other,'renamed'
    inner=copy.deepcopy(schema) if n==2 else renamed;gold=copy.deepcopy(target) if n==2 else other
    defs=inner.pop('$defs',None);outer=obj({'payload':dict(inner,description='Requested structured facts from the source')})
    if defs is not None:outer['$defs']=defs
    return outer,dict(payload=gold),'wrapped' if n==2 else 'wrapped_renamed'


def row(ident,group,document,schema,expected,component,family,split,view='canonical',tags=None,provenance=None):
    return dict(id=ident,source_group=group,group=group,document=document,schema=schema,expected=expected,instruction=INSTRUCTION,
        component=component,family=family,split=split,schema_view=view,tier=component,difficulty='controlled',tags=tags or [],provenance=provenance or {})


def authored(split):
    result=[];n_atomic={'train':50,'development':2,'reserved':4}[split];n_composite={'train':100,'development':4,'reserved':8}[split]
    for skill in ATOMIC:
        for i in range(n_atomic):
            text,schema,target,tags=atomic(skill,split,i);group=f'J142-{split}-atomic-{skill}-{i:03d}'
            for v in range(2):
                s,e,view=views(schema,target,v)
                result.append(row(group+'-'+view,group,text,s,e,'atomic','atomic/'+skill,split,view,tags,dict(generator='atomic_fact_plan',source_index=i,related_grammar='141 TRAIN grammar, new source indices/pools')))
    for family in COMPOSITE:
        for i in range(n_composite):
            text,schema,target,tags=composite(family,split,i);group=f'J142-{split}-composite-{family}-{i:03d}'
            for v in range(4):
                s,e,view=views(schema,target,v)
                result.append(row(group+'-'+view,group,text,s,e,'composite','composite/'+family,split,view,tags,dict(generator='new_composite_fact_plan',source_index=i)))
    return result


def upstream():
    paths={n:ROOT/f'pretraining/data/sft_{n}_{name}_202610{day}/train.jsonl' for n,name,day in [('130','joint_foundation','16'),('140','joint_transfer','17')]}
    banks={n:read_rows(p) for n,p in paths.items()};rng=random.Random(PLAN['seed']);selected=[]
    old=[r for r in banks['130'] if r['component']in ['snips','schema_variation','grounded_qa']]
    for component in ['snips','schema_variation','grounded_qa','massive']:
        pool=[r for r in (banks['140'] if component=='massive' else old) if r['component']==component]
        if component=='snips':
            massive_source=ROOT/'pretraining/data/json_019_sources_20261011/massive-en-US.jsonl'
            test_sources={normalized(r['utt']) for r in read_rows(massive_source) if r['partition']=='test'}
            pool=[r for r in pool if normalized(r['document']) not in test_sources]
            missing=QUOTAS[component]-len(pool)
            candidates=[r for r in pool if len(r['schema']['properties'])>1]
            for j in range(max(0,missing)):
                source=candidates[j];z=copy.deepcopy(source);keys=list(z['schema']['properties'])[:-1]
                z['schema']['properties']={k:v for k,v in z['schema']['properties'].items() if k in keys};z['schema']['required']=[k for k in z['schema']['required'] if k in keys]
                z['expected']={k:v for k,v in z['expected'].items() if k in keys};z['derived142_from']=source['id'];z['id']=source['id']+':train-field-projection142'
                instruction=z['messages'][0]['content'].split('\n\n',1)[0]
                z['messages']=[dict(role='user',content=instruction+'\n\n'+dumps(dict(source_text=z['document'],schema=z['schema']))),dict(role='assistant',content=dumps(z['expected']))]
                pool.append(z)
        rng.shuffle(pool)
        for r in pool[:QUOTAS[component]]:selected.append((r,component,'140' if component=='massive' else '130'))
        assert len(pool)>=QUOTAS[component]
    authored_bank=[r for r in banks['130'] if r['component']=='authored']
    quotas=dict(binding=400,corrections=400,enumeration=400,long_copy=400,missing_roles=400,nested=200,types=200)
    for family,n in quotas.items():
        pool=[r for r in authored_bank if r['family']==family];rng.shuffle(pool);assert len(pool)>=n
        selected += [(r,'prior_authored','130') for r in pool[:n]]
    result=[]
    for i,(r,component,bank) in enumerate(selected):
        z=row(f'J142-upstream-{i:05d}',r['group'],r['document'],copy.deepcopy(r['schema']),copy.deepcopy(r['expected']),component,component+'/'+r['family'],'train',r.get('variant','original'),
            provenance=dict(source_file=str(paths[bank].relative_to(ROOT)),source_id=r.get('derived142_from',r['id']),derivation='TRAIN-only field projection' if 'derived142_from' in r else 'unchanged original TRAIN row',source_partition=r.get('source_partition','upstream TRAIN'),upstream=r.get('provenance',{})))
        # Preserve original per-task generic instruction and exact serialization.
        z['instruction']=r['messages'][0]['content'].split('\n\n',1)[0]
        if component=='massive':assert r['source_partition']=='train'
        result.append(z)
    assert Counter(r['component'] for r in result)=={k:v for k,v in QUOTAS.items() if k not in ['atomic','composite']}
    return result,paths


def compatibility():
    # TRAIN-independent observed DEV cases. No predictions consulted for selection.
    source=ROOT/'pretraining/data/sft_140_joint_transfer_20261017/development.jsonl';bank=read_rows(source);selected=[]
    for kind,n in [('snips',32),('massive',32)]:
        pool=[r for r in bank if r['component']==kind]
        if kind=='snips':
            groups=defaultdict(list)
            for r in pool:groups[r['source']].append(r)
            keys=sorted(groups);j=0
            while len(selected)<n:
                key=keys[j%len(keys)];idx=j//len(keys)
                if idx<len(groups[key]):selected.append(groups[key][idx])
                j+=1
        else:selected+=pool[:n]
    result=[]
    for r in selected:
        z=row('J142-compat:'+r['id'],r['group'],r['document'],r['schema'],r['expected'],'compatibility','compatibility/'+r['component'],'development',r.get('variant','original'),
            provenance=dict(source_file=str(source.relative_to(ROOT)),source_id=r['id']))
        z['instruction']=r['messages'][0]['content'].split('\n\n',1)[0];result.append(z)
    assert len(result)==64
    return result,source


def balanced_batches(rows,lengths):
    rng=random.Random(PLAN['seed']);pools={k:[i for i,r in enumerate(rows) if r['component']==k] for k in QUOTAS}
    for v in pools.values():rng.shuffle(v)
    previous=dict.fromkeys(QUOTAS,0);result=[];n=len(rows)
    for stop in range(64,n+64,64):
        stop=min(stop,n);quota={k:stop*QUOTAS[k]//n for k in QUOTAS};left=stop-sum(quota.values())
        for k in sorted(QUOTAS,key=lambda k:(-(stop*QUOTAS[k]%n),k))[:left]:quota[k]+=1
        ids=[]
        for k in QUOTAS:
            assert quota[k]>=previous[k];ids+=pools[k][previous[k]:quota[k]]
        ids.sort(key=lambda i:lengths[i]);result.append(ids);previous=quota
    assert len(result)==563 and sorted(i for b in result for i in b)==list(range(n))
    return result


def prepare():
    old,paths=upstream();new=authored('train');train=old+new;compat,compat_path=compatibility();dev=authored('development')+compat;reserved=authored('reserved')
    assert len(train)==36000 and len(dev)==448 and len(reserved)==768
    assert Counter(r['component'] for r in train)==QUOTAS
    # Use representative training rows, deliberately not a held-out quality score.
    fit=[]
    for family in ['atomic/'+s for s in ATOMIC]:fit+=next([r] for r in new if r['family']==family and r['schema_view']=='canonical')
    for family in ['composite/'+s for s in COMPOSITE]:fit+=[r for r in new if r['family']==family and r['schema_view']in ['canonical','renamed']][:2]
    assert len(fit)==64
    # Full source exclusion across upstream heldouts and the observed benchmark panels.
    audit_paths=[ROOT/'pretraining/data/sft_130_joint_foundation_20261016'/n for n in ['development.jsonl','reserved.jsonl']]
    audit_paths += [ROOT/'pretraining/data/sft_140_joint_transfer_20261017'/n for n in ['development.jsonl','reserved.jsonl']]
    audit_paths += [ROOT/'pretraining/data/jsonlite_eval_v1/requests.jsonl',ROOT/'reports/model-design/SFT-128_joint_extraction_20261016/release-comparison/original1444.jsonl',ROOT/'reports/model-design/SFT-128_joint_extraction_20261016/release-comparison/playground60.jsonl']
    banned=set()
    for p in audit_paths:
        banned.update(normalized(r.get('document',r.get('source_text',''))) for r in read_rows(p))
    massive_source=ROOT/'pretraining/data/json_019_sources_20261011/massive-en-US.jsonl'
    banned.update(normalized(r['utt']) for r in read_rows(massive_source) if r['partition']=='test')
    pg=ROOT/'pretraining/playground/json-technique-examples.json';banned.update(normalized(r['document']) for r in json.loads(pg.read_text())['cases'])
    assert all(normalized(r['document']) not in banned for r in train),'TRAIN source overlaps observed holdout'
    for a,b in [(train,dev),(train,reserved),(dev,reserved)]:assert not {normalized(r['document']) for r in a}&{normalized(r['document']) for r in b},'Split source overlap'
    enc=Encoding();stats={};lengths=[]
    for name,rows in [('train',train),('development',dev),('reserved',reserved),('fit_probe',fit)]:
        ls=[];ts=[]
        for r in rows:
            assert r['schema']['type']=='object' and isinstance(r['expected'],dict)
            Draft202012Validator.check_schema(r['schema']);Draft202012Validator(r['schema']).validate(r['expected'])
            x,y=enc.row(r,training=True);ls.append(len(x)+1);ts.append(int((y!=-100).sum()))
        stats[name]=dict(rows=len(rows),sources=len({normalized(r['document']) for r in rows}),components=dict(Counter(r['component'] for r in rows)),families=dict(Counter(r['family'] for r in rows)),max_sequence=max(ls),max_target=max(ts),serialized_tokens=sum(ls),assistant_tokens=sum(ts))
        if name=='train':lengths=ls
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    artifacts={'train.jsonl':train,'development.jsonl':dev,'reserved.jsonl':reserved,'fit_probe.jsonl':fit}
    for name,rows in artifacts.items():
        p=DATA/name;text=''.join(dumps(r)+'\n' for r in rows)
        if p.exists():assert p.read_text()==text,'Prepared data changed; do not mix runs'
        else:p.write_text(text)
    p=DATA/'batches.json';text=dumps(balanced_batches(train,lengths))+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)
    coverage=[]
    for component,definitions in [('atomic',ATOMIC),('composite',COMPOSITE)]:
        for name,description in definitions.items():
            family=component+'/'+name;matches=[(i+1,r) for i,r in enumerate(train) if r['family']==family]
            coverage.append(dict(family=family,description=description,training_rows=len(matches),training_sources=len({r['source_group'] for _,r in matches}),distinct_source_texts=len({r['document'] for _,r in matches}),development_rows=sum(r['family']==family for r in dev),reserved_rows=sum(r['family']==family for r in reserved),train_lines=[i for i,_ in matches],example_id=matches[0][1]['id']))
    atomic_json(REPORT/'coverage.json',coverage)
    files={n:sha(DATA/n) for n in list(artifacts)+['batches.json']}
    manifest=dict(plan=PLAN,base_sha256=BASE_SHA,parent_kind='pretrained-only; no inherited SFT weights',mixture=QUOTAS,stats=stats,coverage=coverage,files=files,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in list(paths.values())+audit_paths+[pg,compat_path,massive_source]},instruction=INSTRUCTION,
        scope='One fresh-base joint SFT. Reused130/140 TRAIN rows, new isolated and combined fact tasks. Authored frames/schema views are correlated, not independent natural traffic. Benchmarks/observed tests are evaluation-only; no exact target leakage. No promise of arbitrary JSON Schema support or zero regression.')
    p=DATA/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest
    else:atomic_json(p,manifest)
    lines=['# SFT-142 coverage: fresh pretrained foundation','','Starting weights: Stage-A pretrained132005, SHA '+BASE_SHA+'. No130/140/141 SFT checkpoint is loaded for training.','', '| Training component | Presentations |','|---|---:|']
    lines += [f'| {k} | {v:,} |' for k,v in QUOTAS.items()]
    lines += ['','| Skill/family | Included behavior | Train | Source groups | Unique source texts | Dev | Reserved | Train lines |','|---|---|---:|---:|---:|---:|---:|---|']
    for c in coverage:lines.append(f"| {c['family']} | {c['description']} | {c['training_rows']} | {c['training_sources']} | {c['distinct_source_texts']} | {c['development_rows']} | {c['reserved_rows']} | {c['train_lines'][0]}–{c['train_lines'][-1]} |")
    lines += ['','Atomic =50 sources ×2 schema views per skill; composite =100 sources ×4 schema views per family. Counts are presentations, not independent natural utterances. New DEV448 =128 atomic +256 composite +64 observed SNIPS/MASSIVE compatibility. Reserved768 remains ungenerated in the standard run. See coverage.json for exact IDs and line numbers.']
    (REPORT/'coverage.md').write_text('\n'.join(lines)+'\n')
    examples=['# One TRAIN source/schema/output example per covered family','','References are training targets, not model prompt hints.']
    for c in coverage:
        r=next(r for r in train if r['id']==c['example_id']);examples += ['', '## '+c['family'],'',r['id'],'',r['document'],'','```json',json.dumps(r['schema'],ensure_ascii=False,indent=2),'```','','Target:','```json',json.dumps(r['expected'],ensure_ascii=False,indent=2),'```']
    (REPORT/'examples.md').write_text('\n'.join(examples)+'\n')
    print(dumps(dict(status='prepared',mixture=QUOTAS,stats=stats)),flush=True)
    return manifest


if __name__=='__main__':prepare()
