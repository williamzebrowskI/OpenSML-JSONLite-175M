"""Procedural fact worlds; no model-generated labels or benchmark examples."""
import ast,copy,hashlib,json,random
from .common import dumps,text_key

NAMES={'train':['Ada Vale','Milo Reed','Lena Hart','Owen Moss','Iris Cole','Theo Finch','Nora Lane','Eli Stone'],
       'validation':['Leona Marsh','Kiran West','Amara Wells','Soren Lake']}
CITIES={'train':['Norwyn','Eldermere','Mariton','Westhaven','Asterford','Juniper Bay'],
        'validation':['Larkhaven','Cedarwick','Birchmere']}

def schema_for(value,path=()):
    description=' '.join(str(k).replace('_',' ') for k in path) or 'The complete source record'
    if value is None:return dict(type=['string','null'],description=description+'; null if absent')
    if isinstance(value,bool):return dict(type='boolean',description=description)
    if isinstance(value,int):return dict(type='integer',description=description)
    if isinstance(value,float):return dict(type='number',description=description)
    if isinstance(value,str):return dict(type='string',description=description+'; copy the full source value')
    if isinstance(value,list):
        item=value[0] if value else {'name':'example','quantity':1,'price':1.0}
        return dict(type='array',description=description+'; preserve source order and associations',items=schema_for(item,path+('each record',)))
    return dict(type='object',description=description,properties={k:schema_for(v,path+(k,)) for k,v in value.items()},required=list(value),additionalProperties=False)

def order_world(bucket,split,index):
    seed=int.from_bytes(hashlib.sha256(f'151:{bucket}:{split}:{index}'.encode()).digest()[:8],'little')
    rng=random.Random(seed);name=rng.choice(NAMES[split]);city=rng.choice(CITIES[split]);quantity=rng.randint(1,9)
    alias=rng.choice(['customer','purchaser']) if split=='train' else 'buyer'
    reference=f'{"T" if split=="train" else "V"}-{bucket[:3]}-{index:08d}'
    items=[dict(name=f'{rng.choice(["Amber","Silver","Cedar","Azure"])} {rng.choice(["notebook","lamp","case","panel","mug"])} {index}-{j}',quantity=rng.randint(1,8),price=round(rng.randint(100,9900)/100,2)) for j in range(index%5+1)]
    # Training and validation use different prose layouts and field aliases.
    depth=1+index%4;phone=None if index%4==0 else f'555-{rng.randint(100,999)}-{rng.randint(1000,9999)}'
    customer=dict(name=name,contact=dict(city=city,phone=phone))
    destination=dict(city=rng.choice([c for c in CITIES[split] if c!=city]),street=f'{rng.randint(1,900)} {rng.choice(["Pine","Willow","Oak"])} Avenue')
    order=dict(reference=reference,**{alias:customer},destination=destination,items=items,paid=bool(index%2),priority=rng.choice(['normal','urgent']))
    if depth==1:
        target=dict(reference=reference,name=name,city=city,phone=phone,quantity=quantity,paid=order['paid'])
        sentences=[f'Request {reference} is for {name} in {city}.',f'The quantity is {quantity}.',f'Payment is {"complete" if order["paid"] else "not complete"}.',f'The phone is {phone}.' if phone is not None else 'No phone number was provided.']
    else:
        root='order' if depth==2 else ('shipment' if depth==3 else 'case_file')
        body=order if depth<4 else dict(fulfillment=dict(order=order))
        target={root:body}
        sentences=[f'Record {reference}: {name} is the {alias}, whose contact city is {city}.',
          f'The delivery destination is {destination["street"]} in {destination["city"]}.',
          f'The {alias}\'s phone is {phone}.' if phone is not None else f'No phone number is available for the {alias}.',
          f'Payment is {"complete" if order["paid"] else "not complete"}; priority is {order["priority"]}.']
        for item in items:sentences.append(f'Item {item["name"]} has quantity {item["quantity"]} and price {item["price"]}.')
    corrected=index%7==0
    if corrected:
        # Only the opaque reference changes; both old/new strings are explicit.
        old=reference+'-OLD';sentences.insert(0,f'The earlier record reference was {old}. It was corrected to {reference}.')
    if split=='train':
        mode=index%3
        prose=' '.join(sentences) if mode==0 else ('\n'.join(sentences) if mode==1 else ' '.join(sentences[::2])+'\n\n'+' '.join(sentences[1::2]))
        layout=f'train-layout-{mode}'
    else:
        mode=index%2;prose='Record summary\n'+'\n'.join('- '+s for s in sentences) if mode==0 else '\n\n'.join(sentences)
        layout=f'validation-layout-{mode}'
    schema=schema_for(target)
    # An unstated optional property must not be filled with invented information.
    schema['properties']['memo']=dict(type='string',description='An explicitly supplied memo; omit when no memo is stated.')
    return dict(prose=prose,schema=schema,target=target,layout=layout,depth=depth,corrected=corrected,reference=reference,items=len(items),family='order')

def world(bucket,split,index):
    if index%8==0:return order_world(bucket,split,index//8)
    rng=random.Random(f'151-{bucket}-{split}-{index}')
    family=['order','event','support','inventory','sensor','registration','project','delivery'][index%8]
    name=rng.choice(NAMES[split]);city=rng.choice(CITIES[split]);tag=f'{split[0].upper()}-{family}-{index:09d}'
    n=1+(index//8)%6;nullable=None if (index//48)%3==0 else f'{name.replace(" ",".").lower()}-{index}@example.test'
    base={'event':dict(title=f'Autumn workshop {index}',host=dict(name=name,city=city),
                       attendees=[dict(name=f'Guest {index}-{j}',confirmed=bool(j%2)) for j in range(n)],capacity=n+10),
          'support':dict(reference=tag,requester=dict(name=name,email=nullable),
                         issues=[dict(code=f'CASE-{index}-{j}',severity=rng.choice(['low','high']),resolved=bool(j%2)) for j in range(n)]),
          'inventory':dict(warehouse=dict(city=city,manager=name),
                           products=[dict(sku=f'SKU-{index}-{j}',stock=rng.randint(0,99),price=rng.randint(100,9000)/100) for j in range(n)]),
          'sensor':dict(station=dict(name=f'Station {index}',city=city),
                        readings=[dict(sensor=f'Probe {index}-{j}',value=rng.randint(-100,450)/10,online=bool(j%2)) for j in range(n)]),
          'registration':dict(reference=tag,applicant=dict(name=name,contact=dict(city=city,email=nullable)),
                              courses=[dict(title=f'Course {index}-{j}',credits=rng.randint(1,6)) for j in range(n)],approved=bool(index%2)),
          'project':dict(title=f'Atlas project {index}',owner=dict(name=name,city=city),
                         tasks=[dict(title=f'Milestone {index}-{j}',hours=rng.randint(1,48),complete=bool(j%2)) for j in range(n)]),
          'delivery':dict(reference=tag,sender=dict(name=name,city=city),
                          recipient=dict(name=rng.choice(NAMES[split]),city=rng.choice(CITIES[split])),
                          packages=[dict(label=f'Parcel {index}-{j}',weight=rng.randint(1,99)/10,fragile=bool(j%2)) for j in range(n)])}[family]
    depth=1+(index//48)%4
    target=base
    wrappers=(['record','details','payload'] if split=='train' else ['entry','data','content'])[:depth-1]
    for key in wrappers:target={key:target}
    sentences=[]
    def describe(value,path):
        label=' / '.join(path)
        if isinstance(value,dict):
            for k,v in value.items():describe(v,path+[k.replace('_',' ')])
        elif isinstance(value,list):
            sentences.append(f'{label} has {len(value)} records, in the following order.')
            for j,v in enumerate(value):describe(v,path+[f'record {j+1}'])
        elif value is None:sentences.append(f'No {label} was supplied.')
        elif isinstance(value,bool):sentences.append(f'{label} is {"true" if value else "false"}.')
        else:
            if split=='validation':sentences.append(f'For {label}, the supplied value is {value}.')
            elif index%3==0:sentences.append(f'The {label} is {value}.')
            elif index%3==1:sentences.append(f'{label}: {value}.')
            else:sentences.append(f'{value} is recorded for {label}.')
    describe(target,[])
    correction=index%7==0
    if correction:
        # The corrected record identifier is unambiguous; never fabricate a label.
        sentences.insert(0,f'An obsolete identifier, {tag}-OLD, has been withdrawn. Use the current facts below.')
    prose=('\n\n' if split=='validation' else (' ' if index%2 else '\n')).join(sentences)
    schema=schema_for(target)
    schema['properties']['memo']=dict(type='string',description='Explicitly supplied memo; omit if absent.')
    return dict(prose=prose,schema=schema,target=target,layout=f'{split}-{family}-{index%3}',depth=depth,
                corrected=correction,reference=tag,items=n,family=family)

def leaf_strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from leaf_strings(v)
    elif isinstance(value,list):
        for v in value:yield from leaf_strings(v)

def code_example(value,index,split):
    # Fixed safe code templates and programmatic execution checks. No external code.
    def paths(v,path=()):
        if isinstance(v,dict):
            for key,item in v.items():yield from paths(item,path+(key,))
        elif not isinstance(v,list):yield path
    selected=list(paths(value));path=selected[index%len(selected)]
    if split=='train':
        expression='data'+''.join('['+repr(k)+']' for k in path)
        source='def extract(data):\n    return '+expression+'\n'
    else:
        source='def extract(data):\n    current = data\n    for key in '+repr(list(path))+':\n        current = current[key]\n    return current\n'
    assert all(not isinstance(n,(ast.Import,ast.ImportFrom,ast.Attribute)) for n in ast.walk(ast.parse(source)))
    namespace={};exec(compile(source,'<cpt151-controlled>','exec'),{'__builtins__':{}},namespace)
    result=namespace['extract'](copy.deepcopy(value));expected=value
    for key in path:expected=expected[key]
    assert type(result)==type(expected) and result==expected
    return 'A verified Python example for reading a nested JSON value.\nInput JSON:\n'+dumps(value)+'\nPython:\n'+source+'\nResult as JSON:\n'+dumps(result)

def document(bucket,split,index):
    w=world(bucket,split,index)
    assert all(v in w['prose'] for v in leaf_strings(w['target'])),'Source does not support a target string'
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(w['schema']);Draft202012Validator(w['schema']).validate(w['target'])
    if bucket=='json_structures':text=json.dumps(w['target'],ensure_ascii=False,indent=2 if index%2 else None)
    elif bucket=='schema_instances':text='JSON Schema:\n'+dumps(w['schema'])+'\nValid JSON instance:\n'+dumps(w['target'])
    elif bucket=='prose_schema_json':text='Source text:\n'+w['prose']+'\nJSON Schema:\n'+dumps(w['schema'])+'\nExtracted JSON:\n'+dumps(w['target'])
    elif bucket=='json_code':text=code_example(w['target'],index,split)
    else:raise ValueError(bucket)
    return dict(id=f'CPT151-{bucket}-{split}-{index:08d}',text=text,text_key=text_key(text),prose_key=text_key(w['prose']),
        provenance=dict(kind='procedural_verified_fact_world',split=split,index=index,bucket=bucket,layout=w['layout'],depth=w['depth'],items=w['items'],correction=w['corrected'],family=w['family'],
            limitations='Shared fact grammar; split-disjoint names/cities, field aliases, prose layouts and code templates. Not independent natural documents or exhaustive JSON Schema coverage.'))

def documents(bucket,split,tokenizer,guard):
    # Dedicated rows are reproducible from the pinned generator; oversize worlds
    # are rejected whole rather than losing a source/schema/answer suffix.
    index=0
    while True:
        row=document(bucket,split,index);index+=1
        if row['text_key']in guard or row['prose_key']in guard:raise ValueError('Held-out source overlap')
        if len(tokenizer.encode(row['text']))+1>2048:continue
        yield row
