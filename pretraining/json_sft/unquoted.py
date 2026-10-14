"""Unquoted finite relationship prose, with source-only reconstruction.

This broadens procedural surface forms; it is not a natural-document dataset.
"""
import random
import re
from . import narrative

TASKS=('employment','purchase','attendees','calendar')
TEXT=r'(.+?)'

def source(task,index,style,f):
    if task=='employment':
        forms=(f'{f["name"]} works at {f["company"]} as {f["job_title"]}.',f'At {f["company"]}, {f["name"]} holds the position {f["job_title"]}.',f'The employee {f["name"]} has role {f["job_title"]} with {f["company"]}.',f'{f["company"]} employs {f["name"]} in the role {f["job_title"]}.',f'{f["name"]} serves as {f["job_title"]} for {f["company"]}.',f'The position {f["job_title"]} at {f["company"]} is held by {f["name"]}.')
        text=forms[style]+(' Email: '+f['email']+'.' if f['email'] is not None else '')
    elif task=='purchase':
        forms=(f'{f["customer"]} ordered {f["quantity"]} units of {f["product"]} for delivery to {f["destination"]}.',f'Send {f["quantity"]} units of {f["product"]} to {f["destination"]} for buyer {f["customer"]}.',f'The buyer {f["customer"]} needs {f["product"]}: {f["quantity"]} units, destination {f["destination"]}.',f'Purchase by {f["customer"]}: {f["quantity"]} of {f["product"]}; ship to {f["destination"]}.',f'Deliver {f["product"]} to {f["destination"]}: {f["quantity"]} units purchased by {f["customer"]}.',f'To {f["destination"]}, ship {f["quantity"]} units of {f["product"]} ordered by {f["customer"]}.')
        text=forms[style]+' Order reference '+f['order_id']+'.'
    elif task=='attendees':
        forms=('Participant {name} represents {company}.','{name} is attending for {company}.','The attendee {name} belongs to {company}.','From {company}, we have {name}.','{company} will be represented by {name}.','Attending: {name}, representing {company}.')
        text=' '.join(forms[style].format(**v) for v in f['people']) or 'No participants were listed.'
    else:
        headers=('Event','Meeting','Session','Calendar entry','Scheduled activity','Agenda item')
        text=headers[style]+': '+f['event']+'.'
        if f['date'] is not None:text+=(' Originally dated 2028-01-01; corrected date ' if index%2 else ' Date ')+f['date']+'.'
        if f['time'] is not None:text+=' Starts at '+f['time']+'.'
        if f['location'] is not None:text+=' Venue '+f['location']+'.'
    return text+' Processing code '+str(index+10000000)+' is unrelated.'

def parse(document,task,style):
    def match(pat):
        m=re.search(pat,document)
        if not m:raise ValueError('Unquoted grammar mismatch')
        return list(m.groups())
    if task=='employment':
        pats=(r'^'+TEXT+r' works at '+TEXT+r' as '+TEXT+r'\.',r'^At '+TEXT+r', '+TEXT+r' holds the position '+TEXT+r'\.',r'^The employee '+TEXT+r' has role '+TEXT+r' with '+TEXT+r'\.',r'^'+TEXT+r' employs '+TEXT+r' in the role '+TEXT+r'\.',r'^'+TEXT+r' serves as '+TEXT+r' for '+TEXT+r'\.',r'^The position '+TEXT+r' at '+TEXT+r' is held by '+TEXT+r'\.')
        vals=match(pats[style]);order=((0,1,2),(1,0,2),(0,2,1),(1,0,2),(0,2,1),(2,1,0))[style]
        email=re.search(r' Email: (\S+)\.',document)
        return dict(name=vals[order[0]],company=vals[order[1]],job_title=vals[order[2]],email=email.group(1) if email else None)
    if task=='purchase':
        pats=(r'^'+TEXT+r' ordered (\d+) units of '+TEXT+r' for delivery to '+TEXT+r'\.',r'^Send (\d+) units of '+TEXT+r' to '+TEXT+r' for buyer '+TEXT+r'\.',r'^The buyer '+TEXT+r' needs '+TEXT+r': (\d+) units, destination '+TEXT+r'\.',r'^Purchase by '+TEXT+r': (\d+) of '+TEXT+r'; ship to '+TEXT+r'\.',r'^Deliver '+TEXT+r' to '+TEXT+r': (\d+) units purchased by '+TEXT+r'\.',r'^To '+TEXT+r', ship (\d+) units of '+TEXT+r' ordered by '+TEXT+r'\.')
        vals=match(pats[style]);order=((0,2,1,3),(3,1,0,2),(0,1,2,3),(0,2,1,3),(3,0,2,1),(3,2,1,0))[style]
        return dict(customer=vals[order[0]],product=vals[order[1]],quantity=int(vals[order[2]]),destination=vals[order[3]],order_id=match(r' Order reference (\d+)\.')[0])
    if task=='attendees':
        pats=(r'Participant '+TEXT+r' represents '+TEXT+r'\.',TEXT+r' is attending for '+TEXT+r'\.',r'The attendee '+TEXT+r' belongs to '+TEXT+r'\.',r'From '+TEXT+r', we have '+TEXT+r'\.',TEXT+r' will be represented by '+TEXT+r'\.',r'Attending: '+TEXT+r', representing '+TEXT+r'\.')
        result=[]
        for m in re.finditer(pats[style],document):
            a,b=m.groups()
            # Sentence-start whitespace is not part of a person/company name.
            a=a.lstrip();b=b.lstrip()
            result.append(dict(name=b,company=a) if style in (3,4) else dict(name=a,company=b))
        if not result and not document.startswith('No participants were listed.'):raise ValueError('Missing attendees')
        return dict(people=result)
    event=match(r'^[^:]+: '+TEXT+r'\.')[0]
    date=re.search(r'(?:corrected date| Date) (\d{4}-\d{2}-\d{2})\.',document);time=re.search(r' Starts at (\d{2}:\d{2})\.',document);venue=re.search(r' Venue '+TEXT+r'\.',document)
    return dict(event=event,date=date.group(1) if date else None,time=time.group(1) if time else None,location=venue.group(1) if venue else None)

def records(per_task=1500,holdout=50):
    for ti,task in enumerate(TASKS):
        for split,count in [('train',per_task),('development',holdout),('test',holdout)]:
            for i in range(count):
                index=900000+ti*100000+{'train':0,'development':70000,'test':80000}[split]+i;style=i%4 if split=='train' else 4 if split=='development' else 5
                schema,mapping,nested=narrative.schema_for(task,index);facts=narrative.facts(task,index);target={key:facts[role] for key,role in mapping.items()}
                yield dict(id=f'unquoted-v1:{task}:{index}',source='unquoted_transfer',source_group=f'unquoted:{task}:{style}',split=split,document=source(task,index,style,facts),schema=schema,target={'data':target} if nested else target,
                    instruction='Extract the requested fields from the supplied prose. Preserve exact source text and associations. Use null or [] only as allowed when a field is absent.',
                    oracle=dict(task=task,index=index,style=style),provenance=dict(kind='procedural_synthetic',license='Apache-2.0',attribution='OpenSML',quality='Unquoted finite relationship layouts; independent source reconstruction, not natural-document certification.'))

def verify(row):
    o=row['oracle'];schema,mapping,nested=narrative.schema_for(o['task'],o['index'])
    if schema!=row['schema']:raise ValueError('Unquoted schema changed')
    facts=parse(row['document'],o['task'],o['style']);target={key:facts[role] for key,role in mapping.items()}
    if ({'data':target} if nested else target)!=row['target']:raise ValueError('Unquoted source roles disagree')
