"""Controlled natural-style business messages with a separate source parser.

This is procedural supervision, not a downloaded natural-document dataset.
Schemas are selected before values. Golds use sampled facts; the checker uses
regular expressions over the rendered source, not those sampled facts.
"""
import copy
import json
import random
import re
from .corpus import rank
from json_specialist.synthetic import FIRST, LAST, PRODUCTS, CITIES

DEFINITIONS={
 'contact':{
  'name':('string','The account contact\'s full name.'),
  'company':('string','The account company name.'),
  'email':(['string','null'],'The account email address, or null if no email address is supplied.'),
  'phone':(['string','null'],'The account telephone number, or null if no telephone number is supplied.')},
 'order':{
  'order_id':('string','The purchase order identifier.'),'customer':('string','The customer\'s full name.'),
  'product':('string','The ordered product name.'),'quantity':('integer','The ordered quantity as an integer.'),
  'urgent':('boolean','Whether expedited delivery was requested: yes means true, no means false.')},
 'invoice':{
  'invoice_id':('string','The invoice identifier.'),'vendor':('string','The company issuing the invoice.'),
  'total':('number','The stated invoice total as a number; do not calculate it.'),
  'paid':('boolean','Whether the invoice is paid: yes means true, no means false.'),
  'due_date':(['string','null'],'The explicitly stated due date, or null if not supplied.')},
 'shipment':{
  'shipment_id':('string','The shipment reference.'),'origin':('string','The city the shipment leaves from.'),
  'destination':('string','The city receiving the shipment.'),
  'delivery_date':(['string','null'],'The expected delivery date, or null if not supplied.'),
  'status':('string','The current shipment status, copied exactly.')},
 'booking':{
  'guest':('string','The name of the guest making the booking.'),'city':('string','The city of the booking.'),
  'date':('string','The booking date, copied exactly.'),'guests':('integer','The number of guests as an integer.'),
  'confirmed':('boolean','Whether the booking is confirmed: yes means true, no means false.')},
 'product':{
  'product_name':('string','The product name.'),'price':('number','The stated price as a number.'),
  'currency':('string','The explicitly stated currency code.'),
  'available':('boolean','Whether the product is available: yes means true, no means false.')},
 'ticket':{
  'ticket_id':('string','The support ticket identifier.'),'customer':('string','The customer\'s full name.'),
  'action':(['string','null'],'The explicitly requested action: refund, replacement, or null when none is supplied.'),
  'resolved':('boolean','Whether the support ticket is resolved: yes means true, no means false.')},
 'event':{
  'event_name':('string','The name of the event.'),'venue':(['string','null'],'The event venue, or null when no venue is supplied.'),
  'date':('string','The event date, copied exactly.'),
  'attendees':('array','All listed attendee names in source order. Use [] when no attendee names are supplied.')}
}

LABELS={k:k.replace('_',' ') for family in DEFINITIONS.values() for k in family}
ALIASES={'name':('contact','full_name','person'),'company':('business','organization','company_name'),
 'email':('email_address','email','mail'),'phone':('telephone','phone_number','phone'),
 'order_id':('purchase_id','order_number','reference'),'customer':('buyer','client','customer_name'),
 'product':('item','product_name','ordered_item'),'quantity':('units','count','quantity'),
 'urgent':('expedited','rush','urgent'),'invoice_id':('invoice_number','bill_id','invoice_id'),
 'vendor':('supplier','vendor_name','issuer'),'total':('amount','stated_total','invoice_total'),
 'paid':('is_paid','payment_received','paid'),'due_date':('due','payment_due','due_date'),
 'shipment_id':('tracking_id','shipment_reference','shipment_id'),'origin':('from_city','origin','departure_city'),
 'destination':('to_city','destination','arrival_city'),'delivery_date':('expected_date','delivery_date','arrival_date'),
 'status':('state','shipment_status','status'),'guest':('guest_name','name','guest'),
 'city':('booking_city','location','city'),'date':('date','scheduled_date','event_date'),
 'guests':('party_size','guest_count','guests'),'confirmed':('is_confirmed','confirmed','confirmation'),
 'product_name':('name','product','product_name'),'price':('price','unit_price','cost'),
 'currency':('currency','currency_code','money_unit'),'available':('in_stock','available','is_available'),
 'ticket_id':('case_number','ticket','ticket_id'),'action':('requested_action','request','action'),
 'resolved':('is_resolved','closed','resolved'),'event_name':('event','title','event_name'),
 'venue':('location','venue','place'),'attendees':('people','participants','attendees')}

# Each surface layout has a corresponding source-only parser. Values are
# Alternating prose layouts use quoted and unquoted values. Every fact occupies
# its own line, so source-side boundaries do not depend on model annotations.
RENDERERS=(
 lambda k,v:f'For this request, the {k} is {v}.',
 lambda k,v:f'Please record {v} as the {k}.',
 lambda k,v:f'The message gives {k}: {v}.',
 lambda k,v:f'Our colleague supplied {v} for the {k}.',
 lambda k,v:f'We have {k} = {v} on file.',
 lambda k,v:f'The submitted {k} reads {v}.',
 lambda k,v:f'According to the request, {k} was {v}.',
 lambda k,v:f'The sender listed {v} under {k}.')
TOKEN=r'("(?:[^"\\]|\\.)*"|\[(?:"(?:[^"\\]|\\.)*"(?:,\s*)?)*\]|null|true|false|-?\d+(?:\.\d+)?)'
PATTERNS=(
 r'For this request, the {k} is {v}\.',r'Please record {v} as the {k}\.',
 r'The message gives {k}: {v}\.',r'Our colleague supplied {v} for the {k}\.',
 r'We have {k} = {v} on file\.',r'The submitted {k} reads {v}\.',
 r'According to the request, {k} was {v}\.',r'The sender listed {v} under {k}\.')

def object_schema(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)

def schema_for(family,index):
    rng=random.Random(int(rank('business-schema:'+family+':'+str(index))[:16],16))
    keys=list(DEFINITIONS[family]);rng.shuffle(keys)
    count=rng.randint(1,len(keys));selected=keys[:count]
    properties={};mapping={}
    for key in selected:
        alias=rng.choice(ALIASES[key]);kind,description=DEFINITIONS[family][key]
        field=dict(type=kind,description=description)
        if kind=='array':field['items']=dict(type='string')
        properties[alias]=field;mapping[alias]=key
    nested=rng.randrange(4)==0
    if nested:return object_schema({'record':object_schema(properties)}),mapping,True
    return object_schema(properties),mapping,False

def facts_for(family,index):
    rng=random.Random(620000+index*31+list(DEFINITIONS).index(family))
    person=rng.choice(FIRST)+' '+rng.choice(LAST)
    company=rng.choice(LAST)+' '+rng.choice(('Industries','Consulting','Retail','Services','Logistics','Design'))
    date=f'2028-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}'
    email=f'{person.split()[0].lower()}{index}@example.org'
    missing=lambda v:None if rng.randrange(3)==0 else v
    boolean=lambda:bool(rng.randrange(2))
    product=rng.choice(PRODUCTS)
    return {
     'contact':dict(name=person,company=company,email=missing(email),phone=missing(f'+1-415-555-{index%10000:04d}')),
     'order':dict(order_id=f'PO-{index+700000}',customer=person,product=product,quantity=rng.randint(1,40),urgent=boolean()),
     'invoice':dict(invoice_id=f'INV-{index+700000}',vendor=company,total=rng.randint(100,99000)/100,paid=boolean(),due_date=missing(date)),
     'shipment':dict(shipment_id=f'SHIP-{index+700000}',origin=rng.choice(CITIES),destination=rng.choice(CITIES),delivery_date=missing(date),status=rng.choice(('pending','shipped','delivered'))),
     'booking':dict(guest=person,city=rng.choice(CITIES),date=date,guests=rng.randint(1,12),confirmed=boolean()),
     'product':dict(product_name=product,price=rng.randint(100,50000)/100,currency=rng.choice(('USD','EUR','GBP')),available=boolean()),
     'ticket':dict(ticket_id=f'CASE-{index+700000}',customer=person,action=rng.choice(('refund','replacement',None)),resolved=boolean()),
     'event':dict(event_name=company+' meeting',venue=missing(rng.choice(CITIES)),date=date,attendees=[rng.choice(FIRST)+' '+rng.choice(LAST) for _ in range(rng.randrange(4))])
    }[family]

def source(family,index,style,facts):
    lines=[]
    for key,value in facts.items():
        # Missing facts are omitted entirely; no answer comes from a template default.
        if value is None or value==[]:continue
        kind=DEFINITIONS[family][key][0]
        display=('yes' if value else 'no') if kind=='boolean' else value
        if style%2:
            rendered='; '.join(display) if isinstance(display,list) else str(display)
        else:rendered=json.dumps(display,ensure_ascii=False)
        lines.append(RENDERERS[style](LABELS[key],rendered))
    random.Random(index+555).shuffle(lines)
    headers=('Incoming email','Message for processing','Operations handoff','Customer request','Desk note','Administrative update','Team correspondence','Recorded request')
    lines.insert(0,headers[style]+':')
    lines.append('Internal memo code '+str(index+900000)+' is unrelated to the requested fields.')
    return '\n'.join(lines)

def parse(document,family,style):
    result={}
    for key,(kind,_) in DEFINITIONS[family].items():
        pattern='^'+PATTERNS[style].format(k=re.escape(LABELS[key]),v='(.+?)' if style%2 else TOKEN)+'$'
        matches=list(re.finditer(pattern,document,re.M))
        if len(matches)>1:raise ValueError('Ambiguous source role')
        if not matches:value=[] if kind=='array' else None
        elif not style%2:value=json.loads(matches[0].group(1))
        elif kind in ('integer','number'):value=json.loads(matches[0].group(1))
        elif kind=='array':value=matches[0].group(1).split('; ')
        else:value=matches[0].group(1)
        if kind=='boolean':
            if value not in ('yes','no'):raise ValueError('Missing boolean evidence')
            value=value=='yes'
        if kind in ('string','integer','number') and value is None:raise ValueError('Missing required source value')
        result[key]=value
    return result

def records(train_per_family=800,holdout_per_family=25):
    for fi,family in enumerate(DEFINITIONS):
        for split,styles,count in [('train',range(6),train_per_family),('development',[6],holdout_per_family),('test',[7],holdout_per_family)]:
            for i in range(count):
                index=fi*100000+({'train':0,'development':70000,'test':80000}[split])+i
                style=styles[i%len(styles)]
                schema,mapping,nested=schema_for(family,index)
                facts=facts_for(family,index);document=source(family,index,style,facts)
                target={alias:facts[key] for alias,key in mapping.items()}
                if nested:target={'record':target}
                yield dict(id=f'business-v1:{family}:{index}',source='business_transfer',split=split,document=document,schema=schema,target=target,
                    instruction='Extract only the fields requested by this schema. Use its descriptions to identify the source values. Include all required keys. For unstated values use null or [] as specified; never invent a value.',
                    source_group=f'business-v1:{family}:layout{style}',oracle=dict(family=family,index=index,style=style),
                    provenance=dict(kind='procedural_synthetic',attribution='OpenSML',license='Apache-2.0',quality='Reviewed finite renderer/parser grammar; every row checked by separate source-side parser, not natural-document certification.'))

def verify(row):
    o=row['oracle'];schema,mapping,nested=schema_for(o['family'],o['index'])
    if schema!=row['schema']:raise ValueError('Independent schema changed')
    facts=parse(row['document'],o['family'],o['style'])
    expected={alias:facts[key] for alias,key in mapping.items()}
    if nested:expected={'record':expected}
    if expected!=row['target']:raise ValueError('Source parser disagrees with target')
