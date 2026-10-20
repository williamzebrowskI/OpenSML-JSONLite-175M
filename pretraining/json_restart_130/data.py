"""Deterministic gold conversions and source-group-disjoint authored schema views.
All generated passages are disclosed controlled examples, not real user traffic.
"""
import copy, collections, hashlib, json, random, re
from jsonschema import Draft202012Validator
from .common import *
FAMILIES=['enumeration','binding','corrections','long_copy','missing_roles','types','nested']
ALIASES={
 'artist':['artist','performer','musician','requested_performer','music_creator','act'],
 'album':['album','record_title','album_name','recording','release_title','music_album'],
 'city':['city','town','destination_city','municipality','requested_city','place'],
 'item':['item','product','requested_item','parcel_contents','article','goods'],
 'person':['person','recipient','contact_name','client','named_contact','person_name'],
 'venue':['venue','event_location','screening_place','building_name','theater','site'],
 'movie':['movie','film_title','screening_title','film','movie_name','title'],
 'count':['quantity','requested_count','unit_count','item_total','number_of_items','amount'],
 'price':['price','unit_price','cost_per_item','item_price','price_each','cost'],
 'confirmed':['confirmed','is_confirmed','confirmation','approved','confirmation_flag','active'],
 'note':['note','reference_code','booking_code','comment','record_note','memo']}
DEFS=dict(artist='All musical performers requested; one separate value per performer, in source order. Exclude albums.',
 album='Explicit album titles; exclude musical performers.',city='City or town explicitly assigned to the requested recipient; do not infer country or state.',
 item='Item assigned to the specified person; exclude the other person’s item.',person='The named person receiving the requested item.',
 venue='Explicit screening venue name; exclude film titles.',movie='Explicit movie title; exclude venues.',
 count='Explicit quantity, serialized with the schema type.',price='Explicit price per item, serialized with the schema type.',
 confirmed='Explicit true or false confirmation flag; use the schema type.',note='Explicit reference code, when one is supplied.')
POOLS={
 'train':(['Alina','Basil','Celia','Damon','Elena','Felix','Greta','Hugo','Imani','Jules','Keira','Levi','Maren','Nico','Orla','Pavel'],
 ['Arden','Bennet','Corwin','Darrow','Ellis','Farrow','Granger','Hale','Ives','Jensen','Kellan','Larkin','Morrow','North','Oakley','Pierce'],
 ['Copper','Distant','Evening','Forest','Golden','Harbor','Ivory','Jade','Kindred','Lemon','Meadow','Noble','Opal','Prairie','Quartz','Rustic'],
 ['Atlas','Bridge','Canyon','Drift','Ember','Field','Grove','House','Island','Journey','Key','Lake','Mirror','Nest','Orbit','Path'],
 ['Austin','Boston','Chicago','Dallas','Edinburgh','Florence','Geneva','Hamburg','Izmir','Jakarta','Kyoto','Lisbon','Madrid','Nairobi','Oxford','Perth']),
 'development':(['Rhea','Soren','Talia','Ulric','Veda','Wren','Xenia','Yara'],['Quill','Rowan','Seaton','Thorne','Underwood','Voss','Wilder','Yates'],
 ['Scarlet','Tidal','Umber','Violet','Willow','Xanthic','Yellow','Zephyr'],['Quest','Reef','Summit','Tower','Union','Valley','Wharf','Zenith'],
 ['Zurich','Vienna','Utrecht','Turin','Seville','Riga','Quebec','Porto']),
 'reserved':(['Anouk','Brenna','Cassian','Daria','Emil','Flora','Gavin','Helena','Ilona','Jasper','Kora','Luca','Mika','Noemi','Oren','Petra'],
 ['Zeller','Alden','Briar','Cove','Dune','Elwood','Firth','Gable','Heath','Ingram','Jordane','Knoll','Lowe','Mercer','Nolan','Osborne'],
 ['Apricot','Boreal','Crimson','Dappled','Emerald','Frosted','Granite','Hazel','Indigo','Juniper','Lilac','Mossy','Ochre','Pearl','Russet','Saffron'],
 ['Beacon','Chorus','Delta','Estuary','Flame','Glade','Hearth','Inlet','Jewel','Lagoon','Marsh','Oasis','Petal','Ridge','Shore','Trail'],
 ['Reims','Siena','Toulouse','Uppsala','Verona','Wroclaw','Xanten','York','Aarhus','Brno','Cork','Delft','Erfurt','Faro','Ghent','Haarlem'])}
# Whole sentence frames held out, not random row splits of these same frames.
FRAMES={
 'train':{
 'enumeration':['Play music by {names}. The album is {album}.','For our drive, queue songs by {names} from {album}.','The requested performers are {names}; please start their album {album}.','I want to hear {names}, using the record {album}.','Our listening choices are {names}. Put on the album {album}.','Please select {names} for music, beginning with the album {album}.','Can you play the album {album}, featuring {names}?','My playlist should feature {names}; the album I chose is {album}.'],
 'binding':['Send {item} to {person} in {city}. Send {other_item} to {other} in {other_city}.','The order for {person} contains {item} and goes to {city}. {other} ordered {other_item} for {other_city}.','For {person}: {item}, destination {city}. For {other}: {other_item}, destination {other_city}.','{person} wants {item} delivered to {city}, whereas {other} needs {other_item} in {other_city}.'],
 'corrections':['Send {item} to {person} in {old}. Correction: the destination for {person} is {city}, replacing {old}.','{person} ordered {item} for {old}. Please change the destination to {city}.','The delivery for {person}, containing {item}, was scheduled for {old}. The corrected destination is {city}.','{item} is for {person}. Use {city} as the destination instead of the earlier {old}.'],
 'long_copy':['Find the movie {movie} at the venue {venue} at {time}.','The film is called {movie}; the screening venue is {venue}, and the time is {time}.','At {time}, look for {movie} showing at {venue}.','I want a screening of the film {movie} in {venue} around {time}.'],
 'missing_roles':['We want {cuisine} food in {city} at {time}.','Find a {cuisine} restaurant in {city} around {time}.','For dinner at {time}, look for {cuisine} cuisine in {city}.','Can we have {cuisine} food at {time} while in {city}?'],
 'types':['Send {item} to {person} in {city}. Quantity: {count}; unit price: {price}; confirmed: {confirmed}.','{person} requested {item} for {city}, with quantity {count}, price per item {price} and confirmation {confirmed}.','For {person}, the item is {item}, destination {city}, quantity {count}, each price {price}, confirmed {confirmed}.','The order lists {item} for {person} in {city}. Its quantity is {count}; its unit price is {price}; its confirmation is {confirmed}.'],
 'nested':['Deliver {item} to {person} in {city}, quantity {count}. Deliver {other_item} to {other} in {other_city}, quantity {other_count}.','Two orders: {person} needs {count} of {item} in {city}; {other} needs {other_count} of {other_item} in {other_city}.','{person} has an order for {count} of {item}, sent to {city}. {other} has {other_count} of {other_item}, sent to {other_city}.','The first recipient is {person}: {item}, {count}, destination {city}. The second is {other}: {other_item}, {other_count}, destination {other_city}.']},
 'development':{
 'enumeration':['The artists I would like to hear are {names}. My chosen album is {album}.','For tonight, my performer selection is {names}, and my album selection is {album}.'],
 'binding':['{person} is the recipient of {item}, bound for {city}; separately, {other} receives {other_item} in {other_city}.','Please keep these deliveries separate: {person}, {item}, {city}; {other}, {other_item}, {other_city}.'],
 'corrections':['The note initially puts {person} and {item} in {old}, but the latest amendment gives {city} as the destination.','For {person}’s {item}, the destination has been revised from {old} to {city}.'],
 'long_copy':['Which showing of {movie} can I attend in the venue {venue} at {time}?','My requested screening details are film {movie}, theater {venue}, time {time}.'],
 'missing_roles':['Our dining preference is {cuisine}, and we will be in {city} at {time}.','We are searching {city} for {cuisine} dining around {time}.'],
 'types':['Order details for {person}: {item}, delivered in {city}. It specifies {count} units at {price} per unit; confirmed is {confirmed}.','The recipient {person} in {city} needs {item}; the count entry reads {count}, price entry {price}, confirmation entry {confirmed}.'],
 'nested':['One delivery belongs to {person}: {count} of {item} in {city}. The other belongs to {other}: {other_count} of {other_item} in {other_city}.','Please record each shipment separately. {person} receives {item}, quantity {count}, at {city}; {other} receives {other_item}, quantity {other_count}, at {other_city}.']},
 'reserved':{
 'enumeration':['Please use the album {album} for the following performers: {names}.','I have selected these musicians: {names}; the record to use is {album}.'],
 'binding':['These are distinct consignments. {item} goes to {person} at {city}, while {other_item} goes to {other} at {other_city}.','The goods assigned to {person} are {item}, to be received in {city}. Those assigned to {other} are {other_item}, received in {other_city}.'],
 'corrections':['The current destination for {person}’s {item} is {city}; the superseded destination was {old}.','An update to {person}’s order of {item} replaces the original destination {old} with {city}.'],
 'long_copy':['Locate a showing at {time} for the title {movie}, in the building named {venue}.','The building is {venue}, the feature is {movie}, and the scheduled time is {time}.'],
 'missing_roles':['Our plans call for {cuisine} cuisine, with dinner in {city} at {time}.','At {time}, our group in {city} would like to eat {cuisine} food.'],
 'types':['For a delivery of {item}, {person} is in {city}. The explicitly stated values are quantity {count}, each price {price}, confirmation {confirmed}.','In {city}, {person} has requested {item}. Alongside it, the order records {count} for quantity, {price} for unit price and {confirmed} for confirmed.'],
 'nested':['Record the pair of orders: {count} of {item} to {person}, {city}; and {other_count} of {other_item} to {other}, {other_city}.','{person} in {city} will receive {count} of {item}. Independently, {other} in {other_city} will receive {other_count} of {other_item}.']}}

def row(i, document, schema, expected, family, component, group=None, **meta):
    return dict(id=i, group=group or i, document=document, schema=schema, expected=expected,
        source=family, variant=meta.pop('variant','authored'), family=family, component=component,
        pair_id=None, messages=[dict(role='user',content=prompt(document,schema)),dict(role='assistant',content=dumps(expected))], **meta)

def alias(field,rng):
    key=rng.choice(ALIASES[field]); return rng.choice(['','out_','result_'])+key

def present_prop(desc,typ):
    return dict(type=typ,description=desc) if typ!='array' else dict(type='array',items=dict(type='string'),description=desc)

def authored(split, family, i, *, bank=False):
    salt={'train':130,'development':230,'reserved':330}[split]
    rng=random.Random(salt*10**9+FAMILIES.index(family)*10**6+i)
    first,last,adjs,nouns,cities=POOLS[split]
    person=lambda:rng.choice(first)+' '+rng.choice(last)
    title=lambda:rng.choice(adjs)+' '+rng.choice(nouns)
    a=person();b=person()
    while b==a:b=person()
    city,other_city,old=rng.sample(cities,3)
    data=dict(person=a,other=b,city=city,other_city=other_city,old=old,item=title(),other_item=title(),album=title(),
        movie=title()+' and the '+title()+' Chronicle',venue='The '+title()+' Community Arts and Screening Center',
        time=rng.choice(['next Friday at 7:30 pm','on November the 21st at 8 pm','tomorrow evening at 6:45','on the first Saturday of December']),
        cuisine=rng.choice(['Mughlai','Ethiopian','Korean','Lebanese','Persian','Thai']),
        count=i%13,other_count=(i+3)%11,price=str(round(((i*17)%997)/4,2)),confirmed='true' if i%2 else 'false')
    names=[]
    while len(names)<i%6:
        v=person()
        if v not in names:names.append(v)
    data['names']=(', '.join(names[:-1])+' and '+names[-1]) if len(names)>1 else names[0] if names else ''
    frame=FRAMES[split][family][i%len(FRAMES[split][family])]
    document=frame.format(**data);props={};gold={};witness=[]
    if family=='enumeration' and not names:
        frame={'train':'Play the album {album}.','development':'The record I want is the album {album}.','reserved':'My album selection is {album}.'}[split]
        document=frame.format(**data)
    def add(f,values,desc=None,typ=None):
        key=alias(f,rng)
        # Randomize serialization independently of field identity; absent scalar is nullable.
        typ=typ or rng.choice(['array','nullable','string'] if len(values)==1 else ['array','nullable'] if not values else ['array'])
        if typ=='string' and not values:typ='nullable'
        props[key]=present_prop(desc or DEFS.get(f,f),['string','null'] if typ=='nullable' else typ)
        gold[key]=values if typ=='array' else values[0] if values else None
        for value in values:
            start=document.find(value);assert start>=0,(family,value,document)
            witness.append(dict(path=key,value=value,start=start,end=start+len(value)))
        return key
    if family=='enumeration':
        add('artist',names,typ='array');add('album',[data['album']]);add('note',[])
    elif family in ['binding','corrections']:
        add('item',[data['item']],f'Item assigned to {a}; exclude the other recipient’s item.')
        add('city',[city],f'Current destination city for {a}, using any explicit correction. Exclude old destinations and other recipients.')
        add('note',[])
    elif family=='long_copy':
        add('movie',[data['movie']]);add('venue',[data['venue']]);add('note',[data['time']],'Explicit screening time expression, copied in full.')
    elif family=='missing_roles':
        add('venue',[],'Proper name of the chosen restaurant. Cuisine and generic words such as restaurant are not its name.')
        add('city',[city]);add('note',[data['time']],'Explicit requested dining time expression.')
    elif family=='types':
        add('person',[a]);add('item',[data['item']]);add('city',[city])
        for f,typ,v in [('count','integer',data['count']),('price','number',float(data['price'])),('confirmed','boolean',data['confirmed']=='true')]:
            k=alias(f,rng);props[k]=present_prop(DEFS[f],typ);gold[k]=v
        add('note',[])
    else:
        # Per-record relationships are preserved, and nested field names also vary.
        keys={f:alias(f,rng) for f in ['person','item','city','count']}
        prop=obj({keys[f]:present_prop(DEFS[f],'integer' if f=='count' else 'string') for f in keys})
        root=rng.choice(['shipments','deliveries','orders','records'])
        props[root]=dict(type='array',items=prop,description='Each explicitly stated delivery, in source order. Keep each recipient’s item, city and quantity together.')
        gold[root]=[{keys['person']:a,keys['item']:data['item'],keys['city']:city,keys['count']:data['count']},
                    {keys['person']:b,keys['item']:data['other_item'],keys['city']:other_city,keys['count']:data['other_count']}]
    # Alternate ordering independently of target meaning.
    keys=list(props);rng.shuffle(keys);props={k:props[k] for k in keys};gold={k:gold[k] for k in keys}
    result=row(f'fresh130:{split}:{"bank:" if bank else ""}{family}:{i}',document,obj(props),gold,family,'authored',
        group=f'fresh130:{split}:{family}:{i}',split=split,template=frame,witness=witness,oracle=dict(data,artists=names),
        provenance=dict(dataset='OpenSML programmatically authored facts',quality='Gold from explicit fact oracle; not independently collected natural requests.'))
    return result

def qa_convert(r):
    if r['source']!='squad2':return None
    absent=r['answer_absent'];answers=[] if absent else [a for a in r['answers'] if a and a in r['document']]
    if not absent and not answers:return None
    rng=random.Random(int(hashlib.sha256(r['id'].encode()).hexdigest()[:12],16))
    key=rng.choice(['answer','requested_value','response_value','answer_text','result'])
    typ=rng.choice(['array','nullable'] if absent else ['array','nullable','string'])
    prop=present_prop('Exact source span answering this question: '+r['question'],['string','null'] if typ=='nullable' else typ)
    gold=answers[:1] if typ=='array' else answers[0] if answers else None
    return row('fresh130:qa:'+r['id'],r['document'],obj({key:prop}),{key:gold},'grounded_qa','grounded_qa',group=r['group'],
        upstream_id=r['id'],provenance=r['provenance'],split=r.get('split','upstream'),answer_absent=absent)

def snips_variant(r,i):
    rng=random.Random(13000000+i);props={};gold={}
    for key,v in r['expected'].items():
        new=rng.choice(['field_','value_','requested_','out_'])+key+rng.choice(['','_text','_value'])
        desc=r['schema']['properties'][key]['description']
        typ=rng.choice(['array','nullable','string'] if isinstance(v,list) and len(v)==1 else ['array'])
        vals=v if isinstance(v,list) else [] if v is None else [v]
        if typ!='array' and len(vals)!=1:typ='nullable'
        props[new]=present_prop(desc,['string','null'] if typ=='nullable' else typ)
        gold[new]=vals if typ=='array' else vals[0] if vals else None
    return row('fresh130:snips_variant:'+str(i),r['document'],obj(props),gold,r['source'],'schema_variation',group=r['group'],
        provenance=r['provenance'],upstream_id=r['id'],split='train')

def prepare():
    if DATA.exists():raise ValueError('Prepared data exists; will not overwrite it')
    DATA.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    from .runtime import Encoding
    enc=Encoding()
    def fits(r):
        try:enc.row(r,training=True);return True
        except ValueError:return False
    original=load(ORIGINAL/'train.jsonl');assert len(original)==15000
    # Exact original inputs/targets retained, except additional audit metadata.
    train=[dict(r,component='snips',family='snips',split='train') for r in original]
    if not all(fits(r) for r in train):raise ValueError('An original request exceeds the training budget')
    train += [snips_variant(r,i) for i,r in enumerate(random.Random(130).sample(original,3000))]
    qtrain=[z for r in load(QA/'train.jsonl') if (z:=qa_convert(r)) is not None and fits(z)]
    rng=random.Random(130);present=[r for r in qtrain if not r['answer_absent']];absent=[r for r in qtrain if r['answer_absent']]
    rng.shuffle(present);rng.shuffle(absent);selected=present[:3250]+absent[:750];assert len(selected)==4000
    train+=selected
    counts=dict(enumeration=1800,binding=1600,corrections=1600,long_copy=1600,missing_roles=1600,types=1200,nested=600)
    for f,n in counts.items():train.extend(authored('train',f,i) for i in range(n))
    assert len(train)==32000
    bank=[];seen={normalized(r['document']) for r in train}
    for f,n in counts.items():
        made=0;i=n
        while made<384:
            r=authored('train',f,i,bank=True);i+=1
            if normalized(r['document']) in seen:continue
            seen.add(normalized(r['document']));bank.append(r);made+=1
    used={r['id'] for r in selected};extra=[r for r in qtrain if r['id'] not in used]
    bank+=rng.sample(extra,384)
    olddev=load(ORIGINAL/'development.jsonl');groups=collections.defaultdict(list)
    for r in olddev:
        if r.get('pair_id'):groups[r['pair_id']].append(r)
    dev=[]
    for domain in sorted({r['source'] for r in olddev}):
        pairs=[g for g in groups.values() if len(g)==2 and g[0]['source']==domain]
        pairs.sort(key=lambda g:sha_text(g[0]['id']))
        if len(pairs)<8:raise ValueError('Insufficient original DEV pairs')
        dev.extend(dict(r,family='snips',component='snips',panel='compatibility') for g in pairs[:8] for r in g)
    assert len(dev)==112
    qa_dev=[z for r in load(QA/'development.jsonl') if (z:=qa_convert(r)) and fits(z)]
    for absent_value,n in [(False,24),(True,8)]:
        part=[r for r in qa_dev if r['answer_absent']==absent_value][:n];assert len(part)==n
        dev.extend(dict(r,panel='new') for r in part)
    for f in FAMILIES:dev.extend(dict(authored('development',f,i),panel='new') for i in range(16))
    reserved=[authored('reserved',f,i) for f in FAMILIES for i in range(32)]
    qtest=[z for r in load(QA/'test.jsonl') if (z:=qa_convert(r)) and fits(z)]
    for flag in [False,True]:reserved.extend([r for r in qtest if r['answer_absent']==flag][:16])
    assert len(dev)==256 and len(reserved)==256
    for r in train+bank+dev+reserved:
        Draft202012Validator.check_schema(r['schema']);Draft202012Validator(r['schema']).validate(r['expected'])
        if not fits(r):raise ValueError('Authored row too long: '+r['id'])
        if r['messages']!=[dict(role='user',content=prompt(r['document'],r['schema'])),dict(role='assistant',content=dumps(r['expected']))]:raise ValueError('Prompt/reference contamination')
    held=load(ORIGINAL/'test.jsonl')+load(ORIGINAL/'development.jsonl')+load(QA/'test.jsonl')+load(QA/'development.jsonl')
    held_docs={normalized(r['document']) for r in held}
    train_docs={normalized(r['document']) for r in train+bank}
    if train_docs&held_docs:raise ValueError('Original TRAIN/held-out normalized source collision')
    sets={n:{normalized(r['document']) for r in rows} for n,rows in [('train',train),('bank',bank),('dev',dev),('reserved',reserved)]}
    for a,b in [('train','dev'),('train','reserved'),('bank','dev'),('bank','reserved'),('dev','reserved')]:
        if sets[a]&sets[b]:raise ValueError('New split source collision: '+a+'/'+b)
    # Replay and bank can share upstream TRAIN articles, but not authored sources.
    authored_sets=[{r['document'] for r in rows if r['component']=='authored'} for rows in [train,bank,dev,reserved]]
    if any(authored_sets[i]&authored_sets[j] for i in range(4) for j in range(i)):raise ValueError('Authored source reused')
    for name,rows in [('train',train),('refinement_bank',bank),('development',dev),('reserved',reserved)]:write_rows(DATA/(name+'.jsonl'),rows)
    stats={}
    for n,rows in [('train',train),('refinement_bank',bank),('development',dev),('reserved',reserved)]:
        encoded=[enc.row(r,training=True) for r in rows]
        stats[n]=dict(count=len(rows),unique_sources=len({r['document'] for r in rows}),components=dict(collections.Counter(r['component'] for r in rows)),families=dict(collections.Counter(r['family'] for r in rows)),serialized_positions=sum(len(x)+1 for x,y in encoded),assistant_positions=sum(int((y!=-100).sum()) for x,y in encoded),max_sequence=max(len(x)+1 for x,y in encoded))
    inputs=[ORIGINAL/n for n in ['train.jsonl','development.jsonl','test.jsonl','manifest.json']]+[QA/n for n in ['train.jsonl','development.jsonl','test.jsonl','manifest.json']]
    inputs += [PUBLISHED/n for n in ['original1444.jsonl','replay128_300-original1444.json','playground60.jsonl','replay128_300-playground60.json']]
    atomic_json(DATA/'manifest.json',dict(plan=PLAN,refinement=REFINEMENT,base_sha256=BASE_SHA,release_sha256=RELEASE_SHA,
        files={p.name:sha(p) for p in DATA.glob('*.jsonl')},inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},stats=stats,
        splits='Original upstream article/source-group splits respected. Authored entity pools and full sentence frames split-disjoint. Bank is TRAIN-only; no test-derived targets. Schema/role task families deliberately overlap; not independent natural traffic.',
        licensing='SNIPS CC0-1.0; SQuAD CC-BY-SA-4.0; existing row provenance retained. No Dolly free-form answer is silently treated as a source span.',
        scope='Original 1444/playground are observed compatibility panels, not new blind benchmarks; reserved authored requests have shared task families and no predictions during setup. No Paraloq data used.'))
    print(json.dumps(stats,indent=2))

def sha_text(s):return hashlib.sha256(s.encode()).hexdigest()
if __name__=='__main__':prepare()
