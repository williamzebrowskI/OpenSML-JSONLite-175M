"""All original JSON training plus cardinality replay and auditable controlled prose.
New prose is programmatically authored, not externally collected user traffic.
Every retained value has a literal offset and role witness. Unmentioned roles stay empty.
"""
import collections,copy,hashlib,json,random,string
from .common import *
from sml_v1.common import atomic_json
from json_sampling_119.data import verify_case,normalized,source_audit

FIELDS={
'PlayMusic':['artist','track','album'],
'GetWeather':['city','country','state'],
'BookRestaurant':['restaurant_name','cuisine','party_size_number','timeRange','city'],
'AddToPlaylist':['entity_name','playlist','playlist_owner','music_item'],
'RateBook':['object_name','object_type','rating_value','rating_unit','object_select'],
'SearchCreativeWork':['object_name','object_type'],
'SearchScreeningEvent':['movie_name','location_name','object_location_type','timeRange','city']}
# No explicit "not provided" cues in new TRAIN. Empty answers must follow role scope.
TRAIN={
'PlayMusic':[
'Play {track} by {artist} from the album {album}.',
'I am looking for {artist}; queue the track {track}.',
'For our drive, put on the album {album}.',
'Can you play songs by {artist} this evening?',
'The song is {track}; its performer is {artist} and its album is {album}.',
'I would like to hear {track}, please.',
'Please choose music from {artist}, starting with their album {album}.',
'Put the album {album} on after the track {track}.'],
'GetWeather':[
'Compare the forecasts for {city}.',
'What will the weather be like in {city}, {country}?',
'I am traveling to {city} in the state of {state}; check the weather.',
'Please look up weather in the country {country}.',
'Check the temperatures in {city}; my country entry reads {country}.',
'Our trip takes us through {city}. Will it rain?',
'The destination state is {state}. I need its forecast.',
'How cold is it in {city}, in the country {country}, state {state}?'],
'BookRestaurant':[
'Reserve {restaurant_name} for {party_size_number} people at {timeRange} in {city}.',
'I want {cuisine} food at {timeRange}.',
'Book a table for {party_size_number} at {restaurant_name}.',
'Can we eat at {restaurant_name} in {city}?',
'Our group of {party_size_number} wants {cuisine} cuisine in {city}.',
'For {timeRange}, reserve the restaurant {restaurant_name}.',
'Find {cuisine} food in {city}.',
'{restaurant_name} is where our party of {party_size_number} wants to dine at {timeRange}.'],
'AddToPlaylist':[
'Add {entity_name} to the playlist {playlist}.',
'For {playlist_owner} playlist {playlist}, add {entity_name}.',
'Put the {music_item} {entity_name} on {playlist}.',
'I want {entity_name} added to {playlist_owner} playlist.',
'Please add {entity_name} to the list called {playlist}.',
'The destination playlist is {playlist}; the item to add is {entity_name}.',
'Could you add the {music_item} {entity_name} to {playlist_owner} playlist {playlist}?',
'Add {entity_name}, please.'],
'RateBook':[
'Give the {object_type} {object_name} {rating_value} {rating_unit}.',
'Rate {object_name} {rating_value}.',
'I give {object_name} {rating_value} {rating_unit}.',
'For the {object_select} {object_type}, my rating is {rating_value}.',
'I would rate the {object_type} {object_name} {rating_value}.',
'My rating for {object_name} is {rating_value} {rating_unit}.',
'Please rate the {object_select} {object_type} {rating_value} {rating_unit}.',
'The {object_type} called {object_name} deserves {rating_value} {rating_unit}.'],
'SearchCreativeWork':[
'Find the {object_type} {object_name}.',
'I am searching for {object_name}.',
'Please look up a {object_type} called {object_name}.',
'Can you locate {object_name} for me?',
'The title I need is {object_name}; it is a {object_type}.',
'Look for {object_name}, please.',
'Search for the {object_type} with the title {object_name}.',
'I would like to find {object_name}.'],
'SearchScreeningEvent':[
'Find {movie_name} at the {object_location_type} {location_name} in {city} at {timeRange}.',
'When is {movie_name} showing?',
'What is playing at the {object_location_type} {location_name}?',
'Look up screenings of {movie_name} in {city}.',
'Check the schedule at {location_name} for {timeRange}.',
'I want to see {movie_name} at {timeRange}.',
'The {object_location_type} is {location_name}; the film is {movie_name}.',
'Can I watch {movie_name} at {location_name} in {city}?']}
DEV={
'PlayMusic':['The performers on my listening list are {artist}; I want their song {track}.','Next up should be {album}, the album I have in mind.','Could you play {track}?','My choice of performers is {artist}. Their record {album} is what I want.'],
'GetWeather':['I need forecasts for these places: {city}.','Before leaving for {country}, tell me the conditions in {city}.','Check conditions for the state {state}.','Our destination cities are {city}; the country is {country}.'],
'BookRestaurant':['Could {restaurant_name} accommodate {party_size_number} diners around {timeRange}?','We are after {cuisine} cuisine in {city}.','The booking should be at {restaurant_name}, in {city}.','There are {party_size_number} of us, and we want to eat at {timeRange}.'],
'AddToPlaylist':['The list named {playlist} should get {entity_name}.','Put {entity_name} onto {playlist_owner} playlist.','Please append the {music_item} {entity_name} to {playlist}.','For the playlist {playlist}, the item I chose is {entity_name}.'],
'RateBook':['For {object_name}, I would award {rating_value} {rating_unit}.','The {object_select} {object_type} gets {rating_value} from me.','My score for the {object_type} {object_name} is {rating_value}.','Award {object_name} {rating_value} {rating_unit}, please.'],
'SearchCreativeWork':['Can you get me the {object_type} titled {object_name}?','What I am trying to locate is {object_name}.','Please search for {object_name}.','The work is a {object_type}; its name is {object_name}.'],
'SearchScreeningEvent':['Which showtimes are available for {movie_name} in {city}?','I would like the timetable for the {object_location_type} {location_name}.','The movie I want is {movie_name}, showing at {location_name} around {timeRange}.','Look at {location_name} in {city}, please.']}
# Reserved wording; never predicted during preparation or checkpoint selection.
TEST={
'PlayMusic':['For music tonight, my selected artists are {artist}.','Start with the recording {album}; that is an album title.','The tune to play is {track}, performed by {artist}.','Would you queue the track {track} from album {album}?'],
'GetWeather':['Could you compare temperatures across {city}?','Tell me about rain in {city}; this is a trip to {country}.','The country whose forecast I need is {country}.','For the state {state}, check conditions in {city}.'],
'BookRestaurant':['We would like {restaurant_name} for a group of {party_size_number}, at {timeRange}.','Please find dining options with {cuisine} food.','At {timeRange}, look for tables in {city}.','My restaurant choice is {restaurant_name}; the city is {city}.'],
'AddToPlaylist':['I chose {entity_name} for the playlist called {playlist}.','Append {entity_name} to {playlist_owner} playlist, please.','The {music_item} I want on {playlist} is {entity_name}.','Place {entity_name} in the list {playlist}.'],
'RateBook':['I am awarding the {object_type} {object_name} {rating_value} {rating_unit}.','For the {object_select} {object_type}, record {rating_value} {rating_unit}.','My verdict on {object_name}: {rating_value}.','Give {object_name} a rating of {rating_value} {rating_unit}.'],
'SearchCreativeWork':['I need help finding a {object_type} named {object_name}.','The work I am looking for is {object_name}.','Find me {object_name}, please.','Locate the {object_type} with name {object_name}.'],
'SearchScreeningEvent':['I need a showing of {movie_name} at {timeRange}.','Could you check the {object_location_type} {location_name} for listings?','Are there screenings of {movie_name} at {location_name} in {city}?','What can I see at {location_name} around {timeRange}?']}

def render(template,values):
 text='';spans=[]
 for literal,field,_,_ in string.Formatter().parse(template):
  text+=literal
  if field is None:continue
  items=values[field]
  for j,value in enumerate(items):
   if j:text+=(' and ' if j==len(items)-1 else ', ')
   start=len(text);text+=value;spans.append(dict(field=field,value=value,start=start,end=len(text)))
 return text,spans

def pools(original):
 result=collections.defaultdict(set)
 for r in original:
  for span in r['spans']:result[span['field']].add(span['value'])
 return {f:sorted(v) for f,v in result.items()}

def values(bank,seed,split):
 rng=random.Random(seed+{'train':0,'development':1000000,'reserved':2000000}[split]);result={}
 for f in sorted({f for fs in FIELDS.values() for f in fs}):
  pool=bank[f]
  # Use original TRAIN field values, with longer multiword values frequently sampled.
  long=[v for v in pool if len(v)>=18 and ' ' in v]
  source=long if long and seed%2 else pool
  n=1+(seed%5) if f in ('artist','city') else 1
  result[f]=rng.sample(source,min(n,len(source)))
 return result

def make(domain,template,i,split,variant,defs,bank):
 text,spans=render(template,values(bank,i,split));keys=FIELDS[domain][:]
 if i%4==0:keys=keys[:-1]
 if i%2:keys.reverse()
 mapping={f:('out_'+f if variant=='alias' else f) for f in keys};props={};gold={}
 for f in keys:
  matches=[s['value'] for s in spans if s['field']==f];scalar=variant=='nullable' and len(matches)<=1
  desc=defs[f]
  props[mapping[f]]=dict(type=['string','null'],description=desc) if scalar else dict(type='array',items=dict(type='string'),description=desc)
  gold[mapping[f]]=(matches[0] if matches else None) if scalar else matches
 schema=dict(type='object',properties=props,required=list(props),additionalProperties=False)
 row=dict(id=f'joint128:{split}:{domain}:{i}:{variant}',document=text,schema=schema,expected=gold,messages=[dict(role='user',content=prompt(text,schema)),dict(role='assistant',content=dumps(gold))],spans=[s for s in spans if s['field'] in keys],key_mapping=mapping,pair_id=None,source=domain,variant=variant,category='controlled_prose',template=template,authored=True,training_component='new_prose',scope='Programmatic prose with original TRAIN entity values; separate wording/source texts, not a blind real-user benchmark.')
 verify_case(row);return row

def write(name,rows):
 (DATA/(name+'.jsonl')).write_text(''.join(dumps(r)+'\n' for r in rows))

def build():
 if DATA.exists():raise ValueError('Already prepared; refusing overwrite')
 DATA.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
 original=load(OLD/'train.jsonl');card=[r for r in load(CARD/'train.jsonl') if r['source']=='cardinality'];bank=pools(original);defs={}
 for r in original:
  for f,k in r['key_mapping'].items():defs[f]=r['schema']['properties'][k]['description']
 # Scope definitions emphasize literal roles, never field answers.
 defs.update(artist='Musical performers explicitly requested. Exclude titles; each performer is a separate item.',track='Song or track titles explicitly mentioned. Exclude album names and performers.',album='Album titles explicitly mentioned. Exclude tracks and performer names.',city='City or town names explicitly mentioned. Never infer country or state.',country='Countries explicitly named. Do not infer from a city.',state='Explicit state or region names or abbreviations. Never infer from a city.',location_name='Explicit screening venue name. Exclude movie titles.',movie_name='Explicit movie titles. Exclude venues.',playlist='Name of the destination playlist. Exclude the added item.',entity_name='Named item explicitly added to a playlist. Exclude the playlist name.')
 new=[];dev=[];reserved=[]
 for split,templates,repeats,dest in [('train',TRAIN,32,new),('development',DEV,4,dev),('reserved',TEST,4,reserved)]:
  for domain,patterns in templates.items():
   for t,template in enumerate(patterns):
    for n in range(repeats):
     i=t*repeats+n;variants=['canonical','alias','nullable'] if split=='train' else [['canonical','alias','nullable'][i%3]]
     for v in variants:dest.append(make(domain,template,i,split,v,defs,bank))
 train=[]
 for group,label in [(original,'original_json'),(card,'cardinality_replay'),(new,'new_prose')]:
  for r in group:
   r=copy.deepcopy(r);r['training_component']=label;train.append(r)
 for r in train+dev+reserved:verify_case(r)
 # The original annotation audit checks every original view against pinned upstream SNIPS.
 witness=[dict(r,source_bank=str((OLD/'train.jsonl').relative_to(ROOT)),source_row_id=r['id']) for r in original]
 atomic_json(REPORT/'source-validation.json',source_audit(witness))
 old=load(LEGACY/'development.jsonl');chosen=[];groups=collections.defaultdict(list)
 for r in old:
  if r.get('pair_id'):groups[r['pair_id']].append(r)
 rank=lambda r:hashlib.sha256(('128compact:'+r['id']).encode()).hexdigest()
 for domain in sorted(FIELDS):
  pairs=sorted([g for g in groups.values() if len(g)==2 and g[0]['source']==domain],key=lambda g:rank(g[0]))
  for g in pairs[:6]:chosen.extend(g)
 for source,n in [('roles',8),('observed_lists',12),('single_lists',4),('cardinality',8),('mixsnips',8),('playground',4)]:
  if source=='single_lists':
   for g in [g for g in groups.values() if g[0]['source']==source]:chosen.extend(g)
  else:
   candidates=[r for r in old if r['source']==source]
   if source=='observed_lists':candidates=[r for r in candidates if r['cardinality'] in (4,5)]
   chosen.extend(sorted(candidates,key=rank)[:n])
 if len(chosen)!=128:raise ValueError('Compact control coverage')
 chosen=[dict(r,eval_panel='legacy') for r in chosen]
 dev=[dict(r,eval_panel='new_prose') for r in dev];reserved=[dict(r,eval_panel='reserved_prose') for r in reserved]
 held=load(OLD/'test.jsonl')+old+load(LEGACY/'test.jsonl')
 normalized_sets={n:{normalized(r['document']) for r in rs} for n,rs in [('train',train),('development',dev),('reserved',reserved),('observed',held)]}
 for a,b in [('train','development'),('train','reserved'),('train','observed'),('development','reserved'),('development','observed'),('reserved','observed')]:
  if normalized_sets[a]&normalized_sets[b]:raise ValueError('Source overlap '+a+'/'+b)
 patterns=[{t for ps in d.values() for t in ps} for d in [TRAIN,DEV,TEST]]
 if any(patterns[i]&patterns[j] for i in range(3) for j in range(i)):raise ValueError('Template leakage')
 for n,rs in [('train',train),('development',chosen+dev),('legacy_full',old),('original_test',load(OLD/'test.jsonl')),('reserved',reserved)]:write(n,rs)
 inputs=[OLD/'train.jsonl',OLD/'test.jsonl',CARD/'train.jsonl',LEGACY/'development.jsonl',LEGACY/'test.jsonl',LEGACY.parent/'parent-development.json',LEGACY.parent/'parent-development.identity.json',PRIOR/'retired-experiment/pretraining/data/json_diagnostics_127_20261016/pilot_legacy.jsonl']
 summary=dict(records=len(train),original_json=len(original),cardinality_replay=len(card),new_prose=len(new),new_sources=len({r['document'] for r in new}),domains=dict(collections.Counter(r['source'] for r in train)),components=dict(collections.Counter(r['training_component'] for r in train)),variants=dict(collections.Counter(r['variant'] for r in train)),development=len(chosen+dev),legacy_controls=len(chosen),new_development=len(dev),full_legacy=len(old),original_test=1444,reserved=len(reserved),steps=(len(train)+63)//64)
 atomic_json(DATA/'manifest.json',dict(plan=PLAN,parent_sha256=PARENT_SHA,summary=summary,inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},files={p.name:sha(p) for p in DATA.glob('*.jsonl')},scope='Complete original 15k and cardinality data plus controlled authored prose. Original TRAIN values are deliberately reused. Wording and normalized source texts are split-disjoint; entity values are not held out. All older controls have been observed. Reserved prose is unpredicted but shares construction families. No promise of perfect annotations or universal no-regression.'))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':build()
