"""Fictional, deterministic fact-oracle examples; never sourced from eval outputs."""
import random
from .data import make,h

# Each domain has two roles, source-grounded answer frames and explicit field names.
DOMAINS=[
 ('work assignment','project','office',['Orchid','Cedar','Falcon','Quartz','Harbor','Willow'],['Bristol','York','Dover','Bath','Leeds','Oxford']),
 ('workshop booking','course','venue',['pottery','weaving','drawing','robotics','woodwork','printing'],['Maple Hall','Birch Hall','Oak Hall','Elm Hall','Pine Hall','Ash Hall']),
 ('shipment','item','destination',['lamps','desks','chairs','shelves','mirrors','cabinets'],['Bristol','York','Dover','Bath','Leeds','Oxford']),
 ('research assignment','topic','laboratory',['soil','rainfall','fungi','algae','pollination','erosion'],['Cedar Lab','Birch Lab','Oak Lab','Elm Lab','Pine Lab','Ash Lab']),
 ('travel booking','service','destination',['Ferry Amber','Train Birch','Coach Cedar','Ferry Delta','Train Elm','Coach Flint'],['Bristol','York','Dover','Bath','Leeds','Oxford']),
 ('exhibition entry','artwork','gallery',['Blue Horizon','Quiet Harbor','Winter Path','Silver Lake','Red Garden','Golden Field'],['Cedar Gallery','Birch Gallery','Oak Gallery','Elm Gallery','Pine Gallery','Ash Gallery']),
 ('volunteer assignment','task','site',['sorting books','packing meals','planting trees','repairing bikes','folding clothes','painting benches'],['Cedar Center','Birch Center','Oak Center','Elm Center','Pine Center','Ash Center']),
 ('library request','book','pickup branch',['River Birds','Mountain Trails','Desert Plants','Ocean Life','Forest Paths','Island Maps'],['Cedar Branch','Birch Branch','Oak Branch','Elm Branch','Pine Branch','Ash Branch']),
]
FIRST=['Amina','Boris','Celia','Dario','Elena','Farid','Greta','Hugo','Ines','Jonah','Keiko','Luca','Marta','Nabil','Olga','Pavel','Rosa','Soren','Talia','Victor','Willa','Yara','Zain','Nora','Leila','Remy','Ada','Nico','Vera','Milo']
LAST=['Avery','Benton','Carter','Denton','Ellis','Foster','Grant','Hayes','Irwin','Jensen','Keller','Lane','Morris','Nash','Olsen','Price','Quinn','Reed','Stone','Turner','Underwood','Vale','West','Young','Zeller','Baird','Cole','Dunn','Eaton','Finch']

def generate(split,category,count):
 rng=random.Random(int(h('qa065:'+split+category)[:16],16))
 # Different sentence/question frame IDs across train, dev and test.
 styles=[0,1,2] if split=='train' else [3] if split=='development' else [4]
 domain_offset={'calibration':0,'multipart':2,'corrections':4}[category]
 for i in range(count//2 if category in ('calibration','corrections') else count):
  domain=DOMAINS[(i+domain_offset)%len(DOMAINS)];label,f1,f2,v1s,v2s=domain
  a,b=rng.sample([x+' '+y for x in FIRST for y in LAST],2)
  x,z=rng.sample(v1s,2);y,w,t=rng.sample(v2s,3);style=styles[i%len(styles)]
  # Ledger is intentionally fictional. Unique reference makes records traceable,
  # while entity/field values are independently randomized (no ID-to-answer rule).
  reference=f'{split[:1].upper()}-{category[:1].upper()}-{i:05d}'
  def statement(person,first,second):
   frames=[f"The {label} for {person} lists {f1}: {first}; {f2}: {second}.",
    f"For {person}, the recorded {f1} is {first}, and the {f2} is {second}.",
    f"{person}'s {label} specifies {first} as the {f1} and {second} as the {f2}.",
    f"Looking up {person} in the {label} register gives {first} under {f1} and {second} under {f2}.",
    f"The entry filed for {person} has two details: its {f1} is {first}, while its {f2} is {second}."]
   return frames[style]
  header=f'Fictional {label} notes, reference {reference}. '
  distractor=statement(b,z,w)
  full=statement(a,x,y)
  # Randomize entity order: no always-first-record solution.
  def paragraph(target):
   parts=[target,distractor];rng.shuffle(parts)
   return header+' '.join(parts)+' These notes describe separate entries; do not transfer details between people.'
  group=f'controlled:{split}:{category}:{i}'
  def row(suffix,doc,q,answer,values,rejected,**extra):
   r=make('qa065:'+group+':'+suffix,category,doc,q,[answer],group,**extra)
   r.update(bucket=category,origin='authored_fact_oracle',frame_id=f'{category}:{style}',required_values=values,forbidden_values=rejected,oracle=dict(person=a,other_person=b,first_field=f1,second_field=f2,first_value=x,second_value=y,other_first=z,other_second=w,revised_second=t,domain=label),provenance=dict(dataset='OpenSML authored fictional facts',quality='Symbolically checked; template variants are not independent human-written passages'))
   return r
  if category=='calibration':
   qs=[f'What is the {f2} for {a}?',f'Which {f2} is recorded for {a}?',f'From these notes, give {a}\'s {f2}.',f'I need the {f2} belonging to {a}. What do the notes say?',f'Check {a}\'s entry: can you identify the {f2}?']
   answer=f"The {f2} for {a} is {y}."
   missing=[f"The {label} for {a} lists the {f1} as {x}. No {f2} is recorded for this entry.",f"For {a}, only the {f1}, {x}, was recorded; the {f2} field was left blank.",f"{a}'s {label} mentions {x} as the {f1}, without specifying a {f2}.",f"The register contains {a}'s {f1}, {x}, but supplies no information about that person's {f2}.",f"For the entry filed for {a}, {x} appears under {f1}; nothing was entered under {f2}."][style]
   yes=row('present',paragraph(full),qs[style],answer,[y],[w],pair_id=group)
   no=row('missing',paragraph(missing),qs[style],'Not stated.',[],[w],pair_id=group)
   yes['oracle']['answer_available']=True;no['oracle']['answer_available']=False
   yield yes;yield no
  elif category=='multipart':
   qs=[f'What are the {f1} and {f2} for {a}? Answer both parts in a complete sentence.',f'Tell me both {a}\'s {f1} and {f2}, using the notes.',f'For {a}, identify the {f1} together with the {f2}.',f'I need two details for {a}: the {f1} and the {f2}. Please include both.',f'Look up {a} and report both fields, {f1} and {f2}, in your answer.']
   answer=f"The {f1} for {a} is {x}, and the {f2} is {y}."
   yield row('both',paragraph(full),qs[style],answer,[x,y],[z,w])
  else:
   # Ask both historical and current values; always selecting the last value fails.
   corrections=[f"A correction changes {a}'s {f2} from {y} to {t}. The {f1} remains {x}.",f"The original {f2}, {y}, for {a} was replaced with {t}. No change was made to the {f1}.",f"The initial notes gave {y} as {a}'s {f2}. An authorized revision now specifies {t} instead.",f"An amendment to {a}'s entry says to cross out {y} in the {f2} field and use {t}. Its other field is unchanged.",f"For {a}, a later notice supersedes the earlier {f2} of {y}: the effective value is {t}. Keep the original {f1}."][style]
   doc=paragraph(full)+' '+corrections
   cq=[f'What is the current {f2} for {a}?',f'After the correction, which {f2} applies to {a}?',f'Give the revised {f2} for {a}.',f'Which {f2} should be used for {a} following the amendment?',f'What {f2} is effective for {a} after the later notice?'][style]
   oq=[f'What was the original {f2} for {a}, before the correction?',f'Before the revision, which {f2} was recorded for {a}?',f'Give the earlier {f2} for {a}, not the revised value.',f'Which {f2} appeared for {a} before the amendment?',f'What {f2} did {a}\'s entry give before the later notice superseded it?'][style]
   current=row('current',doc,cq,f"The current {f2} for {a} is {t}.",[t],[y,w],pair_id=group)
   original=row('original',doc,oq,f"The original {f2} for {a} was {y}.",[y],[t,w],pair_id=group)
   current['oracle']['version']='current';original['oracle']['version']='original'
   yield current;yield original

def validate(r):
 o=r['oracle'];a=r['answers'][0];doc=r['document'];person=o['person'];field=o['second_field']
 for v in r['required_values']:assert v in doc and v in a
 for v in r['forbidden_values']:assert v not in a
 if r['source']=='calibration':
  expected=f"The {field} for {person} is {o['second_value']}." if o['answer_available'] else 'Not stated.'
 elif r['source']=='multipart':expected=f"The {o['first_field']} for {person} is {o['first_value']}, and the {field} is {o['second_value']}."
 else:
  expected=f"The current {field} for {person} is {o['revised_second']}." if o['version']=='current' else f"The original {field} for {person} was {o['second_value']}."
 assert a==expected
 assert r['answer_absent']==(a=='Not stated.')

def validate_passage(r):
 """Parse facts from rendered text, independently of target rendering."""
 import re
 o=r['oracle'];doc=r['document'];person=re.escape(o['person']);f1=re.escape(o['first_field']);f2=re.escape(o['second_field']);label=re.escape(o['domain']);style=int(r['frame_id'].split(':')[-1])
 patterns=[rf'The {label} for {person} lists {f1}: (.*?); {f2}: (.*?)\.',rf'For {person}, the recorded {f1} is (.*?), and the {f2} is (.*?)\.',rf"{person}'s {label} specifies (.*?) as the {f1} and (.*?) as the {f2}\.",rf'Looking up {person} in the {label} register gives (.*?) under {f1} and (.*?) under {f2}\.',rf'The entry filed for {person} has two details: its {f1} is (.*?), while its {f2} is (.*?)\.']
 found=re.search(patterns[style],doc)
 if r['answer_absent']:
  assert found is None
  missing=[rf'The {label} for {person} lists the {f1} as (.*?)\. No {f2} is recorded for this entry\.',rf'For {person}, only the {f1}, (.*?), was recorded; the {f2} field was left blank\.',rf"{person}'s {label} mentions (.*?) as the {f1}, without specifying a {f2}\.",rf"The register contains {person}'s {f1}, (.*?), but supplies no information about that person's {f2}\.",rf'For the entry filed for {person}, (.*?) appears under {f1}; nothing was entered under {f2}\.']
  m=re.search(missing[style],doc);assert m and m.group(1)==o['first_value'];return
 assert found and found.groups()==(o['first_value'],o['second_value'])
 if r['source']=='corrections':
  revisions=[rf"A correction changes {person}'s {f2} from (.*?) to (.*?)\.",rf'The original {f2}, (.*?), for {person} was replaced with (.*?)\.',rf"The initial notes gave (.*?) as {person}'s {f2}\. An authorized revision now specifies (.*?) instead\.",rf"An amendment to {person}'s entry says to cross out (.*?) in the {f2} field and use (.*?)\.",rf'For {person}, a later notice supersedes the earlier {f2} of (.*?): the effective value is (.*?)\.']
  m=re.search(revisions[style],doc);assert m and m.groups()==(o['second_value'],o['revised_second'])
