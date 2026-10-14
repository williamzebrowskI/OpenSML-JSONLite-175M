"""Conservative relation-aware equivalence for fictional fact-oracle QA.
Does not grade arbitrary open-domain paraphrases. Unrecognized is explicit.
"""
import re
VERSION='qa-relation-equivalence-v1'
def clean(s):return re.sub(r'\s+',' ',s.strip().lower()).rstrip(' .')
def equivalent(row,text,stop):
 if stop not in ('turn_end','document_end'):return dict(correct=False,recognized=False,reason='non_natural_stop')
 t=clean(text)
 if row['answer_absent']:
  ok=t in ('not stated','the passage does not state this','the passage does not provide this information','this information is not stated')
  return dict(correct=ok,recognized=ok,reason='explicit_absence' if ok else 'not_an_accepted_abstention')
 if 'oracle' not in row:return dict(correct=False,recognized=False,reason='outside_controlled_scope')
 o=row['oracle'];p=re.escape(clean(o['person']));f=re.escape(clean(o['first_field']));g=re.escape(clean(o['second_field']));x=re.escape(clean(o['first_value']));y=re.escape(clean(o['second_value']))
 patterns=[]
 if row['source']=='multipart':
  # Both field names, both values and correct person must be bound explicitly.
  prefixes=[rf'the {f} for {p} is {x}',rf"{p}'s {f} is {x}",rf'the entry (?:filed )?for {p} has two details[:,]? its {f} is {x}']
  for a in prefixes:patterns.append(a+rf',? (?:and|while) (?:the|its) {g} is {y}')
  patterns.append(rf'the {g} for {p} is {y},? (?:and|while) (?:the|their) {f} is {x}')
 else:
  value=o['second_value'] if row['source']=='calibration' or o.get('version')=='original' else o['revised_second'];v=re.escape(clean(value))
  # A bare value directly answers a single-field question. Anchoring disallows
  # additional contradicting statements and all distractor values.
  patterns.append(v)
  if row['source']=='calibration':
   patterns += [rf'the {g} for {p} is {v}',rf"{p}'s {g} is {v}",rf'the {g} is {v}']
  elif o['version']=='current':
   patterns += [rf'the (?:current|revised|effective) {g} for {p} is {v}',rf'the effective value for {p} is {v}',rf'the effective value is {v}',rf'the {g} for {p} is now {v}']
  else:
   patterns += [rf'the (?:original|earlier|previous) {g} for {p} was {v}',rf'the original value was {v}',rf'before the (?:correction|revision), the {g} for {p} was {v}']
 ok=any(re.fullmatch(pat,t) for pat in patterns)
 return dict(correct=ok,recognized=ok,reason='validated_equivalence' if ok else 'unrecognized_or_incorrect')
