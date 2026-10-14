"""Portable fieldwise extraction and JSON assembly; default protocol parity only."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import jsonschema
import numpy as np
from sml_v1.tokenization import Tokenizer
from sft.data import serialize

INSTRUCTION='Copy exact values for each requested field, preserving case and source order. Use [] when a field is not stated.'

GLOSS={'artist':'performer or musical artist','entity_name':'named item to add to a playlist','music_item':'word specifying the requested music category, such as artist, track, song, or album','playlist':'playlist name','playlist_owner':'person owning the playlist','city':'city or town','country':'country','cuisine':'cuisine style','facility':'requested restaurant facility','party_size_description':'description of who is attending','party_size_number':'explicit party size','poi':'named nearby point of interest','restaurant_name':'restaurant name','restaurant_type':'kind of restaurant','served_dish':'food or dish','sort':'requested ranking, ordering, or preference','spatial_relation':'spatial or proximity expression','state':'state or region','timeRange':'date, time, or time-range expression','condition_description':'weather condition','condition_temperature':'temperature description','current_location':'expression referring to current location','geographic_poi':'geographic point of interest','album':'album name','genre':'music genre','service':'music service','track':'track name','year':'year expression','best_rating':'maximum rating scale','object_name':'name of the work being rated or searched for','object_part_of_series_type':'type of book series','object_select':'selection expression referring to the work','object_type':'kind of work or object','rating_unit':'rating unit','rating_value':'given rating','location_name':'screening location name','movie_name':'movie name','movie_type':'kind of movie','object_location_type':'type of screening venue'}

def prompt(text,fields,style):
 variants=['Extract only the requested fields from the text.','Read the request below and copy the values for the listed fields.','Identify the exact text spans for each requested field.','Return the requested information using only the supplied text.']
 return variants[style%4]+" Copy each value exactly, preserving spelling and case. Do not normalize numbers or dates. Output one field: value line per value, in requested-field order and source order within a field. If the text does not supply a requested field, output field: NOT_STATED. Do not add explanations.\n\nRequested fields:\n"+'\n'.join(f'- {f}: {GLOSS[f]}' for f in fields)+'\n\nText:\n'+text

def strict_json(text):
 def pairs(xs):
  d={}
  for k,v in xs:
   if k in d:raise ValueError('duplicate key')
   d[k]=v
  return d
 def constant(x):raise ValueError(x)
 return json.loads(text,object_pairs_hook=pairs,parse_constant=constant)

def decode(r,n,text):
 if n in (90,91,98,99):result=strict_json(text)
 elif n in (92,93):
  result={}
  for line in text.strip().splitlines():
   if ': ' not in line:raise ValueError('invalid field line')
   f,v=line.split(': ',1)
   if f not in r['fields'] or not v:raise ValueError('invalid key/value')
   if v=='NOT_STATED':
    if f in result:raise ValueError('duplicate empty field')
    result[f]=[]
   else:
    if f in result and not result[f]:raise ValueError('empty plus value')
    result.setdefault(f,[]).append(v)
 elif n==94:
  values=strict_json(text)
  if not isinstance(values,list) or len(values)!=len(r['fields']):raise ValueError('wrong field count')
  result=dict(zip(r['fields'],values))
 elif n==95:
  if len(r['fields'])!=1 or not text.strip():raise ValueError('empty output')
  result={r['fields'][0]:[] if text.strip()=='NOT_STATED' else text.strip().splitlines()}
 elif n in (96,97):
  indexes=strict_json(text)
  if not isinstance(indexes,dict) or set(indexes)!=set(r['fields']):raise ValueError('wrong keys')
  result={}
  for f,xs in indexes.items():
   if not isinstance(xs,list):raise ValueError('wrong span list')
   result[f]=[];end=-1
   for pair in xs:
    if not isinstance(pair,list) or len(pair)!=2 or any(type(x)!=int for x in pair):raise ValueError('bad span')
    a,b=pair
    if not (0<=a<b<=len(r['document'])) or a<end:raise ValueError('bad span bounds/order')
    result[f].append(r['document'][a:b]);end=b
 else:raise ValueError(n)
 if not isinstance(result,dict) or set(result)!=set(r['fields']):raise ValueError('wrong final object keys')
 if any(not isinstance(v,list) or any(not isinstance(s,str) for s in v) for v in result.values()):raise ValueError('bad final values')
 return {f:result[f] for f in r['fields']}

def compile_request(options,tokenizer,context_limit):
 if options.max_tokens>160:raise ValueError('This route supports at most160generated tokens per field.')
 schema=strict_json(options.schema_text)
 if not isinstance(schema,dict) or schema.get('type')!='object':raise ValueError('Use an object schema with arrays of strings.')
 if set(schema)-{'$schema','title','description','type','properties','required','additionalProperties'}:raise ValueError('This route supports a closed object of string arrays, without advanced schema constraints.')
 if schema.get('$schema','https://json-schema.org/draft/2020-12/schema')!='https://json-schema.org/draft/2020-12/schema':raise ValueError('Use JSON Schema draft2020-12 for this route.')
 props=schema.get('properties')
 if not isinstance(props,dict) or not 1<=len(props)<=8:raise ValueError('Request one to eight fields.')
 if schema.get('additionalProperties') is not False or set(schema.get('required',[]))!=set(props):raise ValueError('Require every property and set additionalProperties to false.')
 mapping={}
 for key,node in props.items():
  if not isinstance(node,dict) or set(node)-{'type','items','description','title'}:raise ValueError('Unsupported field constraints.')
  if node.get('type')!='array' or node.get('items')!={'type':'string'}:raise ValueError('Each field must be an array with items: {"type":"string"}; missing values use [].')
  desc=node.get('description')
  if key in GLOSS:
   if desc is not None and desc!=GLOSS[key]:raise ValueError(f'For {key}, use its trained definition: {GLOSS[key]}')
   mapping[key]=key
  else:
   matches=[f for f,d in GLOSS.items() if desc==d]
   if len(matches)!=1:raise ValueError('Unknown field definition. Use a supported example field or its exact documented description.')
   mapping[key]=matches[0]
 if len(set(mapping.values()))!=len(mapping):raise ValueError('Do not request the same extraction definition under multiple keys.')
 jsonschema.Draft202012Validator.check_schema(schema)
 if options.instruction!=INSTRUCTION:raise ValueError('This checkpoint uses the fixed exact-span extraction instruction shown in the playground.')
 rows=[];prompt_tokens=0
 for key,f in mapping.items():
  ms=[dict(role='user',content=prompt(options.document,[f],0))]
  ids,_=serialize(tokenizer,ms,generation=True)
  if len(ids)+options.max_tokens>context_limit:raise ValueError('Source is too long for this checkpoint; no truncation is performed.')
  rows.append(dict(key=key,field=f,messages=ms));prompt_tokens+=len(ids)
 return tokenizer,dict(document=options.document,schema=schema,fields=rows,prompt_tokens=prompt_tokens)

class Extractor:
    def __init__(self, weights, repository=None):
        import mlx.core as mx
        from mlx.utils import tree_map
        from sml_v1.model import TransformerLM, TransformerConfig, count_parameters
        from sml_v1.common import file_sha256
        self.root = Path(repository or Path(__file__).resolve().parents[1])
        self.config = json.loads((self.root/'config.json').read_text())
        weights = Path(weights)
        if file_sha256(weights) != self.config['weight_sha256']:
            raise ValueError('Weight SHA-256 does not match the recreated SFT-093/160 release identity')
        self.tokenizer = Tokenizer(self.root/'tokenizer')
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(4*1024**3)
        self.model = TransformerLM(TransformerConfig(**self.config['model']))
        self.model.update(tree_map(lambda value:value.astype(mx.bfloat16),self.model.parameters()))
        self.model.load_weights(str(weights))
        mx.eval(self.model.parameters())
        self.model.eval()
        if count_parameters(self.model) != self.config['parameters']:
            raise ValueError('Unexpected parameter count')

    def generate(self, document, schema, max_tokens=160, instruction=INSTRUCTION):
        import mlx.core as mx
        # This public CLI fixes decoding to the evaluated greedy protocol. It
        # does not promise parity for sampling settings absent from this runner.
        options=SimpleNamespace(document=document,schema_text=json.dumps(schema),
                                instruction=instruction,max_tokens=max_tokens)
        _,row=compile_request(options,self.tokenizer,self.config['context_limit'])
        start=time.perf_counter();value={};pieces=[];natural=True;valid_representation=True
        try:
            for request in row['fields']:
                ids,_=serialize(self.tokenizer,request['messages'],generation=True)
                inputs=mx.array(ids[None]);cache=None;tokens=[];stop='token_limit'
                for _ in range(max_tokens):
                    logits,cache=self.model.logits(inputs,cache)
                    token=int(mx.argmax(logits[0,-1]).item())
                    if token in (1,3):
                        stop='turn_end' if token==3 else 'document_end';break
                    if token in (0,2):
                        stop='unexpected_structure';break
                    tokens.append(token);inputs=mx.array([[token]],dtype=mx.int32)
                text=self.tokenizer.decode(tokens)
                pieces.append(dict(field=request['key'],text=text,tokens=len(tokens),stop_reason=stop))
                natural &= stop in ('turn_end','document_end')
                try:
                    value[request['key']]=decode(dict(document=document,fields=[request['field']]),93,text)[request['field']]
                except (ValueError,TypeError,KeyError,RecursionError):
                    valid_representation=False
            valid=valid_representation and jsonschema.Draft202012Validator(schema).is_valid(value)
            reason='turn_end' if natural else next(p['stop_reason'] for p in pieces if p['stop_reason'] not in ('turn_end','document_end'))
            return dict(value=value if valid else None,success=bool(valid and natural),
                        schema_valid=bool(valid),stop_reason=reason,extractions=pieces,
                        output_mode='fieldwise-assembly',model_calls=len(pieces),
                        text=json.dumps(value,ensure_ascii=False) if valid_representation else '\n\n'.join(p['text'] for p in pieces),
                        output_tokens=sum(p['tokens'] for p in pieces),seconds=time.perf_counter()-start,
                        protocol=dict(temperature=0,repetition_penalty=1,max_tokens_per_field=max_tokens))
        finally:
            mx.clear_cache()
