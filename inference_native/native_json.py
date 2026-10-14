"""Native text/schema to JSON; one whole-object generation, no assembly/repair."""
import json,time
from pathlib import Path
from types import SimpleNamespace
import jsonschema
from sml_v1.tokenization import Tokenizer
from sft.data import serialize

def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def prompt(document,schema):
 # Gold values, offsets, intent, provenance, and key-mapping metadata NEVER enter this function.
 return ('Extract information from the supplied source_text according to the supplied JSON Schema. '
         'Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; '
         'do not normalize dates or numbers. Include every required key and no other keys. '
         'For string arrays, include all matching source spans in source order, or [] when absent. '
         'For nullable strings, use the exact source span or null when absent.\n\n'+
         dumps(dict(source_text=document,schema=schema)))

INSTRUCTION=prompt('',{}).split('\n\n',1)[0]

def strict_json(text):
 def pairs(items):
  d={}
  for k,v in items:
   if k in d:raise ValueError('Duplicate JSON key')
   d[k]=v
  return d
 def nonfinite(value):raise ValueError('Nonfinite JSON number')
 return json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)

def prepare_request(options,tokenizer,context_limit):
 if options.constrained:raise ValueError('This model uses raw decoding. Disable schema-constrained decoding.')
 if options.max_tokens>256:raise ValueError('Use at most 256 output tokens for this model.')
 if options.instruction!=INSTRUCTION:raise ValueError('Use the displayed native JSON instruction to match the training format.')
 schema=strict_json(options.schema_text)
 if not isinstance(schema,dict) or schema.get('type')!='object':raise ValueError('Supply a JSON Schema for an object.')
 if set(schema)-{'$schema','title','description','type','properties','required','additionalProperties'}:raise ValueError('This experiment supports flat objects with strings, nullable strings and string arrays.')
 if schema.get('$schema','https://json-schema.org/draft/2020-12/schema')!='https://json-schema.org/draft/2020-12/schema':raise ValueError('Use JSON Schema draft 2020-12.')
 props=schema.get('properties');required=schema.get('required')
 if not isinstance(props,dict) or not 1<=len(props)<=8:raise ValueError('Request one to eight properties.')
 if not isinstance(required,list) or any(not isinstance(k,str) for k in required) or len(required)!=len(set(required)) or set(required)!=set(props) or schema.get('additionalProperties') is not False:raise ValueError('Require every property and set additionalProperties to false.')
 for node in props.values():
  if not isinstance(node,dict) or set(node)-{'type','items','description','title'}:raise ValueError('Unsupported property constraints.')
  typ=node.get('type')
  if typ=='array':
   if node.get('items')!={'type':'string'}:raise ValueError('Arrays must contain strings.')
  elif typ=='string' or (isinstance(typ,list) and len(typ)==2 and set(typ)=={'string','null'}):
   if 'items' in node:raise ValueError('items applies only to arrays.')
  else:raise ValueError('Use strings, nullable strings or arrays of strings.')
 jsonschema.Draft202012Validator.check_schema(schema)
 messages=[dict(role='user',content=prompt(options.document,schema))]
 ids,_=serialize(tokenizer,messages,generation=True)
 if len(ids)+options.max_tokens>context_limit:raise ValueError('Source and schema exceed the context budget; no truncation is performed.')
 return tokenizer,dict(document=options.document,schema=schema,messages=messages,prompt_tokens=len(ids))

class NativeJSON:
 def __init__(self,weights,repository=None):
  import mlx.core as mx
  from mlx.utils import tree_map
  from sml_v1.model import TransformerLM,TransformerConfig,count_parameters
  from sml_v1.common import file_sha256
  self.root=Path(repository or Path(__file__).resolve().parents[1]);self.config=json.loads((self.root/'config.json').read_text())
  if file_sha256(weights)!=self.config['weight_sha256']:raise ValueError('Weights do not match this native JSON release')
  self.tokenizer=Tokenizer(self.root/'tokenizer');mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3)
  self.model=TransformerLM(TransformerConfig(**self.config['model']));self.model.update(tree_map(lambda v:v.astype(mx.bfloat16),self.model.parameters()));self.model.load_weights(str(weights));mx.eval(self.model.parameters());self.model.eval()
  if count_parameters(self.model)!=self.config['parameters']:raise ValueError('Parameter count mismatch')
 def generate(self,document,schema,max_tokens=256):
  import mlx.core as mx
  options=SimpleNamespace(document=document,schema_text=dumps(schema),instruction=INSTRUCTION,constrained=False,max_tokens=max_tokens)
  _,row=prepare_request(options,self.tokenizer,self.config['context_limit']);ids,_=serialize(self.tokenizer,row['messages'],generation=True)
  inputs=mx.array(ids[None]);cache=None;out=[];reason='token_limit';began=time.perf_counter()
  try:
   for _ in range(max_tokens):
    logits,cache=self.model.logits(inputs,cache);token=int(mx.argmax(logits[0,-1]).item())
    if token in (1,3):reason='turn_end' if token==3 else 'document_end';break
    if token in (0,2):reason='unexpected_structure';break
    out.append(token);inputs=mx.array([[token]],dtype=mx.int32)
   text=self.tokenizer.decode(out);value=None;valid=False
   try:value=strict_json(text);valid=jsonschema.Draft202012Validator(schema).is_valid(value)
   except (ValueError,TypeError):pass
   elapsed=time.perf_counter()-began
   return dict(text=text,value=value if valid else None,schema_valid=valid,success=valid and reason in ('turn_end','document_end'),stop_reason=reason,output_tokens=len(out),seconds=elapsed,output_mode='native-json',model_calls=1,json_assembly=False,output_repair=False,constrained=False,protocol=dict(temperature=0,repetition_penalty=1,max_tokens=max_tokens,serialization='native-sft-chat-v1'))
  finally:mx.clear_cache()
