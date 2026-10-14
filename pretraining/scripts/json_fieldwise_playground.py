"""Pinned native SFT extraction plus explicit JSON assembly for the playground."""
import hashlib,json,time
from pathlib import Path
import jsonschema
from completion_playground import Checkpoint,ModelSource
from json_playground import JSONInferenceEngine,JSONGenerationRequest,sample_json_token
from extract_068.data import GLOSS,prompt
from json_approaches_090.evaluate import strict_json,decode
from sml_v1.tokenization import Tokenizer
from sft.data import serialize

INSTRUCTION='Copy exact values for each requested field, preserving case and source order. Use [] when a field is not stated.'
MODE='fieldwise-assembly'

class FieldwiseCheckpoint(Checkpoint):
 def public(self):
  m=self.metadata
  return dict(checkpoint=self.bundle.name,step=m['step'],tokens=m['tokens'],validation_loss=self.evaluation['exact'],validation_kind='observed development JSON exact; includes assembly',parent_step=379,context_limit=self.config['max_seq_len'],max_output_tokens=160,tokenizer='Own 32K byte-BPE',model_type='json',serialization='native-sft-chat-v1',selection=self.selection,output_mode=MODE,model_note='One extraction call per field, followed by JSON assembly. Known field definitions; string arrays only. Missing values are [].',fixed_instruction=INSTRUCTION)

class FieldwiseModelSource(ModelSource):
 def __init__(self,model_id,label,bundle,tokenizer_dir,evaluation_path,baseline=False):
  super().__init__(model_id,label,Path(bundle),tokenizer_dir,selection='user-selected diagnostic; retention gates failed' if not baseline else 'preserved parent baseline')
  self.evaluation_path=Path(evaluation_path);self.baseline=baseline
 def checkpoint(self):
  bundle=self.run;manifest=json.loads((bundle/'manifest.json').read_text())
  if bundle.is_symlink() or manifest.get('format')!='sml-pretrain-bundle-v1':raise ValueError('Invalid pinned extraction bundle')
  path=bundle/'model.safetensors.json';raw=path.read_bytes();record=manifest['files'][path.name]
  if path.is_symlink() or len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Extraction metadata checksum mismatch')
  meta=json.loads(raw)
  if meta['step']!=manifest['step'] or 'extraction_sft' not in meta:raise ValueError('Expected native extraction SFT checkpoint')
  evaluation=json.loads(self.evaluation_path.read_text())
  if not self.baseline and evaluation['step']!=meta['step']:raise ValueError('Extraction evaluation/update mismatch')
  return FieldwiseCheckpoint(bundle,meta,manifest,model_type='json',evaluation=evaluation['metrics']['json']['all'],selection=self.selection)

def validate_fieldwise(options,checkpoint):
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
 tokenizer=Tokenizer(checkpoint_source_tokenizer(checkpoint))
 rows=[];prompt_tokens=0
 for key,f in mapping.items():
  ms=[dict(role='user',content=prompt(options.document,[f],0))]
  ids,_=serialize(tokenizer,ms,generation=True)
  if len(ids)+options.max_tokens>checkpoint.config['max_seq_len']:raise ValueError('Source is too long for this checkpoint; no truncation is performed.')
  rows.append(dict(key=key,field=f,messages=ms));prompt_tokens+=len(ids)
 return tokenizer,dict(document=options.document,schema=schema,fields=rows,prompt_tokens=prompt_tokens)

def checkpoint_source_tokenizer(checkpoint):
 from completion_playground import DEFAULT_V1_TOKENIZER
 return DEFAULT_V1_TOKENIZER

def generate_field(model,tokenizer,messages,options,cancelled):
 import mlx.core as mx
 import numpy as np
 ids,_=serialize(tokenizer,messages,generation=True);inputs=mx.array(ids[None]);cache=None;out=[];reason='token_limit';rng=np.random.default_rng(options.seed)
 for _ in range(options.max_tokens):
  if cancelled.is_set():return None
  logits,cache=model.logits(inputs,cache)
  token=int(mx.argmax(logits[0,-1]).item()) if options.temperature==0 and options.repetition_penalty==1 else sample_json_token(logits[:,-1,:],out,options,rng)
  if token in (1,3):reason='turn_end' if token==3 else 'document_end';break
  if token in (0,2):reason='unexpected_structure';break
  out.append(token);inputs=mx.array([[token]],dtype=mx.int32)
 return dict(text=tokenizer.decode(out),tokens=len(out),stop_reason=reason)

class FieldwiseInferenceEngine(JSONInferenceEngine):
 def generate(self,options,emit,cancelled):
  source=self.sources[options.model_id or self.default_model];checkpoint=source.checkpoint()
  if not isinstance(checkpoint,FieldwiseCheckpoint):return super().generate(options,emit,cancelled)
  if not isinstance(options,JSONGenerationRequest):raise ValueError('Use the text + schema extraction form for this model.')
  tokenizer,row=validate_fieldwise(options,checkpoint)
  emit(dict(type='status',message=f'Loading {source.model_id.upper()} · update {checkpoint.metadata["step"]}'))
  self.load(checkpoint,source.tokenizer_dir)
  emit(dict(type='start',**checkpoint.public(),model_id=source.model_id,constrained=False,prompt_tokens=row['prompt_tokens'],temperature=options.temperature,repetition_penalty=options.repetition_penalty,top_k=options.top_k,seed=options.seed))
  began=time.perf_counter();value={};pieces=[];natural=True;valid_representation=True
  import mlx.core as mx
  try:
   for request in row['fields']:
    if cancelled.is_set():return
    emit(dict(type='status',message='Extracting '+request['key']))
    result=generate_field(self.model,tokenizer,request['messages'],options,cancelled)
    if result is None:return
    pieces.append(dict(field=request['key'],**result));natural &= result['stop_reason'] in ('turn_end','document_end')
    try:value[request['key']]=decode(dict(document=row['document'],fields=[request['field']]),93,result['text'])[request['field']]
    except (ValueError,TypeError,KeyError,RecursionError):valid_representation=False
   valid=valid_representation and jsonschema.Draft202012Validator(row['schema']).is_valid(value)
   elapsed=time.perf_counter()-began;tokens=sum(p['tokens'] for p in pieces)
   reason='turn_end' if natural else next(p['stop_reason'] for p in pieces if p['stop_reason'] not in ('turn_end','document_end'))
   emit(dict(type='done',text=json.dumps(value,ensure_ascii=False) if valid_representation else '\n\n'.join(p['text'] for p in pieces),value=value if valid else None,schema_valid=valid,success=valid and natural,stop_reason=reason,output_tokens=tokens,generated_tokens=tokens,seconds=elapsed,tokens_per_second=tokens/max(elapsed,1e-6),constrained=False,output_mode=MODE,model_calls=len(pieces),extractions=pieces,temperature=options.temperature,repetition_penalty=options.repetition_penalty,top_k=options.top_k,seed=options.seed,peak_gib=mx.get_peak_memory()/1024**3))
  finally:mx.clear_cache()
