"""Experimental field-wise extraction, keeping arrays of associated items intact.

The input schema alone defines subrequests. The model supplies every value;
there is no repair, gold access or fabricated missing-value fallback. Scores
must distinguish this multi-call inference method from single-call generation.
"""
import copy
from jsonschema import validators,FormatChecker
from json_specialist.data import canonical,strict_json

def fields(schema,path=()):
    if schema.get('type')=='object' and schema.get('properties') and 'enum' not in schema:
        for key,child in sorted(schema['properties'].items()):yield from fields(child,path+(key,))
    else:yield path,copy.deepcopy(schema)

def sub_schema(original,path,leaf):
    node=copy.deepcopy(leaf)
    ancestors=[original];cursor=original
    for key in path[:-1]:cursor=cursor['properties'][key];ancestors.append(cursor)
    for key,parent in reversed(list(zip(path,ancestors))):
        node=dict(type='object',properties={key:node},required=[key],additionalProperties=False,
            **{k:copy.deepcopy(parent[k]) for k in ('title','description') if k in parent})
    if '$schema' in original:node['$schema']=original['$schema']
    return node

def put(result,path,value):
    cursor=result
    for key in path[:-1]:cursor=cursor.setdefault(key,{})
    cursor[path[-1]]=value

def generate(model,tokenizer,row,limit=512,*,constrained=False):
    if row['schema'].get('type')!='object':raise ValueError('Field-wise extraction requires an object root')
    from .evaluate import generate as raw
    if constrained:
        from .constrained import generate as raw
    result={};calls=[];complete=True;tokens=0
    for path,leaf in fields(row['schema']):
        if not path:raise ValueError('Unsupported field-wise root schema')
        schema=sub_schema(row['schema'],path,leaf);request=dict(row,schema=schema)
        # Remove labels even though the shared prompt does not read them.
        for key in ('target','expected','oracle','evidence','original_record'):request.pop(key,None)
        answer=raw(model,tokenizer,request,limit);tokens+=answer['output_tokens'];call=dict(path=list(path),**answer)
        try:
            value=strict_json(answer['text']);validators.validator_for(schema)(schema,format_checker=FormatChecker()).validate(value)
            if answer['stop_reason']!='document_end':raise ValueError('Subrequest did not finish')
            for key in path:value=value[key]
            put(result,path,value);call['accepted']=True
        except (ValueError,KeyError,TypeError) as exc:
            complete=False;call.update(accepted=False,error=type(exc).__name__)
        except Exception as exc:
            from jsonschema import ValidationError,SchemaError
            if not isinstance(exc,(ValidationError,SchemaError)):raise
            complete=False;call.update(accepted=False,error=type(exc).__name__)
        calls.append(call)
    text=canonical(result)
    reason='document_end' if complete else 'subtask_error'
    if len(tokenizer.encode(text,add_special_tokens=False).ids)>limit:reason='token_limit'
    return dict(text=text,output_tokens=tokens,stop_reason=reason,calls=calls)
