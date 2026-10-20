"""Strict object correctness and explicitly labelled structural diagnostics."""
import collections
import json
from decimal import Decimal
from jsonschema import Draft202012Validator


def strict_json(text):
    def pairs(items):
        value={}
        for key,item in items:
            if key in value:raise ValueError('Duplicate JSON key')
            value[key]=item
        return value
    def nonfinite(value):raise ValueError('Nonfinite literal')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=nonfinite)


def typed(value):
    if value is None:return ('null',)
    if isinstance(value,bool):return ('boolean',value)
    if isinstance(value,(int,float)):return ('number',str(Decimal(str(value)).normalize()))
    if isinstance(value,str):return ('string',value)
    if isinstance(value,list):return ('array',tuple(typed(x) for x in value))
    return ('object',tuple(sorted((k,typed(v)) for k,v in value.items())))


def leaves(value,path=()):
    if isinstance(value,dict) and value:
        return [item for k,v in value.items() for item in leaves(v,path+(k,))]
    if isinstance(value,list) and value:
        return [item for i,v in enumerate(value) for item in leaves(v,path+(i,))]
    return [(path,typed(value))]


def strings(value):
    if isinstance(value,str):return [value]
    if isinstance(value,dict):return [s for v in value.values() for s in strings(v)]
    if isinstance(value,list):return [s for v in value for s in strings(v)]
    return []


def allowed_literals(schema):
    if isinstance(schema,list):return set().union(*(allowed_literals(v) for v in schema)) if schema else set()
    if not isinstance(schema,dict):return set()
    values=set()
    if 'const' in schema:values.update(strings(schema['const']))
    if 'enum' in schema:values.update(strings(schema['enum']))
    for v in schema.values():values.update(allowed_literals(v))
    return values


def empty(value):
    return value is None or isinstance(value,(list,dict)) and len(value)==0


def score(source_text,schema,expected,text,stop_reason):
    parsed=None;parse_ok=False;valid=False;natural=stop_reason in ('turn_end','document_end')
    try:
        parsed=strict_json(text);parse_ok=True
        valid=isinstance(parsed,dict) and Draft202012Validator(schema).is_valid(parsed)
    except (ValueError,TypeError):pass
    actual=parsed if isinstance(parsed,dict) else {}
    want=collections.Counter(leaves(expected));got=collections.Counter(leaves(parsed)) if parse_ok else collections.Counter()
    tp=sum((want&got).values())
    correct=lambda key: key in actual and typed(actual[key])==typed(expected[key])
    absent=[k for k,v in expected.items() if empty(v)]
    present=[k for k in expected if k not in absent]
    omitted=[k for k in schema['properties'] if k not in expected and k not in schema.get('required',[])]
    output_strings=strings(parsed) if parse_ok else []
    literals=allowed_literals(schema)
    return dict(exact=int(natural and valid and typed(parsed)==typed(expected)),json_parse_ok=int(parse_ok),schema_ok=int(valid),
        natural_stop=int(natural),token_limited=int(stop_reason=='token_limit'),
        fields=len(set(expected)|set(actual)),field_correct=sum(natural and correct(k) for k in expected),
        tp=tp,fp=sum(got.values())-tp,fn=sum(want.values())-tp,
        absent_total=len(absent),absent_correct=sum(natural and correct(k) for k in absent),
        optional_omitted_total=len(omitted),optional_omitted_correct=sum(natural and k not in actual for k in omitted),
        false_abstentions=sum(k in actual and empty(actual[k]) for k in present),
        missed_absences=sum(k not in actual or not empty(actual[k]) for k in absent),
        unsupported_strings=sum(v not in source_text and v not in literals for v in output_strings),
        predicted_strings=len(output_strings))


def aggregate(rows,outputs):
    assert [r['id'] for r in rows]==[o['id'] for o in outputs]
    groups={'all':list(range(len(rows)))}
    for attr in ['family','tier','difficulty','schema_view']:
        for value in sorted({r[attr] for r in rows}):groups[attr+'/'+value]=[i for i,r in enumerate(rows) if r[attr]==value]
    metrics={}
    for name,indices in groups.items():
        c={k:sum(outputs[i]['checks'][k] for i in indices) for k in outputs[0]['checks']}
        ratio=lambda a,b:a/b if b else None
        metrics[name]=dict(count=len(indices),counts=c,complete_request_accuracy=c['exact']/len(indices),
            json_parse_rate=c['json_parse_ok']/len(indices),schema_valid_rate=c['schema_ok']/len(indices),
            natural_stop_rate=c['natural_stop']/len(indices),token_limit_rate=c['token_limited']/len(indices),
            top_level_field_accuracy=ratio(c['field_correct'],c['fields']),
            structural_path_precision=ratio(c['tp'],c['tp']+c['fp']),structural_path_recall=ratio(c['tp'],c['tp']+c['fn']),
            structural_path_f1=ratio(2*c['tp'],2*c['tp']+c['fp']+c['fn']),
            missing_value_accuracy=ratio(c['absent_correct'],c['absent_total']),
            optional_omission_accuracy=ratio(c['optional_omitted_correct'],c['optional_omitted_total']),
            unsupported_string_rate=ratio(c['unsupported_strings'],c['predicted_strings']))
    pairs=collections.defaultdict(list)
    for r,o in zip(rows,outputs):pairs[r['source_group']].append(o)
    complete=[v for v in pairs.values() if len(v)==2]
    metrics['paired_schema_views']=dict(count=len(complete),both_correct=sum(all(o['checks']['exact'] for o in v) for v in complete),
        incomplete_groups=sum(len(v)!=2 for v in pairs.values()))
    families=[v['complete_request_accuracy'] for k,v in metrics.items() if k.startswith('family/')]
    metrics['macro_family_accuracy']=sum(families)/len(families)
    return metrics
