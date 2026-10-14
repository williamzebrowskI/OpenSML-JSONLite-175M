"""Field-level diagnostics supplement, rather than replace, strict JSON scoring."""
from collections import defaultdict
from json_specialist.data import strict_json
from .evaluate import equal_json

def leaves(value,path='$'):
    if isinstance(value,dict) and value:
        return {p:v for key,item in value.items() for p,v in leaves(item,path+'.'+key).items()}
    if isinstance(value,list) and value:
        return {p:v for i,item in enumerate(value) for p,v in leaves(item,path+'['+str(i)+']').items()}
    return {path:value}

def field_counts(text,target,document):
    expected=leaves(target)
    try:actual=leaves(strict_json(text))
    except (ValueError,TypeError):actual={}
    matches=sum(path in actual and equal_json(value,actual[path]) for path,value in expected.items())
    unknown=[p for p,v in expected.items() if v is None or v==[]]
    unknown_correct=sum(p in actual and equal_json(expected[p],actual[p]) for p in unknown)
    strings=[v for v in actual.values() if isinstance(v,str) and v]
    return dict(expected_fields=len(expected),generated_fields=len(actual),correct_fields=matches,
        missing_fields=sum(p not in actual for p in expected),unknown_fields=len(unknown),unknown_correct=unknown_correct,
        nonverbatim_strings=sum(v not in document for v in strings),generated_strings=len(strings))

def summarize(outputs):
    groups=defaultdict(list)
    for row in outputs:groups[row['source']].append(row)
    result={}
    for source,rows in groups.items():
        counts=[field_counts(r['text'],r['target'],r['document']) for r in rows]
        totals={key:sum(c[key] for c in counts) for key in counts[0]}
        recall=totals['correct_fields']/max(1,totals['expected_fields']);precision=totals['correct_fields']/max(1,totals['generated_fields'])
        result[source]=dict(records=len(rows),**{k:sum(r['checks'][k] for r in rows)/len(rows) for k in ('valid_json','schema_valid','exact_json','natural_stop')},
            field_recall=recall,field_precision=precision,field_f1=2*precision*recall/max(1e-12,precision+recall),
            unknown_accuracy=totals['unknown_correct']/max(1,totals['unknown_fields']),counts=totals)
    return result
