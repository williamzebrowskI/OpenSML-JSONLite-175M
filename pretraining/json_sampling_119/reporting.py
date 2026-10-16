"""Separate gold-assisted oracle diagnostics from reference-free majority choice."""
from .scoring import strict_json,aggregate

def majority_index(samples):
 """Use only raw outputs, natural stopping and schema validity; never gold exact."""
 groups={}
 for i,o in enumerate(samples):
  if not o['checks']['schema_ok'] or not o['checks']['natural_stop']:continue
  obj=strict_json(o['text']);key=__import__('json').dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))
  groups.setdefault(key,[]).append(i)
 if not groups:return None
 return min(groups.values(),key=lambda positions:(-len(positions),positions[0]))[0]

def _metrics(rows,outputs):
 result=aggregate(rows,outputs,[0.]*len(rows));result.pop('paired_requests',None)
 # Likelihoods are not measured by this generation-only audit.
 for value in result.values():
  if isinstance(value,dict):value.pop('output_nll',None)
 for label in sorted({r['audit_stratum'] for r in rows}):
  indices=[i for i,r in enumerate(rows) if r['audit_stratum']==label]
  panel=aggregate([rows[i] for i in indices],[outputs[i] for i in indices],[0.]*len(indices))['all'];panel.pop('output_nll',None);result['stratum/'+label]=panel
 return result

def summarize(rows,records):
 if len(rows)!=len(records) or [r['id'] for r in rows]!=[x['id'] for x in records]:raise ValueError('Summary coverage mismatch')
 if any(not x.get('complete') or len(x['samples'])!=8 for x in records):raise ValueError('Incomplete sampling panel')
 def counts(indices):
  g=sum(records[i]['greedy']['checks']['exact'] for i in indices)
  any_sample=sum(any(o['checks']['exact'] for o in records[i]['samples']) for i in indices)
  recovered=sum(not records[i]['greedy']['checks']['exact'] and any(o['checks']['exact'] for o in records[i]['samples']) for i in indices)
  majority=sum(records[i]['samples'][j]['checks']['exact'] if (j:=majority_index(records[i]['samples'])) is not None else 0 for i in indices)
  lost_all=sum(records[i]['greedy']['checks']['exact'] and not any(o['checks']['exact'] for o in records[i]['samples']) for i in indices)
  draws=sum(sum(o['checks']['exact'] for o in records[i]['samples']) for i in indices)
  return dict(requests=len(indices),greedy_correct=g,sampled_answers_correct=draws,sampled_answers=len(indices)*8,sampled_any_correct=any_sample,greedy_wrong_recovered=recovered,greedy_correct_no_correct_sample=lost_all,oracle_any_of_greedy_and_eight=g+recovered,majority_selected_correct=majority,majority_no_valid_candidate=sum(majority_index(records[i]['samples']) is None for i in indices))
 panels={'all':counts(list(range(len(rows))))}
 for label in sorted({r['audit_stratum'] for r in rows}):panels[label]=counts([i for i,r in enumerate(rows) if r['audit_stratum']==label])
 sampled_rows=[];sampled_outputs=[]
 for r,x in zip(rows,records):
  for j,o in enumerate(x['samples']):
   identifier=f"{r['id']}::sample{j}";sampled_rows.append(dict(r,id=identifier));sampled_outputs.append(dict(o,id=identifier))
 return dict(diagnostics=panels,greedy_metrics=_metrics(rows,[x['greedy'] for x in records]),sampled_draw_metrics=_metrics(sampled_rows,sampled_outputs),likelihoods_measured=False,scope='TRAIN-only stratified diagnostic, not a held-out benchmark. Oracle correctness uses references and is not a deployable selection method. Majority selects one existing valid raw sample without references, assembly or repair.')

def markdown(result):
 lines=['# JSON sampling audit: parent SFT-105/update 64','',result['scope'],'','| Panel | Greedy correct | Any of 8 samples correct (oracle) | Greedy errors recovered | Majority selected correct |','| --- | ---: | ---: | ---: | ---: |']
 for name,p in result['diagnostics'].items():
  n=p['requests'];lines.append(f"| {name} | {p['greedy_correct']}/{n} | {p['sampled_any_correct']}/{n} | {p['greedy_wrong_recovered']} | {p['majority_selected_correct']}/{n} |")
 lines+=['','All sampled answers are scored as generated. Sampling temperature is 0.7, eight fixed independent seeds per request, output limit 256. No weights changed. No benchmark or playground outputs entered this panel.','', 'A recovery gap supports testing candidate-based training; it does not prove that RL, distillation or a runtime selector will close that gap. Requests without any sampled correct answer are candidates for better teaching data, but eight attempts do not prove that the model cannot answer them.','']
 return '\n'.join(lines)
