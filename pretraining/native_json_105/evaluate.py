"""Raw native JSON, compact current-model checks, uncontaminated retention."""
import collections,json,time
from sml_v1.common import atomic_json
from native_json_100.evaluate import score,strict_json,aggregate as original_aggregate
from .data import CORPUS,PLAN,TOK,OLD,PARENT,ROOT,load,sha,ARCHIVE,FIXTURES,PLAYGROUND_CACHE

def aggregate(rows,outputs,losses):
 m=original_aggregate(rows,outputs,losses)
 buckets=collections.defaultdict(list)
 for i,r in enumerate(rows):
  vals=list(r['expected'].values());multi=any(isinstance(v,list) and len(set(v))>1 for v in vals)
  artist=r['expected'].get('artist',r['expected'].get('selected_artist',[]))
  if r['source'].startswith('retention/'):kind='retention'
  elif isinstance(artist,list) and len(set(artist))>1:kind='multi_artist'
  elif multi:kind='multi_other'
  else:kind='single_value'
  buckets[kind].append(i)
 for variant in sorted({r['variant'] for r in rows if r['source'].startswith('retention/')}):
  idx=[i for i,r in enumerate(rows) if r['source'].startswith('retention/') and r['variant']==variant]
  m['retention/variant/'+variant]=original_aggregate([rows[i] for i in idx],[outputs[i] for i in idx],[losses[i] for i in idx])['all']
 for kind,idx in buckets.items():
  part=original_aggregate([rows[i] for i in idx],[outputs[i] for i in idx],[losses[i] for i in idx])['all'];m['category/'+kind]=part
 counts=collections.Counter()
 for r,o in zip(rows,outputs):
  try:obj=strict_json(o['text'])
  except (ValueError,TypeError):obj={}
  if not isinstance(obj,dict):obj={}
  natural=o.get('stop_reason') in ('turn_end','document_end')
  for key,v in r['expected'].items():
   if isinstance(v,list) and len(set(v))>1:
    counts['fields']+=1;counts['correct']+=int(natural and obj.get(key)==v)
    if key=='artist' or key.endswith('_artist'):
     counts['artist_fields']+=1;counts['artist_correct']+=int(natural and obj.get(key)==v)
 
 for count in range(4):
  idx=[i for i,r in enumerate(rows) if r['source']=='cardinality' and r['cardinality']==count]
  if idx:m['cardinality/'+str(count)]=original_aggregate([rows[i] for i in idx],[outputs[i] for i in idx],[losses[i] for i in idx])['all']
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):
  if r['source']=='cardinality':groups[r['contrast_group']].append(i)
 if any(len(idx)!=4 or {rows[i]['cardinality'] for i in idx}!={0,1,2,3} for idx in groups.values()):raise ValueError('Incomplete cardinality contrast group')
 m['cardinality_contrasts']=dict(groups=len(groups),all_four_correct=sum(all(outputs[i]['checks']['exact'] for i in idx) for idx in groups.values()),both_multiple_correct=sum(all(outputs[i]['checks']['exact'] for i in idx if rows[i]['cardinality']>=2) for idx in groups.values()))
 m['multi_value_fields']=dict(counts,accuracy=counts['correct']/max(1,counts['fields']),artist_accuracy=counts['artist_correct']/max(1,counts['artist_fields']))
 return m

def retention(result,baseline):
 changes={}
 for name,old in baseline['metrics'].items():
  if name.startswith(('domain/retention/','retention/variant/')) or name in ('category/retention','domain/probe/single_request'):
   now=result['metrics'][name];changes[name]=dict(exact_delta=now['exact']-old['exact'],field_accuracy_delta=now['field_accuracy']-old['field_accuracy'],false_abstention_delta=now['counts']['false_abstention']-old['counts']['false_abstention'],unsupported_values_delta=now['counts']['unsupported_values']-old['counts']['unsupported_values'],missing_accuracy_delta=now['absent_accuracy']-old['absent_accuracy'],schema_valid_delta=now['schema_valid_rate']-old['schema_valid_rate'])
 if not changes:raise ValueError('No clean retention controls')
 passed=all(v['exact_delta']>=-1e-12 and v['field_accuracy_delta']>=-1e-12 and v['unsupported_values_delta']<=0 and v['false_abstention_delta']<=0 and v['missing_accuracy_delta']>=-1e-12 and v['schema_valid_delta']>=-1e-12 for v in changes.values())
 pooled=baseline['metrics']['category/retention'];now=result['metrics']['category/retention']
 aggregate_changes={k:now[k]-pooled[k] for k in ['f1','precision','recall']}
 passed=passed and all(v>=-1e-12 for v in aggregate_changes.values())
 play_changes={}
 if 'playground' in baseline['metrics']:
  old=baseline['metrics']['playground'];now=result['metrics']['playground'];play_changes={k:now['counts'][k]-old['counts'][k] for k in ['exact','schema_ok','unsupported_values']}
  passed=passed and play_changes['exact']>=0 and play_changes['schema_ok']>=0 and play_changes['unsupported_values']<=0
 paired_delta=result['metrics']['paired_requests']['both_correct']-baseline['metrics']['paired_requests']['both_correct']
 passed=passed and paired_delta>=0
 new_multiple_delta=sum(result['metrics']['cardinality/'+str(n)]['counts']['exact']-baseline['metrics']['cardinality/'+str(n)]['counts']['exact'] for n in [2,3])
 target_improved=result['step']==0 or new_multiple_delta>0
 passed=passed and target_improved
 return dict(passed=passed,new_multiple_correct_delta=new_multiple_delta,target_improved=target_improved,changes=changes,playground_changes=play_changes,legacy_value_metric_deltas=aggregate_changes,paired_correct_delta=paired_delta,gate='No loss of exact, field, missing-field or schema correctness on each original domain/variant and pooled controls; no extra unsupported values or false abstentions. Pooled original F1/precision/recall and paired correctness must also retain baseline. Playground complete correctness/schema validity must retain baseline with no added unsupported values. Trained candidates must also improve complete correctness on the new two/three-value requests. Observed panels, no global no-regression guarantee.')

def selection(r):
 m=r['metrics'];a=m['domain/cardinality'];multi=(m['cardinality/2']['exact']+m['cardinality/3']['exact'])/2
 return (multi,a['exact'],a['field_accuracy'],a['f1'],a['schema_valid_rate'],-a['counts']['unsupported_values'],-a['output_nll'])

def evaluate(model,step,path,split='development'):
 import mlx.core as mx
 from sft.data import Corpus,serialize
 from sft.evaluate import loss_sum,generate
 from sml_v1.tokenization import Tokenizer
 tok=Tokenizer(TOK);rows=load(CORPUS/(split+'.jsonl'));c=Corpus(CORPUS,split);outputs=[];losses=[];began=time.monotonic();model.eval();cached={};prior={}
 if step==0:
  cache_path=ARCHIVE/('evaluations/development_0000000.json' if split=='development' else 'final-test.json')
  identity_path=ARCHIVE/('evaluations/development_0000000.identity.json' if split=='development' else 'final-test-identity.json')
  identity=json.loads(identity_path.read_text());weight=identity.get('model_sha256',identity.get('weight_sha256'))
  if weight!=sha(PARENT/'model.safetensors'):raise ValueError('Parent prediction cache weight identity mismatch')
  cache=json.loads(cache_path.read_text())
  if cache['step']!=0 or cache['protocol']['temperature']!=0 or cache['protocol']['max_tokens']!=256 or cache['protocol']['scorer']!='strict-raw-json100-v1':raise ValueError('Parent protocol changed')
  cached={o['id']:o for o in cache['outputs']};prior={r['id']:r for r in load(ARCHIVE/'recipe'/(split+'.jsonl'))}

 try:
  for i,r in enumerate(rows):
   x,y=c.batch([i]);losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
   if step==0 and r['id'] in cached:
    old=prior[r['id']]
    if old['messages']!=r['messages'] or old['schema']!=r['schema'] or old['expected']!=r['expected']:raise ValueError('Cached parent inputs changed')
    o=cached[r['id']]
    if score(r,o['text'],o['stop_reason'])!=o['checks']:raise ValueError('Parent cached score changed')
    out={k:v for k,v in o.items() if k not in ('id','checks','source','variant')}
   else:
    p,_=serialize(tok,r['messages'][:-1],generation=True)
    if len(p)+PLAN['max_tokens']>model.cfg.max_seq_len:raise ValueError('Would truncate')
    out=generate(model,tok,r['messages'][:-1],limit=PLAN['max_tokens'],temperature=0.)
   outputs.append(dict(**out,id=r['id'],source=r['source'],variant=r['variant'],checks=score(r,out['text'],out['stop_reason'])))
   if (i+1)%50==0 or i+1==len(rows):print(f'[native-json105 eval {step}] {split} {i+1}/{len(rows)} elapsed={time.monotonic()-began:.1f}s',flush=True)
  m=aggregate(rows,outputs,losses);play_outputs=[]
  if split=='development':
   play_rows=load(CORPUS/'playground.jsonl');pc=Corpus(CORPUS,'playground');play_losses=[];play_cache=json.loads(PLAYGROUND_CACHE.read_text())
   if play_cache['model_sha256']!=sha(PARENT/'model.safetensors') or play_cache['examples_sha256']!=sha(FIXTURES):raise ValueError('Playground parent cache changed')
   fixture_rows=json.loads(FIXTURES.read_text())['cases'];cached_play={o['id']:o for o in play_cache['outputs']}
   for i,r in enumerate(play_rows):
    x,y=pc.batch([i]);play_losses.append(float(loss_sum(model,mx.array(x),mx.array(y)).item()))
    if step==0:
     old=cached_play[r['id']]
     if any(old[k]!=r[k] for k in ['document','schema','expected']):raise ValueError('Playground cached inputs differ')
     out=dict(text=old['text'],stop_reason=old['stop_reason'],tokens=old['output_tokens'])
     if score(r,out['text'],out['stop_reason'])!=old['checks']:raise ValueError('Playground parent score changed')
    else:out=generate(model,tok,r['messages'][:-1],limit=256,temperature=0.)
    play_outputs.append(dict(**out,id=r['id'],source=r['source'],variant=r['variant'],checks=score(r,out['text'],out['stop_reason'])))
   m['playground']=original_aggregate(play_rows,play_outputs,play_losses)['all']
   print(f'[native-json105 playground {step}] correct={m["playground"]["counts"]["exact"]}/54 schema={m["playground"]["counts"]["schema_ok"]}/54',flush=True)
  result=dict(step=step,split=split,metrics=m,output_nll=m['all']['output_nll'],outputs=outputs,playground_outputs=play_outputs,protocol=dict(temperature=0,repetition_penalty=1,max_tokens=PLAN['max_tokens'],serialization='native-sft-chat-v1',prompt_truncation=False,model_calls_per_request=1,constrained_decoding=False,output_repair=False,json_assembly=False,scorer='strict-raw-json100-v1'),scope='Raw whole-object generation. Authored cardinality contrasts plus observed original/MixSNIPS/list controls. New authored values exclude existing evaluation source texts; split-disjoint values and phrasing for added controlled panels. BIO targets preserve annotated source order on space-reconstructed text; no independent gold perfection or external blind benchmark claim.')
  atomic_json(path,result);print('[native-json105 metrics]',json.dumps({k:v for k,v in result.items() if k not in ('outputs','playground_outputs')}),flush=True);return result
 finally:model.train()
