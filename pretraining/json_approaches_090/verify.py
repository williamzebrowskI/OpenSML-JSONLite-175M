"""Boundary/adversarial tests of serving representations before any real update."""
import copy,json
from types import SimpleNamespace
from unittest.mock import patch
from .data import sources,messages,schema
from .evaluate import decode,json_score,generate_request

def checks(tok):
 train,demos=sources();r=dict(id='fixture',source='PlayMusic',group='play echo',document='Play Echo',fields=['artist','album'],expected={'artist':['Echo'],'album':[]},spans=[dict(field='artist',start=5,end=9,value='Echo')],pair_id=None)
 assert decode(r,92,'artist: Echo\nalbum: NOT_STATED')==r['expected']
 assert decode(r,94,'[["Echo"],[]]')==r['expected']
 assert decode(r,96,'{"artist":[[5,9]],"album":[]}')==r['expected']
 for n,text in [(92,'artist: NOT_STATED\nartist: Echo\nalbum: NOT_STATED'),(94,'[["Echo"]]'),(96,'{"artist":[[5,99]],"album":[]}'),(97,'{"artist":[[true,9]],"album":[]}'),(90,'{"artist":[],"artist":["Echo"],"album":[]}')]:
  try:decode(r,n,text)
  except (ValueError,TypeError):pass
  else:raise AssertionError('bad representation accepted')
 assert not json_score(r,'{"artist":["made-up"],"album":[]}','turn_end')['exact']
 assert json_score(r,'{"artist":["made-up"],"album":[]}','turn_end')['unsupported_values']==1
 assert not json_score(r,'{"artist":[],"album":["Echo"]}','turn_end')['exact']
 # All single-call inference inputs are independent of expected/spans.
 changed=dict(r,expected={'artist':['SECRET GOLD ANSWER'],'album':['HIDDEN']},spans=[])
 for n in (90,91,92,94,96,97,99):assert messages(r,n,demos,training=False)==messages(changed,n,demos,training=False)
 # The two-stage conversion must see generated extraction, never the gold.
 seen=[]
 def fake_generate(model,tokenizer,ms,limit):
  seen.append(copy.deepcopy(ms))
  return dict(text='artist: Echo\nalbum: NOT_STATED' if len(seen)==1 else '{"artist":["Echo"],"album":[]}',stop_reason='turn_end',tokens=10)
 with patch('sft.evaluate.generate',fake_generate),patch('json_approaches_090.evaluate.nll',lambda *args:0.):
  out=generate_request(SimpleNamespace(cfg=SimpleNamespace(max_seq_len=2048)),tok,changed,98,demos)
 assert len(seen)==2 and len(seen[0])==1 and seen[1][1]['content']=='artist: Echo\nalbum: NOT_STATED'
 assert 'SECRET GOLD ANSWER' not in json.dumps(seen) and 'HIDDEN' not in json.dumps(seen)
 assert json.loads(out['text'])==r['expected']
 print('[preflight boundary] strict representations, wrong/missing content, no oracle serving inputs, generated two-stage history: passed',flush=True)
