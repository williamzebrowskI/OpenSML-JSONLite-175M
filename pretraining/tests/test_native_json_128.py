import collections,json,tempfile,unittest
from pathlib import Path
import numpy as np
from native_json_128.common import *

class JointTests(unittest.TestCase):
 def test_exact_original_coverage_and_grounded_split(self):
  from json_sampling_119.data import verify_case,normalized
  m=check();rows=load(DATA/'train.jsonl');original=load(OLD/'train.jsonl')
  replay=[r for r in rows if r['training_component']=='original_json']
  self.assertEqual(len(replay),15000)
  for a,b in zip(original,replay):
   for k in a:self.assertEqual(a[k],b[k])
  self.assertEqual(len(rows),21144)
  for split in ['train','development','legacy_full','reserved','original_test']:
   for r in load(DATA/(split+'.jsonl')):verify_case(r)
  sets=[{normalized(r['document']) for r in load(DATA/(s+'.jsonl'))} for s in ['train','development','reserved','original_test']]
  self.assertFalse(any(sets[i]&sets[j] for i in range(4) for j in range(i)))
  groups=collections.Counter(r['pair_id'] for r in load(DATA/'development.jsonl') if r.get('pair_id'))
  self.assertEqual(set(groups.values()),{2});self.assertGreater(len(groups),0)
 def test_mask_and_context(self):
  from native_json_128.runtime import Encoding,batch
  enc=Encoding();largest=0
  for split in ['train','development','legacy_full','reserved','original_test']:
   for r in load(DATA/(split+'.jsonl')):
    x,y=enc.row(r);prefix=enc.prefix(r['messages'][:-1]);largest=max(largest,len(x));self.assertTrue(np.all(y[:len(prefix)-1]==-100));self.assertEqual(y[-1],3);self.assertEqual(enc.tokenizer.decode(y[y!=-100][:-1].tolist()),r['messages'][-1]['content'])
  data=[enc.row(r) for r in load(DATA/'development.jsonl')[:3]];x,y=batch(data,[0,1,2]);self.assertTrue(np.all(y[x==0]==-100));self.assertLess(largest,2048)
 def test_cpu_gradient_with_variable_padding_and_partial_microbatch(self):
  import mlx.core as mx,mlx.nn as nn
  from mlx.utils import tree_flatten
  from sml_v1.model import TransformerLM,TransformerConfig
  from native_json_128.runtime import accumulate,batch,loss
  previous=mx.default_device();self.addCleanup(mx.set_default_device,previous);mx.set_default_device(mx.cpu);mx.random.seed(128)
  model=TransformerLM(TransformerConfig(vocab_size=32,d_model=16,n_heads=2,n_kv_heads=1,n_layers=1,max_seq_len=256,mlp_multiple_of=16,attention_impl='vanilla'))
  data=[(np.array([2,4,5]),np.array([-100,5,3])),(np.array([2,6,7,8]),np.array([-100,-100,8,3])),(np.array([2,9]),np.array([-100,3]))]
  value,g,tokens=accumulate(model,data,[0,1,2],microbatch=2);x,y=batch(data,[0,1,2]);v,reference=nn.value_and_grad(model,lambda:loss(model,mx.array(x),mx.array(y)))();self.assertEqual(tokens,5);self.assertAlmostEqual(value,v.item(),places=5)
  for (k,a),(n,b) in zip(tree_flatten(g),tree_flatten(reference)):self.assertEqual(k,n);np.testing.assert_allclose(np.array(a),np.array(b),atol=3e-6,rtol=3e-5)
 def test_save_reload_resume_is_identical(self):
  import mlx.core as mx
  from mlx.utils import tree_flatten
  from sml_v1.model import TransformerLM,TransformerConfig
  from sml_v1.checkpoint_bundle import resolve_bundle
  from native_json_128.train import execute
  from native_json_128.runtime import model_at
  cfg=TransformerConfig(vocab_size=32,d_model=8,n_heads=2,n_kv_heads=1,n_layers=1,max_seq_len=256,mlp_multiple_of=8)
  mx.random.seed(128);a=TransformerLM(cfg);b=TransformerLM(cfg);b.update(a.parameters());data=[(np.array([2,4,5+i%10]),np.array([-100,5+i%10,3])) for i in range(128)]
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);stub=lambda *args:None
   execute(a,data,root/'whole',{'test':1},assessor=stub)
   execute(b,data,root/'split',{'test':1},stop_after=1,assessor=stub)
   execute(b,data,root/'split',{'test':1},resume=True,assessor=stub)
   for (k,x),(n,y) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):self.assertEqual(k,n);np.testing.assert_array_equal(np.array(x),np.array(y))
   pa=resolve_bundle(root/'whole/latest.json');pb=resolve_bundle(root/'split/latest.json');sa=mx.load(pa+'.optimizer.safetensors');sb=mx.load(pb+'.optimizer.safetensors');self.assertEqual(sa.keys(),sb.keys())
   for k in sa:np.testing.assert_array_equal(np.array(sa[k]),np.array(sb[k]))
   restored=model_at(pb)
   for (k,x),(n,y) in zip(tree_flatten(b.parameters()),tree_flatten(restored.parameters())):np.testing.assert_array_equal(np.array(x),np.array(y))
 def test_wrong_but_valid_json_and_case_retention(self):
  from native_json_128.scoring import score
  from native_json_128.evaluate import compare,summary
  from native_json_128.train import schedule,order_for
  self.assertEqual(schedule(331),[100,200,300,331]);self.assertEqual(sorted(order_for(21144)),list(range(21144)))
  r=load(DATA/'development.jsonl')[0];text=dumps(r['expected']);self.assertTrue(score(r,text,'turn_end')['exact']);self.assertFalse(score(r,'```json\n'+text+'\n```','turn_end')['exact'])
  wrong={k:[] if isinstance(v,list) else None for k,v in r['expected'].items()};s=score(r,dumps(wrong),'turn_end');self.assertTrue(s['schema_ok']);self.assertFalse(s['exact'])
  row=dict(r,pair_id=None);good=dict(id=r['id'],text=text,stop_reason='turn_end',checks=score(row,text,'turn_end'),output_nll=.1);bad=dict(good,text=dumps(wrong),checks=s)
  old=dict(outputs=[good],requests_sha256='same',metrics=summary([row],[good],[.1]));new=dict(outputs=[bad],requests_sha256='same',metrics=summary([row],[bad],[.1]));self.assertFalse(compare(new,old,rows=[row])['passed'])
if __name__=='__main__':unittest.main()
