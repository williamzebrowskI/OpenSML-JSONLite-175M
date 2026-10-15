"""Cardinality, absence and distillation regression checks for SFT105."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from native_json_105.data import make_row,verify_row,TEMPLATES,PLAN
from native_json_105.evaluate import score

class CardinalityChecks(unittest.TestCase):
 def row(self,n=2,variant='canonical',field='artist'):
  return make_row(field,['Avery Moon','Rowan Oak','Cedar Lane'][:n],'Silver Sketch','Play songs by {values}.' if n else 'Play the album "{related}".',n,'train',variant)
 def test_zero_is_role_specific_not_all_empty(self):
  r=self.row(0);verify_row(r);self.assertEqual(r['expected'],{'artist':[],'album':['Silver Sketch']})
  self.assertEqual(score(r,json.dumps({'artist':['Silver Sketch'],'album':[]}),'turn_end')['exact'],0)
 def test_nullable_single_value(self):
  r=self.row(1,'nullable');self.assertEqual(r['expected'],{'artist':'Avery Moon','album':None});verify_row(r)
 def test_nullable_multiple_stays_array(self):
  r=self.row(3,'nullable');self.assertEqual(r['schema']['properties']['artist']['type'],'array');self.assertEqual(len(r['expected']['artist']),3)
 def test_array_cardinalities(self):
  for n in range(4):
   r=self.row(n);verify_row(r);self.assertEqual(len(r['expected']['artist']),n)
 def test_merged_omitted_reversed_invented_values_fail(self):
  r=self.row()
  for v in [['Avery Moon and Rowan Oak'],['Avery Moon'],['Rowan Oak','Avery Moon'],['Avery Moon','Rowan Oak','Invented']]:
   self.assertFalse(score(r,json.dumps(dict(r['expected'],artist=v)),'turn_end')['exact'])
 def test_alias_definition_and_exact_case(self):
  r=self.row(2,'alias');verify_row(r);self.assertIn('out_artist',r['expected']);self.assertFalse(score(r,json.dumps(self.row()['expected']),'turn_end')['exact'])
 def test_city_zero_does_not_infer_city(self):
  r=make_row('city',[],'Andorra','Get weather in the country {related}.',0,'train','nullable');self.assertEqual(r['expected'],{'city':None,'country':'Andorra'});verify_row(r)
 def test_distinct_template_splits(self):
  pools=[{t for templates in panel.values() for t in templates} for panel in TEMPLATES.values()]
  for i,a in enumerate(pools):
   for b in pools[i+1:]:self.assertFalse(a&b)
 def test_output_limit_and_schema_validation(self):
  r=self.row();text=r['messages'][-1]['content'];self.assertTrue(score(r,text,'turn_end')['exact']);self.assertFalse(score(r,text,'token_limit')['exact']);self.assertFalse(score(r,'```json\n'+text+'\n```','turn_end')['exact'])
 def test_wrong_offset_rejected(self):
  r=self.row();r['spans'][0]['start']+=1
  with self.assertRaises(ValueError):verify_row(r)
 def test_baseline_commit_and_wrong_step_rejection(self):
  import native_json_105.train as tr
  with tempfile.TemporaryDirectory() as d:
   report=Path(d);val={'step':0};(report/'parent-development.json').write_text(json.dumps(val))
   with patch.object(tr,'REPORT',report):
    tr.initialize_cached_baseline(report/'eval/0.json');self.assertEqual(json.loads((report/'eval/0.json').read_text()),val)
    (report/'parent-development.json').write_text('{"step":32}')
    with self.assertRaises(ValueError):tr.initialize_cached_baseline(report/'wrong.json')
 def test_one_pass_schedule(self):
  from native_json_105.train import order_for,lr_for
  batches=order_for(3072,PLAN);self.assertEqual(len(batches),96);self.assertEqual(np.concatenate(batches).tolist(),list(range(3072)));self.assertAlmostEqual(lr_for(96,96,PLAN),PLAN['final_lr'])

class ObjectiveChecks(unittest.TestCase):
 def test_real_value_and_grad_callable_signature(self):
  import mlx.core as mx
  import mlx.nn as nn
  from native_json_105.runtime import objectives
  class Tiny(nn.Module):
   def __init__(self):super().__init__();self.w=mx.ones((3,3))
   def logits(self,x):return x@self.w,None
  student=Tiny();teacher=Tiny();teacher.freeze()
  x=mx.array([[[1.,2.,3.],[3.,1.,0.]]]);y=mx.array([[0,1]])
  for fn in objectives(student,teacher,1.):
   value,g=fn(x,y);mx.eval(value,g)
   self.assertTrue(np.isfinite(float(value.item())));self.assertEqual(g['w'].shape,(3,3))
 def test_kl_matches_numpy_and_ignores_prompt(self):
  import mlx.core as mx
  from native_json_105.runtime import masked_objective
  s=np.array([[[1.,2.,3.],[1.,0.,-1.],[0.,1.,0.]]],np.float32);t=s.copy();t[0,1]=[0.,2.,1.];y=np.array([[-100,2,1]],np.int32)
  def logsoft(x):return x-np.log(np.exp(x).sum(-1,keepdims=True))
  ls,lt=logsoft(s),logsoft(t);ce=-ls[0,[1,2],[2,1]].mean();kl=(np.exp(lt)*(lt-ls)).sum(-1)[0,1:].mean()
  got=float(masked_objective(mx.array(s),mx.array(t),mx.array(y),1.).item());self.assertAlmostEqual(got,float(ce+kl),places=5)
  s[0,0]=[100.,-100.,0.];self.assertAlmostEqual(float(masked_objective(mx.array(s),mx.array(t),mx.array(y),1.).item()),got,places=5)
 def test_zero_kl_at_parent_and_reference_gradient_blocked(self):
  import mlx.core as mx
  from native_json_105.runtime import masked_objective
  x=mx.array([[[1.,2.,3.],[0.,1.,2.]]]);y=mx.array([[2,1]])
  ce=masked_objective(x,None,y,0.);same=masked_objective(x,x,y,1.);self.assertAlmostEqual(float(ce.item()),float(same.item()),places=6)
  g=mx.grad(lambda t:masked_objective(x,t,y,1.))(x);self.assertTrue(np.all(np.asarray(g)==0))
 def test_student_gradient_is_finite_and_masked(self):
  import mlx.core as mx
  from native_json_105.runtime import masked_objective
  x=mx.array([[[1.,2.,3.],[0.,1.,2.]]]);y=mx.array([[-100,1]])
  g=mx.grad(lambda v:masked_objective(v,x,y,1.))(x);a=np.asarray(g);self.assertTrue(np.isfinite(a).all());self.assertTrue((a[0,0]==0).all());self.assertGreater(np.abs(a[0,1]).sum(),0.)
 def test_replay_and_new_contributions_weighted_by_actual_records(self):
  from native_json_105.runtime import accumulate_gradients
  import mlx.core as mx
  class C: replay_indices={0,1,2}
  def fake(fn,c,idx,**kw):
   return (2. if fn=='replay' else 6.),{'p':mx.array(4. if fn=='replay' else 8.)},{}
  with patch('native_json_100.runtime.accumulate_gradients',fake):
   value,g,profile=accumulate_gradients(('replay','new'),C(),[0,1,2,3],policy='micro_multiple128')
  self.assertEqual(value,3.);self.assertEqual(float(g['p'].item()),5.);self.assertEqual(profile['actual_records'],4)

if __name__=='__main__':unittest.main()
