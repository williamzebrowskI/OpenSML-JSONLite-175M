"""Guard against hidden assembly, scorer permissiveness, and target leakage."""
import copy,json,unittest
from native_json_100.data import make,prompt
from native_json_100.evaluate import score,strict_json,aggregate,retention

class NativeJsonChecks(unittest.TestCase):
 def setUp(self):
  self.base=dict(id='unit:1',group='play lark by mira sol',source='PlayMusic',document='Play Lark by Mira Sol',fields=['artist','album'],expected={'artist':['Mira Sol'],'album':[]},spans=[dict(field='artist',start=13,end=21,value='Mira Sol')],pair_id=None,provenance={})
  self.row=make(self.base,'canonical','train')
 def check_text(self,text,stop='turn_end'):return score(self.row,text,stop)
 def test_valid_reordered_object(self):
  self.assertEqual(self.check_text('{"album":[],"artist":["Mira Sol"]}')['exact'],1)
 def test_bad_outputs(self):
  for text in ['{"artist":["Mira Sol"],"album":[],"artist":["Mira Sol"]}','```json\n{"artist":["Mira Sol"],"album":[]}\n```','{"artist":["Mira Sol"],"album":[]} Done','{"artist":["Mira Sol"],"album":null}','{"artist":["Mira Sol"]}','{"artist":["Mira Sol"],"album":[],"extra":[]}','{"artist":"Mira Sol","album":[]}','[]','NaN','{"artist":["Mira Sol","Mira Sol"],"album":[]}']:
   with self.subTest(text=text):self.assertEqual(self.check_text(text)['exact'],0)
 def test_truncation_fails_even_valid_json(self):self.assertEqual(self.check_text('{"artist":["Mira Sol"],"album":[]}',stop='token_limit')['exact'],0)
 def test_wrong_and_unsupported_values(self):
  a=self.check_text('{"artist":["Lark"],"album":["invented"]}')
  self.assertEqual(a['exact'],0);self.assertEqual(a['unsupported_values'],1);self.assertEqual(a['fp'],2)
 def test_false_abstention(self):self.assertEqual(self.check_text('{"artist":[],"album":[]}')['false_abstention'],1)
 def test_target_not_in_prompt(self):
  changed=copy.deepcopy(self.base);changed['expected']['artist']=['ZZZ GOLD CANARY'];changed['spans']=[]
  self.assertEqual(self.row['messages'][0],make(changed,'canonical','train')['messages'][0])
  self.assertNotIn('ZZZ GOLD CANARY',make(changed,'canonical','train')['messages'][0]['content'])
 def test_schema_alias(self):
  r=make(self.base,'alias','development');self.assertEqual(score(r,r['messages'][-1]['content'],'turn_end')['exact'],1)
  self.assertEqual(score(r,self.row['messages'][-1]['content'],'turn_end')['exact'],0)
 def test_nullable(self):
  r=make(self.base,'nullable','train');self.assertEqual(r['expected'],{'artist':'Mira Sol','album':None})
  self.assertEqual(score(r,'{"artist":"Mira Sol","album":null}','turn_end')['exact'],1)
 def test_nonfinite_and_nested_duplicate_rejected(self):
  for text in ['{"a": Infinity}', '{"a": {"x":1,"x":2}}']:
   with self.assertRaises(ValueError):strict_json(text)
 def test_empty_null_not_interchangeable(self):self.assertEqual(self.check_text('{"artist":["Mira Sol"],"album":null}')['absent_correct'],0)
 def test_retention_blocks_one_domain_loss(self):
  old={'metrics':{'all':{'exact':.5,'field_accuracy':.8,'counts':{'false_abstention':1,'unsupported_values':1}},'domain/PlayMusic':{'exact':.6,'field_accuracy':.9,'counts':{'false_abstention':1,'unsupported_values':1}}}}
  candidate=copy.deepcopy(old);candidate['metrics']['all']['exact']=.7;candidate['metrics']['domain/PlayMusic']['exact']=.5
  self.assertFalse(retention(candidate,old)['passed'])
  self.assertTrue(retention(old,old)['passed'])
if __name__=='__main__':unittest.main()
