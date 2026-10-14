"""SFT-059 compositional pure-Python curriculum, separate specification oracles.

Trusted local specifications only. Candidate code is bounded-interpreted, never
executed. Two response styles share one program/split. This is synthetic,
correlated compositional practice, not thousands of independently sourced tasks.
"""
from dataclasses import dataclass
import ast
import hashlib
import itertools
import json
import random
import re
from .bounded_python import run_function
from .behavior_v8 import typed_equal
from .evidence_instructions import row
from .quality import digest

@dataclass
class Operation:
    name: str
    input_type: str
    output_type: str
    description: str
    expression: str
    oracle: object

SAMPLES={
 'text':['',' ','\t A a B! \n','aA-a_b.c','0 1 2','café','aaabbbb','  apple,pear  ','x\ny\tz','AbC','--A--','123','a  b c'],
 'numbers':[[],[0],[1],[-1],[2,0,-2,2,3],[-5,-3,-1],[3,2,1],[1,1,1],[0,-1,0],[10,-10,5]],
 'integer':[-12,-3,-1,0,1,2,3,5,10,12],
}

def operations():
 ops=[]
 def add(name,src,dst,desc,code,oracle):ops.append(Operation(name,src,dst,desc,code,oracle))
 # Oracles use explicit scans for transformations that candidate code expresses
 # through methods/comprehensions. They are written from the stated specification.
 add('lower','text','text','convert letters to lowercase','x.lower()',lambda x:''.join(c.lower() for c in x))
 add('upper','text','text','convert letters to uppercase','x.upper()',lambda x:''.join(c.upper() for c in x))
 add('trim','text','text','remove whitespace at both ends','x.strip()',lambda x:trim(x,True,True))
 add('left_trim','text','text','remove whitespace only at the beginning','x.lstrip()',lambda x:trim(x,True,False))
 add('right_trim','text','text','remove whitespace only at the end','x.rstrip()',lambda x:trim(x,False,True))
 add('reverse_text','text','text','reverse the characters','x[::-1]',lambda x:''.join(reversed(x)))
 add('remove_spaces','text','text','remove every literal space but keep tabs and newlines',"x.replace(' ', '')",lambda x:''.join(c for c in x if c!=' '))
 add('collapse_whitespace','text','text','replace runs of whitespace with one space and remove end whitespace',"' '.join(x.split())",lambda x:' '.join(words(x)))
 for c in ('a','b','-',',','.'):
  add('drop_'+repr(c),'text','text',f'remove every occurrence of {c!r}',f'x.replace({c!r}, "")',lambda x,c=c:''.join(v for v in x if v!=c))
 for a,b in [('a','z'),('-','_'),(',',';'),('A','b'),(' ','_')]:
  add('replace_'+repr((a,b)),'text','text',f'replace every {a!r} character with {b!r}',f'x.replace({a!r}, {b!r})',lambda x,a=a,b=b:''.join(b if v==a else v for v in x))
 for n in (1,2,3,4):
  add(f'first_chars_{n}','text','text',f'keep at most the first {n} characters',f'x[:{n}]',lambda x,n=n:''.join(v for i,v in enumerate(x) if i<n))
  add(f'last_chars_{n}','text','text',f'keep at most the last {n} characters',f'x[-{n}:]',lambda x,n=n:''.join(v for i,v in enumerate(x) if i>=len(x)-n))
 add('text_length','text','integer','count the characters','len(x)',lambda x:sum(1 for _ in x))
 add('word_count','text','integer','count whitespace-separated words','len(x.split())',lambda x:len(words(x)))
 for c in ('a','b',' ','-'):
  add('count_'+repr(c),'text','integer',f'count occurrences of {c!r}',f'x.count({c!r})',lambda x,c=c:sum(v==c for v in x))
 add('reverse_list','numbers','numbers','reverse the order without mutating the input','x[::-1]',lambda x:list(reversed(x)))
 add('sort_list','numbers','numbers','sort the numbers in ascending order','sorted(x)',lambda x:ordered(x))
 add('absolute_list','numbers','numbers','replace each number with its absolute value','[abs(v) for v in x]',lambda x:map_scan(x,lambda v:v if v>=0 else -v))
 for n in (-3,-1,0,1,2,4):
  add(f'greater_{n}','numbers','numbers',f'keep only numbers strictly greater than {n}, preserving order',f'[v for v in x if v > {n}]',lambda x,n=n:filter_scan(x,lambda v:v>n))
  add(f'at_least_{n}','numbers','numbers',f'keep only numbers at least {n}, preserving order',f'[v for v in x if v >= {n}]',lambda x,n=n:filter_scan(x,lambda v:v>=n))
  add(f'below_{n}','numbers','numbers',f'keep only numbers strictly less than {n}, preserving order',f'[v for v in x if v < {n}]',lambda x,n=n:filter_scan(x,lambda v:v<n))
 for n in (-3,-1,1,2,3,4):
  add(f'add_list_{n}','numbers','numbers',f'add {n} to each number',f'[v + ({n}) for v in x]',lambda x,n=n:map_scan(x,lambda v:v+n))
  add(f'multiply_list_{n}','numbers','numbers',f'multiply each number by {n}',f'[v * ({n}) for v in x]',lambda x,n=n:map_scan(x,lambda v:v*n))
 for n in (2,3,5):
  add(f'divisible_{n}','numbers','numbers',f'keep only numbers divisible by {n}, preserving order',f'[v for v in x if v % {n} == 0]',lambda x,n=n:filter_scan(x,lambda v:v//n*n==v))
  add(f'prefix_list_{n}','numbers','numbers',f'keep at most the first {n} numbers',f'x[:{n}]',lambda x,n=n:map_scan([v for i,v in enumerate(x) if i<n],lambda v:v))
 add('list_sum','numbers','integer','sum the numbers, returning zero for an empty list','sum(x)',lambda x:total(x))
 add('list_length','numbers','integer','count the numbers','len(x)',lambda x:total([1 for _ in x]))
 add('list_min','numbers','integer','return the smallest number or zero for an empty list','min(x) if x else 0',lambda x:extreme(x,False))
 add('list_max','numbers','integer','return the largest number or zero for an empty list','max(x) if x else 0',lambda x:extreme(x,True))
 add('absolute','integer','integer','take the absolute value','abs(x)',lambda x:x if x>=0 else -x)
 add('negate','integer','integer','negate the value','-x',lambda x:0-x)
 add('square','integer','integer','square the value','x * x',lambda x:x**2)
 for n in (-3,-1,1,2,3,5):
  add(f'add_int_{n}','integer','integer',f'add {n}',f'x + ({n})',lambda x,n=n:x-(-n))
  add(f'mul_int_{n}','integer','integer',f'multiply by {n}',f'x * ({n})',lambda x,n=n:sum(x for _ in range(n)) if n>0 else -sum(x for _ in range(-n)))
 for n in (2,3,5):
  add(f'floor_int_{n}','integer','integer',f'use floor division by {n}',f'x // {n}',lambda x,n=n:__import__('math').floor(x/n))
 return ops

def trim(x,left,right):
 start=0;end=len(x)
 if left:
  while start<end and x[start].isspace():start+=1
 if right:
  while end>start and x[end-1].isspace():end-=1
 return x[start:end]
def words(x):
 out=[];current=''
 for c in x:
  if c.isspace():
   if current:out.append(current);current=''
  else:current+=c
 if current:out.append(current)
 return out
def map_scan(xs,fn):
 out=[]
 for x in xs:out.append(fn(x))
 return out
def filter_scan(xs,fn):
 out=[]
 for x in xs:
  if fn(x):out.append(x)
 return out
def ordered(xs):
 out=[]
 for x in xs:
  i=0
  while i<len(out) and out[i]<=x:i+=1
  out.insert(i,x)
 return out
def total(xs):
 result=0
 for x in xs:result+=x
 return result
def extreme(xs,largest):
 if not xs:return 0
 result=xs[0]
 for x in xs[1:]:
  if (x>result if largest else x<result):result=x
 return result

def curriculum(seed=265900):
 ops=operations();pairs=[(a,b) for a,b in itertools.product(ops,repeat=2) if a.output_type==b.input_type and a.name!=b.name]
 random.Random(seed).shuffle(pairs)
 seen=set()
 for a,b in pairs:
  second=re.sub(r"\bx\b", "intermediate", b.expression)
  code=f'def transform(x):\n    intermediate = {a.expression}\n    return {second}\n'
  tree=ast.parse(code);key=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
  if key in seen:continue
  tests=[dict(args=[x],expected=b.oracle(a.oracle(x))) for x in SAMPLES[a.input_type]]
  if len({json.dumps(t['expected'],sort_keys=True) for t in tests})<3:continue
  if not all(typed_equal(run_function(tree,'transform',t['args']),t['expected']) for t in tests):raise ValueError((a.name,b.name,'oracle mismatch'))
  # Hold out whole ordered operation pairs, never code-only versus explained
  # copies of the same function. Primitive overlap is explicitly disclosed.
  seen.add(key);g='059:program:'+key
  p=dict(repo='local/sft059-compositional-python',revision='v1',split='authored',license='CC0-1.0',operations=[a.name,b.name],method='Trusted specifications + separately written reference oracles; all deterministic edge inputs checked with bounded AST; correlated operation pairs')
  spec=f'Write Python transform(x). Input x is '+dict(text='a string',numbers='a list of integers',integer='an integer')[a.input_type]+f'. First, {a.description}. Then, using that intermediate result, {b.description}. Return the final result. Do not modify the input.'
  co=dict(kind='python',function='transform',tests=tests)
  fence='```python\n'+code.strip()+'\n```'
  r=row('python',g,[(spec+' Return only a Python code block.',fence,co)],p,'ordered operation pair')
  yield r
  # Contrast presentation on the identical task, not an extra algorithm.
  prose=f'The first operation is to {a.description}. The second operation is to {b.description}, using the intermediate result. The function returns that final result.'
  co2=dict(co,code_only=False,explanation=True)
  r=row('python',g,[(spec+' Give a brief explanation followed by a Python code block.',prose+'\n\n'+fence,co2)],p,'ordered operation pair with explanation')
  r['python_contract']['code_only']=False
  yield r
