#!/usr/bin/env python3
"""Generate native JSON from source text and schema; raw output by default."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'pretraining'),str(ROOT/'inference_native')]
from native_json import NativeJSON,strict_json

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--weights',type=Path,default=ROOT/'model.safetensors')
 s=p.add_mutually_exclusive_group(required=True);s.add_argument('--example',type=int);s.add_argument('--text-file',type=Path)
 p.add_argument('--schema-file',type=Path);p.add_argument('--max-tokens',type=int,default=256);p.add_argument('--json',action='store_true',help='Return raw text plus validation/stop metadata as a JSON record')
 a=p.parse_args()
 if not 1<=a.max_tokens<=256:p.error('--max-tokens must be 1..256')
 reference=None
 if a.example is not None:
  cases=json.loads((ROOT/'examples/cases.json').read_text())['cases']
  if not 1<=a.example<=len(cases):p.error('Example number outside 1..54')
  if a.schema_file:p.error('--schema-file is for --text-file mode')
  row=cases[a.example-1];text=row['document'];schema=row['schema'];reference=row['expected']
 else:
  if not a.schema_file:p.error('--text-file requires --schema-file')
  text=a.text_file.read_text();schema=strict_json(a.schema_file.read_text())
 model=NativeJSON(a.weights,ROOT);result=model.generate(text,schema,max_tokens=a.max_tokens)
 if reference is not None:result['exact']=bool(result['success'] and result['value']==reference)
 if a.json:print(json.dumps(result,indent=2,ensure_ascii=False))
 else:sys.stdout.write(result['text'])
if __name__=='__main__':main()
