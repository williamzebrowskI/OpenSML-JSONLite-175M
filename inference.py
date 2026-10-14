#!/usr/bin/env python3
"""Run the preserved JSONLite checkpoint with supplied source text and schema."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'pretraining'),str(ROOT/'inference_native')]
from adapter import Extractor,INSTRUCTION

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,required=True)
    sources=parser.add_mutually_exclusive_group(required=True)
    sources.add_argument('--example',type=int)
    sources.add_argument('--text-file',type=Path)
    parser.add_argument('--schema-file',type=Path)
    parser.add_argument('--max-tokens',type=int,default=160)
    args=parser.parse_args()
    if not 1<=args.max_tokens<=160:parser.error('--max-tokens must be between 1 and 160')
    reference=None
    if args.example is not None:
        cases=json.loads((ROOT/'examples/cases.json').read_text())['cases']
        if not 1<=args.example<=len(cases):parser.error('Example number is outside available cases')
        if args.schema_file:parser.error('--schema-file is used only with --text-file')
        row=cases[args.example-1];text=row['document'];schema=row['schema'];reference=row['expected']
    else:
        if not args.schema_file:parser.error('--text-file requires --schema-file')
        text=args.text_file.read_text();schema=json.loads(args.schema_file.read_text())
    model=Extractor(args.weights,ROOT)
    result=model.generate(text,schema,max_tokens=args.max_tokens)
    # The reference remains caller-side and is never passed to the extractor.
    if reference is not None:result['exact']=result['success'] and result['value']==reference
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
