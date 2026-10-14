"""CLI chat inference using exactly the SFT serialization and stopping rules."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_map
from sml_v1.checkpoint_bundle import resolve_bundle
from sml_v1.model import TransformerLM, TransformerConfig
from sml_v1.tokenization import Tokenizer
from sml_v1.common import read_json
from .evaluate import generate


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--prompt'); p.add_argument('--messages',type=Path); p.add_argument('--max-tokens',type=int,default=256)
    a=p.parse_args()
    if bool(a.prompt)==bool(a.messages): p.error('Provide either --prompt or a JSON --messages array')
    weights=resolve_bundle(a.checkpoint); meta=read_json(weights+'.json')
    if 'sft' not in meta: raise ValueError('SFT chat requires an SFT checkpoint')
    tokenizer=Tokenizer(a.tokenizer)
    if meta['sft']['contract']['tokenizer']!=tokenizer.fingerprint: raise ValueError('Tokenizer mismatch')
    model=TransformerLM(TransformerConfig(**meta['recipe']['model'])); model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()))
    model.load_weights(weights); model.eval()
    messages=read_json(a.messages) if a.messages else [dict(role='user',content=a.prompt)]
    print(json.dumps(generate(model,tokenizer,messages,a.max_tokens),ensure_ascii=False,indent=2))
