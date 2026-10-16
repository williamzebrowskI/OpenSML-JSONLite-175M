"""Inference only: immutable parent, native raw generation, no optimizer import."""
import json
from .data import PARENT,TOK,PLAN

def load_model():
 import mlx.core as mx
 from mlx.utils import tree_map
 from sml_v1.model import TransformerLM,TransformerConfig
 metadata=json.loads((PARENT/'model.safetensors.json').read_text());model=TransformerLM(TransformerConfig(**metadata['recipe']['model']))
 model.update(tree_map(lambda v:v.astype(mx.bfloat16),model.parameters()));model.load_weights(str(PARENT/'model.safetensors'));model.freeze();model.eval();mx.eval(model.parameters());return model

def make_generator(model):
 from sml_v1.tokenization import Tokenizer
 from sft.evaluate import generate
 tok=Tokenizer(TOK)
 def one(messages,temperature,seed):
  return generate(model,tok,messages,limit=PLAN['max_tokens'],temperature=temperature,seed=seed)
 return one
