"""Optional hypothesis: emphasize extracted values while still teaching syntax/EOS.

Gold targets define training loss weights only; they never enter the prompt.
Evaluation remains ordinary generated exact/field accuracy and unweighted NLL.
"""
import json
import re
from pathlib import Path
import numpy as np
from json_specialist.data import canonical,strict_json
from sft.data import Corpus
from .corpus import load_tokenizer

TOKENS=re.compile(r'"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null')

def value_spans(text):
    strict_json(text);spans=[]
    for match in TOKENS.finditer(text):
        token=match.group()
        if token.startswith('"'):
            if text[match.end():].lstrip().startswith(':'):continue
            a,b=match.start()+1,match.end()-1
        else:a,b=match.span()
        if a<b:spans.append((a,b))
    return spans

def target_weights(tokenizer,target,factor):
    if not 1<=factor<=8:raise ValueError('Invalid value loss weight')
    text=canonical(target);encoded=tokenizer.encode(text,add_special_tokens=False);spans=value_spans(text)
    weights=np.ones(len(encoded.ids)+1,np.float32)
    for i,(a,b) in enumerate(encoded.offsets):
        if any(a<d and b>c for c,d in spans):weights[i]=factor
    return encoded.ids,weights

class ValueWeightedCorpus(Corpus):
    def __init__(self,directory,split,factor):
        super().__init__(directory,split);tokenizer=load_tokenizer();self.weights=[]
        with (Path(directory)/(split+'.jsonl')).open() as f:rows=[strict_json(s) for s in f]
        if len(rows)!=len(self):raise ValueError('Wrong weight row count')
        for i,row in enumerate(rows):
            ids,weights=target_weights(tokenizer,row['target'],factor);a,b=map(int,self.offsets[i:i+2]);mask=self.mask[a:b];selected=self.ids[a:b][mask]
            if selected.tolist()!=ids+[1]:raise ValueError('Target/weight alignment differs')
            prefix=len(mask)-len(weights)
            if not np.array_equal(mask,np.asarray([False]*prefix+[True]*len(weights))):raise ValueError('Unsupported loss mask')
            self.weights.append(np.concatenate([np.zeros(prefix,np.float32),weights]))
    def weight_batch(self,indices,shape):
        weights=np.zeros(shape,np.float32)
        for row,index in enumerate(indices):
            shifted=self.weights[index][1:];weights[row,:len(shifted)]=shifted
        return weights

def loss_sum(model,x,y,weights):
    import mlx.core as mx
    losses=model(x,targets=y)['token_loss'];weights=weights*(y!=-100)
    return ((losses*weights).sum(axis=1)/mx.maximum(weights.sum(axis=1),1)).sum()
