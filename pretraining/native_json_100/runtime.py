"""Actual batch32 microbatch2 accumulation; no unused 64-slot fake examples."""
import numpy as np
from sft.data import shifted
from json_evidence_binding.runtime import sequence_width

def accumulate_gradients(one,corpus,indices,*,policy,microbatch=2):
 import mlx.core as mx
 from mlx.utils import tree_map
 indices=np.asarray(indices,np.int64)
 if indices.ndim!=1 or not 0<len(indices)<=32 or np.any(indices<0) or np.any(indices>=len(corpus)):raise ValueError('Invalid batch')
 total=None;loss=0.;widths=[]
 for start in range(0,len(indices),microbatch):
  chosen=indices[start:start+microbatch];width=sequence_width(int(max(corpus.lengths[chosen])),policy)
  x=np.zeros((len(chosen),width),np.int32);y=np.full(x.shape,-100,np.int32)
  for slot,index in enumerate(chosen):
   a,b=map(int,corpus.offsets[index:index+2]);xi,yi=shifted(corpus.ids[a:b],corpus.mask[a:b]);x[slot,:len(xi)]=xi;y[slot,:len(yi)]=yi
  value,g=one(mx.array(x),mx.array(y));g=tree_map(lambda v:v.astype(mx.float32),g)
  total=g if total is None else tree_map(lambda a,b:a+b,total,g);mx.eval(total,value);loss+=float(value.item());widths.append(width)
 total=tree_map(lambda v:v/len(indices),total);mx.eval(total)
 return loss/len(indices),total,dict(actual_records=len(indices),microbatch=microbatch,widths=widths,optimizer_updates=0)
