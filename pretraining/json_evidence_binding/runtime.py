"""Padding-only JSON-036 optimization; batch order and CE stay unchanged.

Each existing 64-slot batch retains its exact sequential record order. Padding
can shrink per microbatch, and measured grouping can use 2, 4 or 8 slots. No
length sorting, repacking, resampling, optimizer or cursor change occurs here.
"""
from dataclasses import dataclass
import math
import numpy as np
from sft.data import shifted

POLICIES = ('global_power2', 'micro_power2', 'micro_multiple256', 'micro_multiple128')
RUNTIME_FORMAT = 'json036-exact-order-micro-padding-v1'


@dataclass(frozen=True)
class Microbatch:
    slot_start: int
    indices: tuple
    inputs: np.ndarray
    labels: np.ndarray

    @property
    def actual_records(self): return len(self.indices)

    @property
    def width(self): return self.inputs.shape[1]


def sequence_width(length, policy, *, context_limit=2048):
    if type(length) is not int or length < 0 or type(context_limit) is not int or context_limit < 16:
        raise ValueError('Invalid sequence/context length')
    if length > context_limit:
        raise ValueError('Record exceeds context; padding optimization never truncates')
    if policy in ('global_power2', 'micro_power2'):
        width = max(16, 2**math.ceil(math.log2(max(1,length))))
    elif policy in ('micro_multiple128','micro_multiple256'):
        multiple = 128 if policy == 'micro_multiple128' else 256
        width = max(16, math.ceil(length/multiple)*multiple)
    else:
        raise ValueError('Unknown padding policy')
    # A non-power-of-two context can cap only unused trailing padding.
    return min(context_limit,width)


def _indices(corpus, indices, size, microbatch):
    array=np.asarray(indices)
    if type(size) is not int or size != 64 or type(microbatch) is not int or microbatch not in (2,4,8):
        raise ValueError('Keep batch64 and a measured microbatch2/4/8')
    if array.ndim != 1 or not 0 < len(array) <= size or array.dtype.kind not in 'iu':
        raise ValueError('Expected 1..64 integer record indices')
    if np.any(array < 0) or np.any(array >= len(corpus)):
        raise ValueError('Record index outside corpus')
    return array.astype(np.int64,copy=False)


def microbatches(corpus, indices, *, policy, size=64, microbatch=2, context_limit=2048):
    """Yield the same ordered 64 slots, grouped into measured microbatches.

    The final partial batch still has 64 slots. Empty slots contain zeros and
    -100 labels, and the caller divides gradient sums by real record count.
    """
    indices=_indices(corpus,indices,size,microbatch)
    if policy not in POLICIES:
        raise ValueError('Unknown padding policy')
    lengths=np.asarray(corpus.lengths[indices],np.int64)
    if np.any(lengths < 1):
        raise ValueError('Expected nonempty shifted records')
    global_width=sequence_width(int(lengths.max()),'global_power2',context_limit=context_limit)
    for start in range(0,size,microbatch):
        chosen=indices[start:start+microbatch]
        longest=int(np.max(corpus.lengths[chosen])) if len(chosen) else 0
        width=global_width if policy=='global_power2' else sequence_width(longest,policy,context_limit=context_limit)
        x=np.zeros((microbatch,width),np.int32)
        y=np.full((microbatch,width),-100,np.int32)
        for slot,index in enumerate(chosen):
            a,b=map(int,corpus.offsets[index:index+2])
            xi,yi=shifted(corpus.ids[a:b],corpus.mask[a:b])
            if len(xi) != int(corpus.lengths[index]) or len(xi)>width:
                raise ValueError('Packed lengths/offsets disagree or record would truncate')
            x[slot,:len(xi)]=xi;y[slot,:len(yi)]=yi
        yield Microbatch(start,tuple(map(int,chosen)),x,y)


def batch_profile(corpus, indices, *, policy, microbatch=2):
    pieces=list(microbatches(corpus,indices,policy=policy,microbatch=microbatch))
    return dict(format=RUNTIME_FORMAT,policy=policy,slots=64,microbatch=microbatch,
        actual_records=sum(piece.actual_records for piece in pieces),
        record_indices=[index for piece in pieces for index in piece.indices],
        widths=[piece.width for piece in pieces],
        model_input_positions=sum(piece.inputs.size for piece in pieces),
        nonpadding_input_positions=sum(int(corpus.lengths[index]) for piece in pieces for index in piece.indices),
        supervised_output_eos_positions=sum(int(np.sum(piece.labels != -100)) for piece in pieces))


def iter_microbatches(corpus, indices, *, size=64, microbatch=2, policy='micro_multiple128'):
    """Trainer-facing arrays; pass the measured policy explicitly from plan."""
    for piece in microbatches(corpus,indices,policy=policy,size=size,microbatch=microbatch):
        yield piece.inputs,piece.labels


def accumulate_gradients(one, corpus, indices, *, policy, microbatch=2):
    """Production-equivalent sequential FP32 accumulation, no optimizer update.

    ``one`` is nn.value_and_grad(model, the unchanged output/EOS record CE sum).
    The returned gradients and loss are divided by the actual number of records
    exactly as in JSON-035. Gradient clipping/AdamW/counters remain the caller's.
    """
    import mlx.core as mx
    from mlx.utils import tree_map
    indices=_indices(corpus,indices,64,microbatch)
    total=None;loss_total=0.;widths=[]
    for piece in microbatches(corpus,indices,policy=policy,microbatch=microbatch):
        value,gradients=one(mx.array(piece.inputs),mx.array(piece.labels))
        gradients=tree_map(lambda value:value.astype(mx.float32),gradients)
        total=gradients if total is None else tree_map(lambda left,right:left+right,total,gradients)
        mx.eval(total,value)
        loss_total+=float(value.item());widths.append(piece.width)
    total=tree_map(lambda value:value/len(indices),total)
    mx.eval(total)
    return loss_total/len(indices),total,dict(policy=policy,widths=widths,
        actual_records=len(indices),slots=64,microbatch=microbatch,optimizer_updates=0)
