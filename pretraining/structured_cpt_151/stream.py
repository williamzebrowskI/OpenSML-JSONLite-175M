"""Resume original HF cursors; generate new whole documents without a finite bank."""
import copy,hashlib
from collections import Counter
import numpy as np
from sml_v1.stream import StreamingTokens,stream_manifest
from sml_v1.corpus import content_key
from .common import PLAN,MIX,NEW_WEIGHTS,text_key
from .synthetic import document

class GuardedOriginal(StreamingTokens):
    def __init__(self,*args,guard,**kwargs):
        self.guard=guard
        super().__init__(*args,**kwargs)
        if self.shuffle:
            for state in self.shuffle['sources'].values():
                if any(text_key(t) in guard for t in state['documents']):
                    raise ValueError('Known evaluation text in inherited pending documents')

    def _accepted_document(self,source):
        while True:
            text=super()._accepted_document(source)
            if text_key(text) not in self.guard:return text

class WholeRows:
    def __init__(self,tokenizer,split,guard,state=None):
        self.tokenizer,self.split,self.guard=tokenizer,split,guard
        self.state=copy.deepcopy(state) if state else {
            k:dict(index=0,pending=None,documents=0,rejected=0,features={}) for k in NEW_WEIGHTS}

    def next_ids(self,label):
        st=self.state[label]
        if st['pending'] is not None:
            ids=st['pending'];st['pending']=None;return ids
        while True:
            row=document(label,self.split,st['index']);st['index']+=1
            if row['text_key'] in self.guard or row['prose_key'] in self.guard:
                raise ValueError('Known evaluation text overlaps new document')
            ids=self.tokenizer.assert_roundtrip(row['text'])+[self.tokenizer.eos]
            if len(ids)>PLAN['context']:
                st['rejected']+=1;continue
            st['documents']+=1
            features=Counter(st['features']);p=row['provenance']
            features.update([f'depth:{p["depth"]}',f'items:{p["items"]}',f'layout:{p["layout"]}',
                             f'correction:{p["correction"]}',f'family:{p.get("family","order")}'])
            st['features']=dict(features)
            return ids

    def row(self,label):
        buf=[self.tokenizer.eos]
        while True:
            ids=self.next_ids(label)
            if len(buf)-1+len(ids)>PLAN['context']:
                self.state[label]['pending']=ids;break
            buf.extend(ids)
            if len(buf)==PLAN['context']+1:break
        n=len(buf)-1
        if n<1:raise ValueError('Empty structured row')
        x=np.zeros(PLAN['context'],np.int32);y=np.full(PLAN['context'],-100,np.int32)
        x[:n]=buf[:-1];y[:n]=buf[1:]
        return x,y

    def state_dict(self):return copy.deepcopy(self.state)

class MixedStream:
    def __init__(self,corpus,base_recipe,tokenizer,base_cursor,guard,state=None):
        manifest=stream_manifest(corpus,tokenizer,base_recipe['stream'])
        self.original=GuardedOriginal(manifest,tokenizer,'train',PLAN['context'],
                                     state['original'] if state else base_cursor,guard=guard)
        self.structured=WholeRows(tokenizer,'train',guard,state['structured'] if state else None)
        self.offsets=dict(state['offsets']) if state else {k:0 for k in MIX}
        if set(self.offsets)!=set(MIX):raise ValueError('Incompatible mixed cursor')

    def row(self):
        label=min(MIX,key=lambda s:self.offsets[s]/MIX[s])
        x,y=(self.structured.row(label) if label in NEW_WEIGHTS else self.original.row(label))
        self.offsets[label]+=int(np.count_nonzero(y!=-100))
        return x,y

    def global_batch(self,accum,batches):
        rows=[[self.row() for _ in range(sum(batches))] for _ in range(accum)]
        return np.stack([np.stack([np.stack([r[a] for r in step]) for step in rows]) for a in (0,1)])

    def state_dict(self):
        return dict(original=self.original.state_dict(),structured=self.structured.state_dict(),offsets=dict(self.offsets))

    def close(self):self.original.close()
