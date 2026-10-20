"""Document-aligned2048-token rows; exact token accounting and explicit padding."""
import json,os
from pathlib import Path
import numpy as np
from .common import DATA,PLAN,dumps,sha,atomic_json

def packed_rows(documents,tokenizer,*,whole):
    context=PLAN['context'];buffer=[tokenizer.eos];metadata=[]
    def output():
        n=len(buffer)-1;x=np.zeros(context,np.int32);y=np.full(context,-100,np.int32)
        x[:n]=buffer[:-1];y[:n]=buffer[1:]
        return x,y,list(metadata)
    for row in documents:
        ids=tokenizer.assert_roundtrip(row['text'])+[tokenizer.eos]
        if whole and len(ids)>context:raise ValueError('Structured document exceeds context; no truncation: '+row['id'])
        if whole:
            if len(buffer)-1+len(ids)>context:
                yield output();buffer=[tokenizer.eos];metadata=[]
            buffer+=ids;metadata.append(row)
        else:
            metadata.append(row);at=0
            while at<len(ids):
                take=min(context+1-len(buffer),len(ids)-at);buffer+=ids[at:at+take];at+=take
                if len(buffer)==context+1:
                    yield output();last=buffer[-1];buffer=[last];metadata=[row] if at<len(ids) else []
    if len(buffer)>1:yield output()
