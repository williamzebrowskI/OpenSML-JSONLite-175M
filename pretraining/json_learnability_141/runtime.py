import numpy as np
from .common import *
from json_restart_130.runtime import model_at, batch, loss, gradients, generate


class Encoding:
    def __init__(self):
        from sml_v1.tokenization import Tokenizer
        self.tokenizer=Tokenizer(TOKENIZER)

    def prefix(self,document,schema,instruction=INSTRUCTION):
        from sft.data import serialize
        user=instruction+'\n\n'+dumps(dict(source_text=document,schema=schema))
        return serialize(self.tokenizer,[dict(role='user',content=user)],generation=True)[0].tolist()

    def row(self,r,training=False):
        p=self.prefix(r['document'],r['schema'],r['instruction']);t=self.tokenizer.assert_roundtrip(dumps(r['expected']))+[3]
        assert len(p)+PLAN['max_tokens']<=PLAN['context'] and len(t)<=PLAN['max_tokens'],'No truncation: '+r['id']
        assert not training or len(p)+len(t)<=PLAN['train_max_sequence'],'Training context overflow'
        return np.array((p+t)[:-1],np.int32),np.array([-100]*(len(p)-1)+t,np.int32)


def generate_one(model,enc,document,schema,instruction):
    # Only source, schema and a generic instruction. No targets or metadata.
    prefix=enc.prefix(document,schema,instruction)
    assert len(prefix)+PLAN['max_tokens']<=PLAN['context']
    return generate(model,enc.tokenizer,prefix,PLAN['max_tokens'])
