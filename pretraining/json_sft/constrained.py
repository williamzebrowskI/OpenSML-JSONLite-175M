"""Optional schema-masked MLX decoding; always validate the finished output.

Format enforcement is an inference setting, not evidence of learned accuracy.
The isolated lm-format-enforcer dependency does not alter the training venv.
"""
from functools import lru_cache
import json
import sys
from pathlib import Path
from .ordered_serialization import prompt
ROOT=Path(__file__).resolve().parents[2]
DECODER_REVISION='opensml-schema-mask-v3-stable-greedy-ties'

class DecoderContinuationError(RuntimeError):pass

@lru_cache(maxsize=1)
def install_enum_compatibility():
    """Compile finite values as JSON literals, fixing the pinned backend's mixed enums.

    The original schema is unchanged. This adapter reads schema values only,
    never extraction labels. Full validation still governs completion.
    """
    sys.path.insert(0,str(ROOT/'pretraining/data/json_sft_decoder_20261010'))
    import lmformatenforcer.jsonschemaparser as backend
    from lmformatenforcer.characterlevelparser import StringParser,UnionParser
    from jsonschema import validators,FormatChecker
    original=backend.get_parser

    class LiteralParser(StringParser):
        def get_allowed_characters(self):
            return ' \t\r\n'+super().get_allowed_characters()
        def add_character(self,char):
            if char in ' \t\r\n':return self
            return StringParser(self.target_str).add_character(char)

    def get_parser(state,node):
        if node is not None and (node.enum or 'const' in node.extras):
            schema=node.model_dump(by_alias=True,exclude_unset=True) if hasattr(node,'model_dump') else node.dict(by_alias=True,exclude_unset=True)
            schema.pop('extras',None);schema.pop('__extras__',None);schema.update(node.extras)
            validator=validators.validator_for(schema)(schema,format_checker=FormatChecker())
            values=[node.extras['const']] if 'const' in node.extras else node.enum
            parsers=[LiteralParser(json.dumps(value,ensure_ascii=True,allow_nan=False,separators=(',',':')))
                     for value in values if validator.is_valid(value)]
            if not parsers:raise ValueError('Schema has no valid finite enum/const values')
            return parsers[0] if len(parsers)==1 else UnionParser(parsers)
        return original(state,node)
    backend.get_parser=get_parser

def complete_document(text,schema):
    from json_specialist.data import strict_json
    from jsonschema import validators,FormatChecker,ValidationError
    try:value=strict_json(text)
    except (ValueError,TypeError):return False
    try:validators.validator_for(schema)(schema,format_checker=FormatChecker()).validate(value)
    except ValidationError:return False
    return True

def guarded_tokens(allowed,text,schema):
    # The backend falls back to EOS on internal errors. Never label that success.
    if 1 in allowed and not complete_document(text,schema):allowed=[i for i in allowed if i!=1]
    if not allowed:raise DecoderContinuationError('Schema decoder has no valid continuation; premature EOS rejected')
    return allowed

def greedy_allowed_token(logits,allowed):
    """Break equal-score ties by global token ID, independent of mask ordering.

    The backend constructs allowed lists from sets and may include duplicates.
    Match full-vocabulary argmax's lowest-ID tie rule without sorting that list.
    """
    import mlx.core as mx
    if not allowed:raise DecoderContinuationError('Schema decoder has no allowed tokens')
    choices=mx.array(allowed,dtype=mx.int32)
    scores=logits[choices]
    token=int(mx.min(mx.where(scores==mx.max(scores),choices,logits.shape[0])).item())
    if token==logits.shape[0]:raise DecoderContinuationError('Schema decoder has non-finite token scores')
    return token

@lru_cache(maxsize=1)
def tokenizer_data():
    sys.path.insert(0,str(ROOT/'pretraining/data/json_sft_decoder_20261010'))
    from lmformatenforcer import TokenEnforcerTokenizerData
    from .corpus import load_tokenizer
    tok=load_tokenizer()
    tokens=[(i,tok.decode([i],skip_special_tokens=False),False) for i in range(4,tok.get_vocab_size())]
    return TokenEnforcerTokenizerData(tokens,lambda ids:tok.decode(ids,skip_special_tokens=False),1,False,tok.get_vocab_size())

def generate(model,tokenizer,row,limit=512):
    import mlx.core as mx
    data=tokenizer_data()
    install_enum_compatibility()
    from lmformatenforcer import JsonSchemaParser,TokenEnforcer
    enforcer=TokenEnforcer(data,JsonSchemaParser(row['schema']))
    ids=tokenizer.encode(prompt(row),add_special_tokens=False).ids
    if len(ids)>1536 or any(i<4 for i in ids):raise ValueError('Invalid input; no truncation')
    inputs=mx.array([ids],dtype=mx.int32);cache=None;out=[];reason='token_limit'
    for _ in range(min(limit,model.cfg.max_seq_len-len(ids))):
        logits,cache=model.logits(inputs,cache)
        allowed=enforcer.get_allowed_tokens(out).allowed_tokens
        try:
            allowed=guarded_tokens(allowed,tokenizer.decode(out,skip_special_tokens=False),row['schema'])
            token=greedy_allowed_token(logits[0,-1],allowed)
        except DecoderContinuationError:
            reason='decoder_error';break
        if token==1:reason='document_end';break
        out.append(token);inputs=mx.array([[token]],dtype=mx.int32)
    return dict(text=tokenizer.decode(out,skip_special_tokens=False),output_tokens=len(out),stop_reason=reason,decoder_revision=DECODER_REVISION)
