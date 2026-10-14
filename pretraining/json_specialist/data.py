"""One extraction envelope, with output-only loss and strict JSON handling."""
import json
import numpy as np

RULES = ('Use only the supplied context. Return only JSON matching the schema. '
         'Use null for unknown values only where the schema permits null.')

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)

def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)

def prompt(record):
    # Metadata/evidence/gold must never enter model input.
    # Canonicalize nested schema order so JSONL save/reload cannot change input IDs.
    schema = strict_json(canonical(record['schema']))
    envelope = {'task': 'extract', 'rules': RULES, 'schema': schema,
                'question': record['instruction'], 'document': record['document']}
    return json.dumps(envelope, ensure_ascii=False, separators=(',', ':')) + '\nOUTPUT_JSON:\n'

def serialize(tokenizer, record):
    prefix = tokenizer.encode(prompt(record), add_special_tokens=False).ids
    answer = tokenizer.encode(canonical(record['target']), add_special_tokens=False).ids
    eos = tokenizer.token_to_id('<|doc_end|>')
    if eos != 1:
        raise ValueError('Unexpected frozen tokenizer EOS')
    if len(prefix) > 1536 or len(answer) + 1 > 512:
        raise ValueError('Record exceeds reserved context; no truncation permitted')
    if any(i < 4 for i in prefix + answer):
        raise ValueError('Ordinary text emitted structural token IDs')
    ids = np.asarray(prefix + answer + [eos], dtype=np.int32)
    mask = np.asarray([False] * len(prefix) + [True] * (len(answer) + 1))
    if tokenizer.decode(prefix, skip_special_tokens=False) != prompt(record):
        raise ValueError('Prompt does not round-trip')
    if tokenizer.decode(answer, skip_special_tokens=False) != canonical(record['target']):
        raise ValueError('Target does not round-trip')
    return ids, mask
