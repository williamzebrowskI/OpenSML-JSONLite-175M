"""Explicit supported-schema boundary for the experimental JSON extractor."""
from jsonschema import validators,FormatChecker
from .ordered_serialization import prompt

KEYWORDS={'$schema','title','description','type','properties','required','additionalProperties','items','enum','format'}
KINDS={'object','array','string','integer','number','boolean','null'}

def validate_request(schema,document,instruction,tokenizer,style):
    if not isinstance(document,str) or not document.strip():raise ValueError('Nonempty source text is required')
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('An extraction instruction is required')
    if not isinstance(schema,dict) or schema.get('type')!='object':raise ValueError('This extractor supports object-root schemas')
    def visit(node,depth=0):
        if depth>4 or not isinstance(node,dict):raise ValueError('Unsupported schema depth or schema node')
        if set(node)-KEYWORDS:raise ValueError('Unsupported schema keyword: '+','.join(sorted(set(node)-KEYWORDS)))
        kind=node.get('type');kinds=kind if isinstance(kind,list) else [kind]
        if not kinds or any(v not in KINDS for v in kinds):raise ValueError('Explicit supported schema types required')
        if isinstance(kind,list) and (len(kind)!=2 or 'null' not in kind or 'object' in kind or 'array' in kind):raise ValueError('Only nullable scalar type unions are supported')
        if 'format' in node and node['format'] not in FormatChecker.checkers:raise ValueError('Unknown schema format')
        if 'object' in kinds:
            properties=node.get('properties',{})
            if not isinstance(properties,dict) or not 1<=len(properties)<=8:raise ValueError('Expected 1–8 object properties')
            if node.get('additionalProperties') is not False:raise ValueError('Explicit additionalProperties: false is required')
            for value in properties.values():visit(value,depth+1)
        if 'array' in kinds:visit(node.get('items'),depth+1)
    visit(schema);v=validators.validator_for(schema,default=None) if '$schema' in schema else validators.validator_for(schema)
    if v is None:raise ValueError('Unsupported JSON Schema draft')
    v.check_schema(schema)
    row=dict(document=document,schema=schema,instruction=instruction,_prompt_style=style)
    ids=tokenizer.encode(prompt(row),add_special_tokens=False).ids
    if len(ids)>1536 or any(i<4 for i in ids):raise ValueError('Request exceeds the supported prompt budget; no truncation')
    return row
