import hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def load(p):
 with Path(p).open() as stream:return [json.loads(l) for l in stream if l.strip()]

def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)

def prompt(document,schema):
 # Gold values, offsets, intent, provenance, and key-mapping metadata NEVER enter this function.
 return ('Extract information from the supplied source_text according to the supplied JSON Schema. '
         'Return exactly one JSON object, with no explanation or Markdown. Copy values exactly, preserving case and spelling; '
         'do not normalize dates or numbers. Include every required key and no other keys. '
         'For string arrays, include all matching source spans in source order, or [] when absent. '
         'For nullable strings, use the exact source span or null when absent.\n\n'+
         dumps(dict(source_text=document,schema=schema)))
