"""Freeze provenance and small validation panels; training remains streamed."""
import argparse,copy,gzip,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from .common import *
from .stream import WholeRows
from .synthetic import document
from sml_v1.common import fingerprint
from sml_v1.stream import stream_manifest,StreamingTokens
from sml_v1.tokenization import Tokenizer

def exclusions():
    # Only source text is hashed; no labels or model predictions enter generation.
    paths=set()
    for folder in [ROOT/'pretraining/data',ROOT/'reports/model-design']:
        for name in ['requests.jsonl','development.jsonl','reserved.jsonl','original1444.jsonl']:
            paths.update(folder.glob('**/'+name))
    massive=ROOT/'pretraining/data/json_019_sources_20261011/massive-en-US.jsonl'
    if massive.exists():paths.add(massive)
    keys=set();receipts={}
    for path in sorted(paths):
        receipts[str(path.relative_to(ROOT))]=sha(path)
        for line in path.open():
            row=json.loads(line)
            if path==massive and row.get('partition')=='train':continue
            for key in ['source_text','document','utt']:
                value=row.get(key)
                if isinstance(value,str) and value.strip():keys.add(text_key(value))
    if len(keys)<1000:raise ValueError('Missing known evaluation exclusions')
    return dict(source_keys=sorted(keys),input_files=receipts,
                scope='Exact normalized source exclusion. Inherited web pretraining cannot be certified free of every public benchmark.')

def contract():
    dirs=[ROOT/'pretraining/structured_cpt_151',ROOT/'pretraining/sml_v1']
    modules={str(p.relative_to(ROOT)):sha(p) for d in dirs for p in sorted(d.glob('*.py'))}
    for name in ['five_mac_cluster.py','five_mac_transport.py','run_cpt_151_structured_cluster.sh']:
        p=ROOT/'pretraining/scripts'/name;modules[str(p.relative_to(ROOT))]=sha(p)
    vendor=DATA/'vendor'
    modules.update({str(p.relative_to(ROOT)):sha(p) for p in sorted(vendor.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})
    return dict(plan=PLAN,corpus_sha256=sha(CORPUS),tokenizer_sha256=sha(TOKENIZER/'manifest.json'),modules=modules,
                parent_weight_sha256=BASE_SHA,parent_optimizer_sha256=BASE_OPT_SHA,
                parent_metadata_sha256=sha(Path(str(BASE)+'.json')),parent_cursor_sha256=sha(Path(str(BASE)+'.rank0.data_state.json.gz')))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validation',action='store_true');a=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);protected()
    guard=exclusions();atomic_json(DATA/'exclusion-guard.json',guard)
    tokenizer=Tokenizer(TOKENIZER);counts=Counter();samples=[];rejected=Counter()
    # Cover multiple cycles, not just tiny easy instances.
    for split in ['train','validation']:
        for label in NEW_WEIGHTS:
            for i in range(384):
                row=document(label,split,i)
                assert row['text_key'] not in guard['source_keys'] and row['prose_key'] not in guard['source_keys']
                ids=tokenizer.assert_roundtrip(row['text']);p=row['provenance']
                if len(ids)+1>PLAN['context']:rejected[f'{split}/{label}']+=1;continue
                counts.update([f'{split}/{label}',f'{split}/family:{p["family"]}',f'{split}/depth:{p["depth"]}',f'{split}/items:{p["items"]}'])
                if split=='train' and i<8:samples.append(row)
    for split in ['train','validation']:
        for label in NEW_WEIGHTS:assert counts[f'{split}/{label}']>=192,(split,label,counts)
        for n in range(1,7):assert counts[f'{split}/items:{n}']>0
    atomic_json(REPORT/'data-audit.json',dict(validated_worlds=3072,accepted_counts=dict(counts),oversize_rejected=dict(rejected),
        source_keys_excluded=len(guard['source_keys']),schema_validation=True,controlled_code_executed=True,
        limitations='Procedural shared grammars, not independently collected natural requests. Whole oversize documents rejected; accepted coverage recorded explicitly.'))
    atomic_json(REPORT/'review-samples.json',samples)
    meta=load(Path(str(BASE)+'.json'));manifest=stream_manifest(load(CORPUS),tokenizer,meta['recipe']['stream'])
    assert fingerprint(manifest)==meta['v2_contract']['data'],'Original stream contract changed'
    if a.validation:
        target=DATA/'validation.npz'
        if target.exists() and (DATA/'validation.json').exists():
            receipt=load(DATA/'validation.json');assert sha(target)==receipt['sha256']
            # Reusing a panel must not silently retain targets from an older generator.
            whole=WholeRows(tokenizer,'validation',set(guard['source_keys']))
            with np.load(target,allow_pickle=False) as frozen:
                for label in NEW_WEIGHTS:
                    for x,y in zip(frozen[label+'_x'],frozen[label+'_y']):
                        xx,yy=whole.row(label);assert np.array_equal(x,xx) and np.array_equal(y,yy),'Validation generator changed; create a new panel/experiment'
            print('[validation] reused frozen panel; structured rows reproduced exactly',flush=True)
        else:
            arrays={};stream=StreamingTokens(manifest,tokenizer,'validation',PLAN['context'])
            try:
                for label in OLD_WEIGHTS:
                    print('[validation-stream] '+label,flush=True)
                    pairs=[stream.row(label) for _ in range(PLAN['validation_rows_per_source'])]
                    arrays[label+'_x']=np.stack([p[0] for p in pairs]);arrays[label+'_y']=np.stack([p[1] for p in pairs])
            finally:stream.close()
            whole=WholeRows(tokenizer,'validation',set(guard['source_keys']))
            for label in NEW_WEIGHTS:
                pairs=[whole.row(label) for _ in range(PLAN['validation_rows_per_source'])]
                arrays[label+'_x']=np.stack([p[0] for p in pairs]);arrays[label+'_y']=np.stack([p[1] for p in pairs])
            np.savez(target,**arrays)
            atomic_json(DATA/'validation.json',dict(sha256=sha(target),rows_per_source=PLAN['validation_rows_per_source'],
                sources=list(MIX),tokenizer_sha256=tokenizer.fingerprint,
                scope='New fixed small NLL diagnostics, not the historical2.5563 panel or generated-extraction accuracy. No labels from benchmark suites.'))
    atomic_json(REPORT/'plan.json',PLAN)
    atomic_json(REPORT/'prepared.json',dict(contract=contract(),exclusion_sha256=sha(DATA/'exclusion-guard.json'),
        validation_sha256=sha(DATA/'validation.npz') if (DATA/'validation.npz').exists() else None,training_started=False))
    print('[prepared] Open-ended five-Mac data/parent plan; training not started',flush=True)

if __name__=='__main__':main()
