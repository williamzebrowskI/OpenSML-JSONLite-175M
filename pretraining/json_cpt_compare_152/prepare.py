"""Freeze the unchanged142 bank and rescore its verified cached control."""
import collections,copy,hashlib,importlib.metadata,shutil
from jsonschema import Draft202012Validator
from .common import *
from json_foundation_142 import runtime as R,scoring as S
from json_foundation_142.run import aggregate

def prepare():
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    assert not any(a['run'].exists() for a in ARMS.values()),'Do not prepare over initialized SFT arms'
    protected();old=load(CONTROL/'session.json');m=load(SOURCE/'manifest.json')
    assert old['base_sha256']==BASE_SHA and m['base_sha256']==BASE_SHA
    assert {k:v for k,v in old['plan'].items() if k!='retention'}=={k:v for k,v in PLAN.items() if k!='retention'}
    for name in ['mlx','numpy','tokenizers','jsonschema']:assert importlib.metadata.version(name)==old['versions'][name],name
    for name in CRITICAL:
        path=ROOT/'pretraining'/name
        assert sha(path)==old['files'][str(path.relative_to(ROOT))],'Original control implementation changed: '+name
    for name,h in m['files'].items():
        assert sha(SOURCE/name)==h,'Original data changed: '+name
        if (DATA/name).exists():assert sha(DATA/name)==h,'Prepared data changed: '+name
        else:shutil.copy2(SOURCE/name,DATA/name)
    shutil.copy2(SOURCE/'manifest.json',DATA/'original-142-manifest.json')
    train=rows(DATA/'train.jsonl');dev=rows(DATA/'development.jsonl');reserved=rows(DATA/'reserved.jsonl')
    fit=rows(DATA/'fit_probe.jsonl');batches=load(DATA/'batches.json')
    assert len(train)==36000 and len(dev)==448 and len(reserved)==768 and len(fit)==64 and len(batches)==563
    order=[i for b in batches for i in b]
    assert sorted(order)==list(range(36000)) and all(0<len(b)<=64 and len(b)%2==0 for b in batches)
    for group in [train,dev,reserved,fit]:assert len({r['id'] for r in group})==len(group)
    sources={normalized(r['document']) for r in train}
    for group in [dev,reserved]:assert not sources&{normalized(r['document']) for r in group}
    excluded={}
    for p in BENCHMARKS:
        overlap=sources&{normalized(source_text(r)) for r in rows(p)}
        assert not overlap,'Benchmark source overlap: '+str(p)
        excluded[str(p.relative_to(ROOT))]=dict(sha256=sha(p),source_overlap=0)
    train_groups={r['source_group'] for r in train}
    assert not train_groups&{r['source_group'] for r in dev+reserved}
    enc=R.Encoding();maximum=0;mask_checked=0
    for i,r in enumerate(train):
        assert isinstance(r['expected'],dict)
        Draft202012Validator(r['schema']).validate(r['expected'])
        x,y=enc.row(r,training=True);prefix=enc.prefix(r['document'],r['schema'],r['instruction'])
        target=enc.tokenizer.assert_roundtrip(dumps(r['expected']))+[3]
        assert y.tolist()==[-100]*(len(prefix)-1)+target and x.tolist()==(prefix+target)[:-1]
        assert len(prefix)+PLAN['max_tokens']<=2048
        maximum=max(maximum,len(x));mask_checked+=1
        if (i+1)%6000==0:print(f'[SFT152 preparation] validated {i+1}/36000 schemas, masks and budgets',flush=True)
    for r in dev+reserved+fit:
        Draft202012Validator(r['schema']).validate(r['expected']);enc.row(r)
    control_receipts={}
    for step in PLAN['eval_steps']:
        for panel,requests in [('development',dev),('fit_probe',fit)]:
            source=CONTROL/'evaluations'/f'{panel}_{step:07d}.json';cached=load(source)
            identity=cached['identity']
            assert cached['status']=='complete' and len(cached['outputs'])==len(requests)
            assert identity['rows_sha256']==hashlib.sha256(dumps(requests).encode()).hexdigest()
            assert identity['runtime_sha256']==sha(Path(R.__file__)) and identity['scorer_sha256']==sha(Path(S.__file__))
            assert identity['max_output_tokens']==384 and identity['reference_in_input'] is False and identity['raw_greedy'] is True
            for r,o in zip(requests,cached['outputs']):
                assert r['id']==o['id'] and o['checks']==S.score(r['document'],r['schema'],r['expected'],o['text'],o['stop_reason'])
            actual=aggregate(requests,cached['outputs']);stored=copy.deepcopy(cached['metrics']);stored['all'].pop('output_nll',None)
            assert actual==stored,'Cached control metrics do not rescore identically'
            dest=REPORT/'control'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            control_receipts[str(dest.relative_to(ROOT))]=sha(dest)
    components=dict(collections.Counter(r['component'] for r in train));families=dict(collections.Counter(r['family'] for r in train))
    record=dict(status='passed',training_rows=36000,development_rows=448,reserved_rows=768,fit_rows=64,
        gold_schemas_validated=36000,assistant_masks_checked=mask_checked,max_train_sequence=maximum,components=components,families=families,
        exact_original_batch_order=True,epochs=1,benchmark_source_exclusions=excluded,fit_scope='Intentionally sampled TRAIN-fit cases, not generalization',
        selection_scope='Frozen original142 development selection; same563 endpoint reported separately. No benchmark-based selection or targets.')
    atomic_json(REPORT/'data-audit.json',record)
    atomic_json(DATA/'manifest.json',dict(format='matched-cpt152-sft142-v1',plan=PLAN,source_manifest_sha256=sha(SOURCE/'manifest.json'),
        original_control_contract_sha256=sha(CONTROL/'session.json'),files=dict(m['files'],**{'original-142-manifest.json':sha(DATA/'original-142-manifest.json')}),
        control_receipts=control_receipts,arms={k:dict(parent=str(a['parent'].relative_to(ROOT)),sha256=a['sha256'],added_tokens=a['added_tokens']) for k,a in ARMS.items()},
        treatment='Starting weight payload only; identical schema/query/JSON training targets, order, seed, optimizer reset and evaluation protocol.'))
    atomic_json(REPORT/'control-verification.json',dict(status='passed',original_contract_sha256=sha(CONTROL/'session.json'),
        numerical_recipe_unchanged=True,training_files_identical=True,training_engine_unchanged=True,decoder_scorer_unchanged=True,
        runtime_versions_identical=True,cached_panels_rescored=12,receipts=control_receipts,
        intermediate_weight_scope='Control100–500 cached outputs reused; deleted intermediate weights not recreated. Kept563 control weight verified independently.'))
    print('[SFT152 preparation] Frozen unchanged142 data; all36000 masks/golds checked; cached controls rescored',flush=True)

if __name__=='__main__':prepare()
