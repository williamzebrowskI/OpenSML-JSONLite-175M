"""Explicit, resumable inference-only runner; no training or promotion."""
import argparse
import importlib.metadata
import json
import time
from .common import *
from . import scoring as S
from sml_v1.common import atomic_json,lock


def verify():
    manifest=json.loads((DATA/'manifest.json').read_text())
    assert manifest['count']==384 and manifest['root']=='object'
    for name,h in manifest['files'].items():assert sha(DATA/name)==h,'Eval data changed'
    assert sha(Path(__file__).with_name('build.py'))==manifest['builder_sha256'],'Builder changed'
    requests=read_rows(DATA/'requests.jsonl');refs=read_rows(DATA/'references.jsonl')
    assert [r['id'] for r in requests]==[r['id'] for r in refs]
    ready=json.loads((REPORT/'ready.json').read_text())
    assert ready['status']=='passed' and ready['manifest_sha256']==sha(DATA/'manifest.json')
    for path,h in ready['source_files'].items():assert sha(ROOT/path)==h,'Readiness is stale: '+path
    return manifest,requests,{r['id']:r for r in refs}


def generate_one(model,tokenizer,source_text,schema):
    # The signature intentionally excludes references, IDs and annotation metadata.
    from sft.data import serialize
    from json_restart_130.runtime import generate
    prefix=serialize(tokenizer,[dict(role='user',content=content(source_text,schema))],generation=True)[0].tolist()
    assert len(prefix)+256<=2048,'No truncation'
    return generate(model,tokenizer,prefix,256)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--model',choices=['130','140','release','both','all'],default='both')
    parser.add_argument('--smoke',action='store_true',help='32 paired-schema cases; reported separately, never a full-suite score')
    args=parser.parse_args();manifest,requests,refs=verify()
    names=['130','140'] if args.model=='both' else ['130','140','release'] if args.model=='all' else [args.model]
    if args.smoke:
        groups={r['source_group'] for r in requests if '-01-' in r['id']}
        requests=[r for r in requests if r['source_group'] in groups]
        assert len(requests)==32
    if not args.run:
        print(dumps(dict(training=False,evaluation_started=False,models=names,cases_each=len(requests),results=str(REPORT/'results'))));return
    import mlx.core as mx
    from sml_v1.tokenization import Tokenizer
    from json_restart_130.runtime import model_at
    assert importlib.metadata.version('mlx')=='0.32.3','Use the evaluated .venv-jsonaudit119 runtime'
    with lock(LOCK):
        mx.set_default_device(mx.gpu);mx.set_cache_limit(4*1024**3)
        tokenizer=Tokenizer(TOKENIZER)
        outputs_by_model={}
        for name in names:
            weight,h=MODELS[name];path=ROOT/weight;assert sha(path)==h,'Pinned model changed'
            protocol=dict(suite_sha256=sha(DATA/'manifest.json'),requests_sha256=sha(DATA/'requests.jsonl'),
                reference_sha256=sha(DATA/'references.jsonl'),scorer_sha256=sha(Path(S.__file__)),
                runner_sha256=sha(Path(__file__)),instruction=INSTRUCTION,model=name,weight_sha256=h,
                max_output_tokens=256,context=2048,temperature=0,repetition_penalty=1,
                raw_greedy=True,model_calls_per_request=1,truncation=False,repair=False,assembly=False,
                reference_inference_input=False,serialization='native-sft-chat-v1',smoke=args.smoke,
                python=__import__('platform').python_version(),mlx=importlib.metadata.version('mlx'),tokenizer_sha256=tokenizer.fingerprint)
            directory=REPORT/('smoke-results' if args.smoke else 'results');directory.mkdir(parents=True,exist_ok=True)
            identity=directory/(name+'-identity.json');predictions=directory/(name+'-predictions.jsonl');result_path=directory/(name+'-result.json')
            if identity.exists():assert json.loads(identity.read_text())==protocol,'Mismatched existing run; refusing to mix results'
            else:atomic_json(identity,protocol)
            outputs=read_rows(predictions) if predictions.exists() else []
            assert len(outputs)<=len(requests)
            for row,o in zip(requests,outputs):
                assert row['id']==o['id']
                assert o['checks']==S.score(row['source_text'],row['schema'],refs[row['id']]['expected'],o['text'],o['stop_reason'])
            model=None;began=time.monotonic()
            try:
                if len(outputs)<len(requests):model=model_at(path);model.eval()
                with predictions.open('a') as stream:
                    for row in requests[len(outputs):]:
                        raw=generate_one(model,tokenizer,row['source_text'],row['schema'])
                        record=dict(id=row['id'],**raw,checks=S.score(row['source_text'],row['schema'],refs[row['id']]['expected'],raw['text'],raw['stop_reason']))
                        outputs.append(record);stream.write(dumps(record)+'\n');stream.flush()
                        if len(outputs)%24==0 or len(outputs)==len(requests):
                            m=S.aggregate(requests[:len(outputs)],outputs)
                            print('[JSON object eval] '+dumps(dict(model=name,completed=len(outputs),total=len(requests),correct=m['all']['counts']['exact'],schema_valid=m['all']['schema_valid_rate'])),flush=True)
                result=dict(status='complete',protocol=protocol,count=len(outputs),metrics=S.aggregate(requests,outputs),predictions_sha256=sha(predictions),
                    elapsed_seconds_this_invocation=time.monotonic()-began,scope=manifest['scope'])
                atomic_json(result_path,result);outputs_by_model[name]=outputs
            except KeyboardInterrupt:
                atomic_json(result_path,dict(status='interrupted',protocol=protocol,completed=len(outputs),total=len(requests)))
                print('Interrupted; the same command resumes saved predictions.',flush=True);return
            finally:
                del model;mx.clear_cache()
            assert sha(path)==h
        if len(outputs_by_model)>1:
            a,b=names[:2];left=outputs_by_model[a];right=outputs_by_model[b]
            fixed=[r['id'] for r,x,y in zip(requests,left,right) if not x['checks']['exact'] and y['checks']['exact']]
            lost=[r['id'] for r,x,y in zip(requests,left,right) if x['checks']['exact'] and not y['checks']['exact']]
            atomic_json(directory/'comparison.json',dict(baseline=a,candidate=b,fixed=fixed,lost=lost,net=len(fixed)-len(lost),scope='Paired cases on this suite only; shared sources are correlated.'))
        print('[JSON object eval complete] '+dumps(dict(models=names,cases_each=len(requests),results=str(directory),smoke=args.smoke)),flush=True)


if __name__=='__main__':main()
