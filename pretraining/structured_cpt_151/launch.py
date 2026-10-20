"""Prepare/verify before --run; indefinite production, separate two-update smoke."""
import argparse,datetime,os,shutil,sys,uuid
from pathlib import Path
from .common import *
from .prepare import contract
from sml_v1.common import lock,fingerprint

NETWORK=LOCK.parent

def prepared():
    p=load(REPORT/'prepared.json')
    if load(REPORT/'unit-checks.json')['status']!='passed':raise ValueError('Unit checks must pass before launch')
    if p['contract']!=contract():raise ValueError('Prepared code/data changed; rerun preparation and verification')
    if p['validation_sha256']!=sha(DATA/'validation.npz'):raise ValueError('Fixed validation changed')
    if p['exclusion_sha256']!=sha(DATA/'exclusion-guard.json'):raise ValueError('Source exclusion changed')
    return p

def stage(job,checkpoint,receipt):
    source=job/'input';source.mkdir(parents=True)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    for name in ['sml_v1','structured_cpt_151']:
        shutil.copytree(ROOT/'pretraining'/name,source/name,ignore=ignore)
    shutil.copytree(TOKENIZER,source/'tokenizer');shutil.copytree(DATA/'vendor',source/'vendor',ignore=ignore)
    for name in ['validation.npz','exclusion-guard.json']:shutil.copy2(DATA/name,source/name)
    shutil.copy2(CORPUS,source/'corpus.json')
    from sml_v1.checkpoint_bundle import resolve_bundle
    weights=Path(resolve_bundle(checkpoint))
    shutil.copytree(weights.parent,source/'checkpoint')
    atomic_json(source/'config.json',dict(plan=PLAN,contract=receipt['contract'],validation_sha256=receipt['validation_sha256']))
    return source

def run_job(cluster,output,checkpoint,receipt,*,smoke=False):
    from five_mac_cluster import PYTHON
    job=output/'jobs'/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]);job.mkdir(parents=True)
    cluster.preflight(job)
    for physical in range(5):
        code="import subprocess,os; rows=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True).splitlines(); busy=[r for r in rows if '-m structured_cpt_151.worker' in r and ' -c ' not in r]; assert not busy,busy; print('CPT workers idle')"
        cluster.remote(physical,[PYTHON,'-c',code])
    source=stage(job,checkpoint,receipt);cluster.stage(source)
    worker=['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','structured_cpt_151.worker',
            '--input',str(source),'--output',str(output),'--job-dir',str(job)]
    if smoke:worker+=['--smoke','--steps','1']
    cluster.launch(job,source,worker,timeout=1200 if smoke else None,stop_dir=output)
    result=load(job/'result.json')
    if result['status']!='stopped':raise ValueError('Workers did not acknowledge a clean save/stop')
    return result,job

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--run',action='store_true');g.add_argument('--verify',action='store_true')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    receipt=prepared()
    print(dumps(dict(mode='verify' if a.verify else ('run' if a.run else 'plan'),base='pretrained132005',
        cluster='five Macs, existing JACCL/RDMA',token_limit=None,time_limit=None,stop='Ctrl+C -> complete current update -> save all state -> stop',
        save_every_real_tokens=PLAN['save_every_tokens'],mixture={k:v/2 for k,v in MIX.items()},output=str(RUN))),flush=True)
    if not(a.run or a.verify):return
    protected()
    sys.path.insert(0,str(ROOT/'pretraining/scripts'))
    from five_mac_cluster import Cluster
    cluster=Cluster(NETWORK)
    with lock(LOCK):
        if a.verify:
            smoke=REPORT/'distributed-smoke'/uuid.uuid4().hex[:8]
            first,j1=run_job(cluster,smoke,BASE,receipt,smoke=True)
            second,j2=run_job(cluster,smoke,smoke/'latest.json',receipt,smoke=True)
            assert first['updates']==1 and second['updates']==2
            assert first['step']==132006 and second['step']==132007 and second['added_tokens']>first['added_tokens']>0
            protected()
            atomic_json(REPORT/'five-mac-verification.json',dict(status='passed',contract_sha256=fingerprint(receipt['contract']),
                first=first,resumed=second,job_dirs=[str(j1),str(j2)],scope='Actual175M model/FP32 optimizer, global token-weighted update, five-rank matching replica digests, streamed input, transactional save and resume.'))
            print('[verified] Five-rank update, checkpoint and resumed update passed. No overnight training started.',flush=True)
        else:
            proof=load(REPORT/'five-mac-verification.json')
            if proof['status']!='passed' or proof['contract_sha256']!=fingerprint(receipt['contract']):raise ValueError('Run five-Mac verification for these exact inputs first')
            output=RUN;output.mkdir(parents=True,exist_ok=True)
            with lock(output/'.launcher.lock'):
                exists=(output/'latest.json').exists()
                if exists!=a.resume:raise ValueError('Use --run --resume for this run\'s saved progress; omit --resume for the first launch')
                if a.resume:(output/'STOP').unlink(missing_ok=True)
                elif (output/'STOP').exists():raise ValueError('Unexpected STOP in an unstarted output')
                run_job(cluster,output,output/'latest.json' if exists else BASE,receipt)

if __name__=='__main__':main()
