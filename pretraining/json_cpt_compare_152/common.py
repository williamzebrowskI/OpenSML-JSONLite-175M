import copy,json
from pathlib import Path
from sml_v1.common import atomic_json,file_sha256 as sha,fingerprint
from structured_cpt_151.common import ROOT,TOKENIZER,LOCK,BASE,BASE_SHA,KEPT
from json_foundation_142.common import PLAN as ORIGINAL_PLAN

SOURCE=ROOT/'pretraining/data/sft_142_fresh_foundation_20261018'
CONTROL=ROOT/'pretraining/runs/sft_142_fresh_foundation_20261018'
DATA=ROOT/'pretraining/data/sft_152_cpt_comparison_20261019'
REPORT=ROOT/'reports/model-design/SFT-152_CPT_comparison_20261019'
CPT=ROOT/'pretraining/runs/cpt_151_structured_20261018'
PLAN=dict(copy.deepcopy(ORIGINAL_PLAN),retention='Keep every checkpoint. Matched142 control; no automatic promotion or reserved generation.')
ARMS={
 '25m':dict(label='152',parent=CPT/'step_0132213_08e411ee0483/model.safetensors',
    sha256='f12e0fa29038726cc50dc31260a49e97fb69c4452fc59182143f7c46e4ea8860',added_tokens=25010797,
    run=ROOT/'pretraining/runs/sft_152_cpt25m_foundation_20261019'),
 '100m':dict(label='153',parent=CPT/'step_0132838_224f03f16825/model.safetensors',
    sha256='57d0933d5d806e7f589a60170b5080ff46895b732be480aa953650701f4e19cc',added_tokens=100096883,
    run=ROOT/'pretraining/runs/sft_153_cpt100m_foundation_20261019')}
BENCHMARKS=[
 ROOT/'pretraining/data/jsonlite_eval_v1/requests.jsonl',
 ROOT/'reports/model-design/SFT-142_fresh_foundation_20261018/playground108-v2/requests.jsonl',
 ROOT/'reports/model-design/SFT-128_joint_extraction_20261016/release-comparison/original1444.jsonl',
 ROOT/'pretraining/data/massive_slot_list_139_20261017/requests.jsonl']
CRITICAL=[
 'json_restart_130/train.py','json_restart_130/runtime.py','json_restart_130/common.py','jsonlite_eval/common.py','json_foundation_142/common.py',
 'json_foundation_142/runtime.py','json_foundation_142/scoring.py','json_foundation_142/run.py',
 'sft/data.py','sml_v1/model.py','sml_v1/precision.py','sml_v1/checkpoint_bundle.py',
 'sml_v1/common.py','sml_v1/tokenization.py','sml_v1/fast_loss.py','sml_v1/fast_swiglu.py']

def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).open()]
def dumps(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def normalized(s):return ' '.join(s.casefold().split())
def source_text(r):return r.get('document',r.get('source_text',r.get('utt','')))

def protected():
    assert sha(BASE)==BASE_SHA,'Original base changed'
    for label,(folder,h) in KEPT.items():assert sha(ROOT/'pretraining/runs'/folder/'model.safetensors')==h,'Kept model changed: '+label
    for arm in ARMS.values():assert sha(arm['parent'])==arm['sha256'],'CPT starting weights changed'

def contract(arm):
    m=load(DATA/'manifest.json')
    for name,h in m['files'].items():assert sha(DATA/name)==h,'Changed prepared file: '+name
    a=ARMS[arm]
    import importlib.metadata,platform
    paths=list(Path(__file__).parent.glob('*.py'))+list(TOKENIZER.glob('*'))
    paths += [ROOT/'pretraining'/n for n in CRITICAL]
    paths += [DATA/'manifest.json',ROOT/'pretraining/scripts/run_sft_152_cpt_comparison.sh',Path(str(a['parent'])+'.json')]
    return dict(format='sft152-matched-cpt-v1',arm=arm,parent_sha256=a['sha256'],plan=PLAN,
                files={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},
                python=platform.python_version(),versions={n:importlib.metadata.version(n) for n in ['mlx','numpy','tokenizers','jsonschema']})
