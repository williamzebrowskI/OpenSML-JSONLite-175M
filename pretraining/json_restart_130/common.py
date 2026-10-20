from pathlib import Path
from json_sampling_119.common import sha, load, dumps, prompt
from sml_v1.common import atomic_json
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'pretraining/data/sft_130_joint_foundation_20261016'
REPORT = ROOT/'reports/model-design/SFT-130_joint_foundation_20261016'
RUN = ROOT/'pretraining/runs/sft_130_joint_foundation_20261016'
REFINE = ROOT/'pretraining/runs/sft_131_json_refinement_20261016'
BASE = ROOT/'pretraining/runs/research_30b_176m_stage_a_continue_20261009/step_0132005_63f79113c3d2/model.safetensors'
BASE_SHA = '5151567e266244c9e32fd6e01775192633abe227320269ecdefe28ce43e9ccc3'
RELEASE = ROOT/'pretraining/runs/sft_128_no_prune_20261016/step_0000300_43e2a79d3885/model.safetensors'
RELEASE_SHA = '781597baeb9b51d535fc27cd6f941dbc202e01d62c8d963587a3f3d565128560'
TOK = ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
LOCK = ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'
ORIGINAL = ROOT/'pretraining/data/sft_100_native_json_20261014'
QA = ROOT/'pretraining/data/sft_065_targeted_qa_20261013'
PUBLISHED = ROOT/'reports/model-design/SFT-128_joint_extraction_20261016/release-comparison'
PLAN = dict(seed=202610160130, batch_size=64, microbatch=2, presentations=32000,
    epochs=1, peak_lr=3e-5, final_lr=3e-6, warmup=25, betas=[.9,.95], eps=1e-8,
    weight_decay=0., clip=1., context=2048, max_tokens=256, train_max_sequence=1024,
    eval_every=200, precision='BF16 compute / FP32 master AdamW', adaptation='full weights',
    objective='Per-request mean assistant JSON and turn EOS cross entropy; all inputs masked.',
    retention='All checkpoints retained; no automatic promotion; observed losses explicitly reported.')
REFINEMENT = dict(seed=202610160131, batch_size=32, microbatch=2, presentations=4096,
    replay=3072, focus=1024, peak_lr=2e-6, final_lr=4e-7, warmup=8,
    eval_steps=[32,64,96,128], max_focus_families=3,
    policy='Choose weak families from foundation development rates only. No test/reserved targets or generated guesses.')

def write_rows(path, rows):
    Path(path).write_text(''.join(dumps(r)+'\n' for r in rows))

def normalized(s): return ' '.join(s.casefold().split())

def obj(props): return dict(type='object', properties=props, required=list(props), additionalProperties=False)

def check():
    m=__import__('json').loads((DATA/'manifest.json').read_text())
    if m['plan']!=PLAN or m['refinement']!=REFINEMENT: raise ValueError('Prepared plan changed')
    for n,h in m['files'].items():
        if sha(DATA/n)!=h: raise ValueError('Prepared data changed: '+n)
    for n,h in m['inputs'].items():
        if sha(ROOT/n)!=h: raise ValueError('Source changed: '+n)
    if sha(BASE)!=BASE_SHA or sha(RELEASE)!=RELEASE_SHA: raise ValueError('Protected weights changed')
    return m
