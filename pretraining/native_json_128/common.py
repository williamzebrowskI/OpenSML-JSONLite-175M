from pathlib import Path
from json_sampling_119.common import sha,load,dumps,prompt
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'pretraining/data/sft_128_joint_extraction_20261016'
RUN=ROOT/'pretraining/runs/sft_128_joint_extraction_20261016'
REPORT=ROOT/'reports/model-design/SFT-128_joint_extraction_20261016'
PARENT=ROOT/'pretraining/runs/sft_105_cardinality_distillation_20261014/step_0000064_cdd8be1e8d5a'
PARENT_SHA='655fb5cfba009163cd8a7f7fe7b170b998bb452c217b1b473cfe10b33de0ccd1'
TOK=ROOT/'reference/opensml-150m/tokenizer/bytebpe32k_v1'
OLD=ROOT/'pretraining/data/sft_100_native_json_20261014'
CARD=ROOT/'pretraining/data/sft_105_cardinality_distillation_20261014'
LEGACY=ROOT/'reports/model-design/SFT-118_activation_subspace_20261015/retired-experiment'
PRIOR=ROOT/'reports/model-design/JSON-AUDIT-127_diagnostics_20261016'
LOCK=ROOT/'pretraining/diagnostics/m5_migration_20260922/network_20261007_084622/.training-launch.lock'
PLAN=dict(seed=202610160128,batch_size=64,microbatch=2,epochs=1,peak_lr=5e-6,final_lr=1e-6,warmup=20,weight_decay=0.,betas=[.9,.95],eps=1e-8,clip=1.,context=2048,max_tokens=256,eval_every=100,adaptation='all_parameters',precision='BF16 compute / FP32 master AdamW',objective='Mean per-request assistant-JSON token cross entropy, including turn EOS. No teacher KL, no inference repair or assembly.',order='One deterministic joint shuffle, all prepared records exactly once; no late domain-only phase.')
def check():
 import json
 m=json.loads((DATA/'manifest.json').read_text())
 if sha(PARENT/'model.safetensors')!=PARENT_SHA or m['plan']!=PLAN:raise ValueError('Parent/plan changed')
 for name,h in m['files'].items():
  if sha(DATA/name)!=h:raise ValueError('Prepared data changed: '+name)
 for name,h in m['inputs'].items():
  if sha(ROOT/name)!=h:raise ValueError('Source changed: '+name)
 return m
