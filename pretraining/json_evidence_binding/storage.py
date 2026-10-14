"""Rotate only this run's superseded, unevaluated recovery checkpoints."""
import json
from pathlib import Path
import re
import shutil
from json_sft.corpus import file_sha
from sml_v1.common import atomic_json


def prune_transient(output, evaluation_steps, latest_step):
    output = Path(output)
    session = json.loads((output / 'session.json').read_text())
    protected = set(evaluation_steps) | {latest_step}
    for name in ('best.json', 'latest.json', 'best_trained.json'):
        pointer = output / name
        if pointer.exists():
            bundle = json.loads(pointer.read_text())['bundle']
            if Path(bundle).name != bundle:
                raise ValueError('Unsafe checkpoint pointer')
            protected.add(json.loads((output / bundle / 'model.safetensors.json').read_text())['step'])
    receipt = output / 'checkpoint-rotation.json'
    records = json.loads(receipt.read_text())['removed'] if receipt.exists() else []
    for path in sorted(output.glob('step_*_*')):
        if path.is_symlink() or not path.is_dir() or not re.fullmatch(r'step_\d{7}_[0-9a-f]{12}', path.name):
            continue
        manifest = path / 'manifest.json'
        metadata = path / 'model.safetensors.json'
        if not manifest.exists() or not metadata.exists():
            continue
        meta = json.loads(metadata.read_text())
        if meta.get('json_sft', {}).get('contract') != session['contract']:
            continue
        step = meta['step']
        if step in protected or (output / f'evaluations/development_step_{step:07d}.json').exists():
            continue
        records.append(dict(bundle=path.name, step=step, manifest_sha256=file_sha(manifest),
                            bytes=sum(p.stat().st_size for p in path.iterdir() if p.is_file())))
        shutil.rmtree(path)
    atomic_json(receipt, dict(policy='Preserve parent, every evaluated checkpoint, best and latest; rotate only superseded transient checkpoints of this run.', removed=records))
