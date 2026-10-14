"""Auditable schema-question permutation pairs and full admitted replay.

No new answer annotations are fabricated. All source families keep their old
split; swapping descriptions must also swap the corresponding target values.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from json_specialist.data import canonical, strict_json
from .corpus import ROOT, file_sha, load_tokenizer, near_cross_split, rank
from .grounded import check as previous_check, BASE
from .ordered_serialization import serialize
from .serialization import PLAIN
from .serving import validate_request

FORMAT = 'opensml-json-question-contrasts-v1'
READING = ROOT / 'pretraining/data/json_012_grounded_human_v2'
AMBIGUOUS = {'56dfe86b7aa994140058e253', '572f73b704bcaa1900d76970'}


def transform(original, swapped):
    row = copy.deepcopy(original)
    row['id'] = 'contrast:' + row['id'] + (':swapped' if swapped else ':original')
    row['source'] = 'question_binding'
    row['original_record'] = original
    row['permutation'] = int(swapped)
    props = row['schema']['properties']
    target = row['target']
    if 'fields' in props:
        props = props['fields']['properties']
        target = target['fields']
    names = sorted(props)
    if len(names) < 2:
        raise ValueError('Contrasts require at least two questions')
    if swapped:
        old_props = copy.deepcopy(props)
        old_target = copy.deepcopy(target)
        for i, name in enumerate(names):
            other = names[(i + 1) % len(names)]
            props[name] = old_props[other]
            target[name] = old_target[other]
    return row


def validate(row, tok):
    from jsonschema import validators
    validate_request(row['schema'], row['document'], row['instruction'], tok, PLAIN)
    validators.validator_for(row['schema'])(row['schema']).validate(row['target'])
    if row['source'] == 'question_binding':
        if canonical(row) != canonical(transform(row['original_record'], bool(row['permutation']))):
            raise ValueError('Field/question/target permutation differs')
        if any(q['id'] in AMBIGUOUS for q in row['original_record']['oracle']['questions']):
            raise ValueError('Reviewed ambiguous source excluded')
    return serialize(tok, row)


def prepare(output):
    previous_check(BASE); previous_check(READING)
    tok = load_tokenizer(); rows = []
    for split in ('train', 'development', 'test'):
        # Full admitted old training replay; no cap. Keep all original holdouts.
        for old in map(strict_json, (BASE / (split + '.jsonl')).read_text().splitlines()):
            if split == 'train' and any(qid in canonical(old) for qid in AMBIGUOUS):
                continue
            try:
                validate_request(old['schema'], old['document'], old['instruction'], tok, PLAIN)
            except ValueError:
                continue
            rows.append(dict(old, source='routine_retention', original_record=old))
        originals = [r for r in map(strict_json, (READING / (split + '.jsonl')).read_text().splitlines())
                     if r['source'] == 'long_reading' and len(r['oracle']['questions']) >= 2
                     and all(q['id'] not in AMBIGUOUS for q in r['oracle']['questions'])]
        cap = 3000 if split == 'train' else 100000
        for original in sorted(originals, key=lambda r: rank('contrast:' + r['id']))[:cap]:
            rows.extend([transform(original, False), transform(original, True)])
    near_cross_split(rows)
    output = Path(output)
    if output.exists():
        raise ValueError('Corpus already exists')
    with tempfile.TemporaryDirectory(prefix='.contrasts-', dir=output.parent) as temp:
        stage = Path(temp); stats = {}
        for split in ('train', 'development', 'test'):
            subset = sorted((r for r in rows if r['split'] == split), key=lambda r: rank(r['id']))
            arrays = []; masks = []; offsets = [0]
            for row in subset:
                x, y = validate(row, tok); arrays.append(x); masks.append(y); offsets.append(offsets[-1] + len(x))
            (stage / (split + '.jsonl')).write_text(''.join(canonical(r) + '\n' for r in subset))
            for name, data in [('tokens', np.concatenate(arrays)), ('mask', np.concatenate(masks)), ('offsets', np.asarray(offsets, np.uint64))]:
                np.save(stage / (split + '.' + name + '.npy'), data, allow_pickle=False)
            lengths = list(map(len, arrays))
            stats[split] = dict(records=len(subset), total_tokens=sum(lengths), sources=dict(Counter(r['source'] for r in subset)), max_total_tokens=max(lengths))
        for name in ('extraction-smoke-10.json', 'jsonschemabench-structural-smoke-10.json', 'reserved_public_eval_sources.json'):
            shutil.copyfile(BASE / name, stage / name)
        m = dict(format=FORMAT, tier='curated_annotations', serialization=PLAIN, status='experimental_counterbalanced_field_assignment',
            splits=stats, inputs={str(p): file_sha(p / 'manifest.json') for p in (BASE, READING)},
            tokenizer_sha256=file_sha(ROOT / 'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'),
            code={str(Path(__file__).relative_to(ROOT)): file_sha(__file__)},
            scope=['No new human labels: exact old source replay plus two field-question permutation views of each selected multi-question passage.',
                   'Full admitted V8 replay to reduce forgetting. Both views and every article stay in their original split.',
                   'Independent transformation checks do not certify every inherited annotation. Reviewed ambiguous Duke and SVCD-maximum questions excluded from training.',
                   'No benchmark/probe labels used for supervision. Frozen SOB test is regression only, not fresh holdout.'],
            files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n'); stage.rename(output)
    return m


def check(root, full=True):
    root = Path(root); m = strict_json((root / 'manifest.json').read_text())
    if m['format'] != FORMAT:
        return previous_check(root, full=full)
    for name, digest in m['files'].items():
        if Path(name).name != name or file_sha(root / name) != digest:
            raise ValueError('Contrast corpus changed')
    for name, digest in m['code'].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError('Contrast code changed')
    if full:
        tok = load_tokenizer(); originals = {}; allrows = []
        if file_sha(ROOT / 'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json') != m['tokenizer_sha256']:
            raise ValueError('Tokenizer differs')
        for path, digest in m['inputs'].items():
            path = Path(path)
            if file_sha(path / 'manifest.json') != digest:
                raise ValueError('Input corpus identity changed')
            previous_check(path)
            for split in ('train', 'development', 'test'):
                originals.update({(str(path),r['id']):r for r in map(strict_json, (path / (split + '.jsonl')).read_text().splitlines())})
        for split in ('train', 'development', 'test'):
            rows = list(map(strict_json, (root / (split + '.jsonl')).read_text().splitlines()))
            ids = np.load(root / (split + '.tokens.npy'), mmap_mode='r'); mask = np.load(root / (split + '.mask.npy'), mmap_mode='r'); offsets = np.load(root / (split + '.offsets.npy'), mmap_mode='r')
            if len(rows) != m['splits'][split]['records'] or len(offsets) != len(rows)+1 or offsets[0] != 0 or offsets[-1] != len(ids) or len(mask) != len(ids):
                raise ValueError('Array lengths differ')
            for i, row in enumerate(rows):
                old = row['original_record']
                origin = READING if row['source']=='question_binding' else BASE
                if row['split'] != split or old['split'] != split or canonical(old) != canonical(originals[(str(origin),old['id'])]):
                    raise ValueError('Frozen source/split differs')
                if row['source'] == 'routine_retention':
                    if canonical(row) != canonical(dict(old, source='routine_retention', original_record=old)):
                        raise ValueError('Routine replay changed')
                x, y = validate(row, tok); a,b = map(int, offsets[i:i+2])
                if not np.array_equal(x, ids[a:b]) or not np.array_equal(y, mask[a:b]):
                    raise ValueError('Serialization/mask differs')
            allrows.extend(rows)
        near_cross_split(allrows)
    return m
