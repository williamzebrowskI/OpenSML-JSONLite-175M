"""Long-passage post-training experiment, preserving published source labels.

Only training/validation source splits supply new supervision. The frozen SOB
test benchmark remains excluded. Annotation replay is not factual certification.
"""
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pyarrow.parquet as pq
from jsonschema import FormatChecker, validators, ValidationError
from json_specialist.data import canonical, strict_json
from .corpus import ROOT, file_sha, load_tokenizer, near_cross_split, norm, rank, sha
from .ordered_serialization import serialize, prompt
from .serialization import PLAIN
from .benchmark import literal_evidence, REVISION as SOB_REVISION
from .serving import validate_request
from .precise import check as previous_check

FORMAT = 'opensml-json-grounded-long-v1'
BASE = ROOT / 'pretraining/data/json_007_relational_v8'
QA = ROOT / 'pretraining/data/json_002_sources_20261010'
SOB = Path('/tmp/opensml-json-eval-audit-20261009')
REJECT = {'78f02c3e55a3c756e73fbd8cea486d2b7903231bb60b9a75a28f95e685a55092':
          'Film character job incorrectly assigned the mythological oak-nymph description.'}


def qa_split(title):
    bucket = int(rank('squad-title:' + title)[:8], 16) % 100
    return 'development' if bucket < 8 else 'test' if bucket < 16 else 'train'


def inputs():
    receipt = strict_json((QA / 'receipt.json').read_text())
    qa_file = QA / 'train-00000-of-00001.parquet'
    if file_sha(qa_file) != receipt['files'][qa_file.name]['sha256']:
        raise ValueError('Pinned SQuAD bytes changed')
    downloads = strict_json((ROOT / 'reports/model-design/JSON-003_benchmark-audit-20261009/download_manifest.json').read_text())
    expected = {r['file']: r['sha256'] for r in downloads['files']}
    sob = {}
    for split in ('train', 'validation', 'test'):
        name = 'sob__data__' + split + '-00000-of-00001.parquet'
        if file_sha(SOB / name) != expected[name]:
            raise ValueError('Pinned SOB bytes changed')
        # Test labels and questions are never used to construct training rows.
        columns = ['context', 'source_id'] if split == 'test' else None
        sob[split] = pq.read_table(SOB / name, columns=columns).to_pylist()
    questions = pq.read_table(qa_file).to_pylist()
    return questions, sob, dict(squad=receipt, sob_revision=SOB_REVISION,
        sob_files={k: v for k, v in expected.items() if k.endswith('.parquet') and k.startswith('sob__')})


def admitted_question(q):
    texts = q['answers']['text']
    if not texts or len(texts) != len(q['answers']['answer_start']) or len(set(texts)) != 1 or len(texts[0]) > 100 or len(texts[0].split()) > 16:
        return False
    if not 250 <= len(q['context']) <= 1800:
        return False
    if not re.match(r'^(Who|What|When|Where|Which|How many|How much)\b', q['question']):
        return False
    if re.search(r'\b(not|never|except|least)\b', q['question'], re.I):
        return False
    if re.search(r'\b(he|his|her|she|they|their|this|these|those)\b', q['question'], re.I):
        return False
    return all(q['context'][int(start):int(start)+len(text)] == text
               for text, start in zip(texts, q['answers']['answer_start']))


def qa_row(questions, distractors, split, variant):
    """Schema construction uses question IDs/text, never answer values."""
    contexts = [questions[0]['context']] + distractors
    position = int(rank(questions[0]['id'] + ':position')[:8], 16) % len(contexts)
    contexts.insert(position, contexts.pop(0))
    document = '\n\n'.join(contexts)
    names = ('answer', 'detail', 'result', 'value', 'text', 'extracted')
    mode = int(rank(questions[0]['id'] + ':schema')[:8], 16) % 3
    props = {}
    keys = []
    for i, q in enumerate(questions):
        key = names[(int(rank(q['id'])[:8], 16) + i) % len(names)]
        while key in props:
            key += '_2'
        keys.append(key)
        field = dict(type='string', description=q['question'] + ' Copy the answering span verbatim from the supplied source. Do not copy facts answering a different question.')
        if mode == 1:
            field = dict(type='array', items=dict(type='string'), description=field['description'] + ' Return a one-element array.')
        props[key] = field
    schema = dict(type='object', properties=props, required=list(props), additionalProperties=False)
    target = {key: ([q['answers']['text'][0]] if mode == 1 else q['answers']['text'][0]) for key, q in zip(keys, questions)}
    if mode == 2:
        schema = dict(type='object', properties={'fields': schema}, required=['fields'], additionalProperties=False)
        target = {'fields': target}
    return dict(id='grounded-qa:' + ':'.join(q['id'] for q in questions) + ':' + str(variant),
        source='long_reading', source_group='squad-title:' + questions[0]['title'], split=split,
        document=document, schema=schema, target=target, _prompt_style=PLAIN,
        instruction='Answer each field question using the supplied passages. Choose the correct entity and relation; unrelated passages are distractors. Copy an answering span verbatim, preserving case and punctuation. Return only the required JSON.',
        oracle=dict(questions=questions, distractors=distractors, variant=variant),
        provenance=dict(dataset='rajpurkar/squad_v2', revision='3ffb306f725f7d2ce8394bc1873b24868140c412',
            license='CC-BY-SA-4.0', attribution='SQuAD authors and contributing Wikipedia authors',
            quality='Published human answer spans with exact replay and consensus filter; sampled review, not exhaustive re-adjudication. Distractors are published passages from the same article and split.'))


def sob_row(original, split):
    schema = strict_json(original['json_schema'])
    target = strict_json(original['ground_truth'])
    if original['record_id'] in REJECT:
        raise ValueError('reviewed_bad_reference')
    if canonical(target) != canonical(strict_json(original['validated_output'])):
        raise ValueError('reference_versions_differ')
    literal_evidence(original['context'], target)
    validators.validator_for(schema)(schema, format_checker=FormatChecker()).validate(target)
    return dict(id='grounded-sob:' + original['record_id'], source='native_reading',
        source_group='sob:' + original['source_id'], split=split, document=original['context'],
        schema=schema, target=target, instruction=original['question'], _prompt_style=PLAIN,
        oracle=dict(upstream_split=original['split'], record_id=original['record_id']),
        provenance=dict(dataset='interfaze-ai/sob', revision=SOB_REVISION, license='See LICENSE-SOB.txt',
            quality='Screened original synthetic reference; schema/versions/literal evidence checked, sampled semantic review only. No claim all labels are correct.'))


def validate(row, tokenizer):
    validate_request(row['schema'], row['document'], row['instruction'], tokenizer, PLAIN)
    validators.validator_for(row['schema'])(row['schema'], format_checker=FormatChecker()).validate(row['target'])
    if row['source'] == 'long_reading':
        o = row['oracle']
        rebuilt = qa_row(o['questions'], o['distractors'], row['split'], o['variant'])
        for name in ('id', 'document', 'schema', 'target', 'instruction'):
            if canonical(row[name]) != canonical(rebuilt[name]):
                raise ValueError('Source/question/span replay differs: ' + name)
        if any(qa_split(q['title']) != row['split'] or not admitted_question(q) for q in o['questions']):
            raise ValueError('Question admission/split differs')
    elif row['source'] == 'native_reading':
        literal_evidence(row['document'], row['target'])
    elif row['source'] == 'routine_retention':
        old = row['original_record']
        for name in ('document', 'schema', 'target', 'instruction', 'source_group', 'split'):
            if canonical(old[name]) != canonical(row[name]):
                raise ValueError('Replay record changed')
    else:
        raise ValueError('Unexpected grounded source')
    return serialize(tokenizer, row)


def prepare(output):
    previous_check(BASE)
    questions, sob, source_receipt = inputs()
    tokenizer = load_tokenizer()
    blocked_ids = {r['source_id'] for r in sob['test']}
    blocked_docs = {sha(norm(r['context'])) for r in sob['test']}
    held_ids = {r['source_id'] for r in sob['validation']}
    held_docs = {sha(norm(r['context'])) for r in sob['validation']}
    rows = []
    counters = Counter()
    # Preserve routine and older natural-reading holdouts; sampled training replay.
    caps = {'squad_extract': 4000, 'business_transfer': 1500, 'sgd': 1000,
            'narrative_transfer': 1000, 'controlled': 500}
    for split in ('train', 'development', 'test'):
        for old in map(strict_json, (BASE / (split + '.jsonl')).read_text().splitlines()):
            try:
                validate_request(old['schema'], old['document'], old['instruction'], tokenizer, PLAIN)
            except ValueError:
                counters['unsupported_replay:' + split] += 1
                continue
            if split == 'train':
                key = 'replay:' + old['source']
                if counters[key] >= caps.get(old['source'], 0):
                    continue
                counters[key] += 1
            rows.append(dict(old, source='routine_retention', original_record=old))
    grouped = defaultdict(list)
    article_passages = defaultdict(set)
    for q in questions:
        if 200 <= len(q['context']) <= 1800:
            article_passages[q['title']].add(q['context'])
        if admitted_question(q):
            grouped[(q['title'], q['context'])].append(q)
    counts = Counter()
    for (title, context), qs in sorted(grouped.items(), key=lambda x: rank('long:' + x[0][1])):
        split = qa_split(title)
        cap = 8000 if split == 'train' else 150
        if counts[split] >= cap:
            continue
        qs = sorted(qs, key=lambda q: rank(q['id']))
        for variant in range(min(2, len(qs))):
            if counts[split] >= cap:
                break
            batch = qs[variant:variant + (1 + int(rank(qs[variant]['id'])[:8], 16) % 3)]
            distractors = []
            for paragraph in sorted(article_passages[title] - {context}, key=lambda p: rank(qs[variant]['id'] + p)):
                candidate = qa_row(batch, distractors + [paragraph], split, variant)
                try:
                    validate(candidate, tokenizer)
                except ValueError:
                    continue
                distractors.append(paragraph)
                if len(tokenizer.encode(prompt(candidate), add_special_tokens=False).ids) >= 1050:
                    break
            if not distractors:
                continue
            row = qa_row(batch, distractors, split, variant)
            if any(sha(norm(p)) in blocked_docs for p in [context] + distractors):
                continue
            rows.append(row)
            counts[split] += 1
            if counts[split] % 1000 == 0:
                print('[prepare-grounded] reading ' + split + ': ' + str(counts[split]), flush=True)
    seen = set()
    native_counts = Counter()
    for upstream in ('validation', 'train'):
        for original in sorted(sob[upstream], key=lambda r: rank('long-native:' + r['record_id'])):
            doc_hash = sha(norm(original['context']))
            if original['source_id'] in blocked_ids or doc_hash in blocked_docs or original['source_id'] in seen:
                continue
            if upstream == 'train' and (original['source_id'] in held_ids or doc_hash in held_docs):
                continue
            split = ('development' if int(rank('sob-holdout:' + original['source_id'])[:8], 16) % 2 else 'test') if upstream == 'validation' else 'train'
            if native_counts[split] >= (1000 if split == 'train' else 80):
                continue
            try:
                row = sob_row(dict(original, split=upstream), split)
                validate(row, tokenizer)
            except (ValueError, ValidationError):
                counters['native_excluded'] += 1
                continue
            rows.append(row)
            seen.add(original['source_id'])
            native_counts[split] += 1
    # Check benchmark component passages as well as whole documents: bundles must
    # not hide a known evaluation paragraph inside a larger training input.
    frozen = [strict_json(s) for s in (ROOT / 'pretraining/data/json_benchmark_sob_text_v1/cases.jsonl').read_text().splitlines()]
    components = {norm(p) for r in frozen for p in r['document'].split('\n\n') if len(p) >= 80}
    components.update(norm(p.split(': ',1)[1]) for r in frozen for p in r['document'].split('\n\n') if ': ' in p and len(p) >= 80)
    kept = []
    for row in rows:
        if row['source'] == 'long_reading':
            pieces = [row['oracle']['questions'][0]['context']] + row['oracle']['distractors']
        else:
            pieces = row['document'].split('\n\n')
        if row['split'] == 'train' and any(norm(p) in components for p in pieces):
            counters['benchmark_component_excluded'] += 1
            continue
        kept.append(row)
    guards = [dict(id='reserved-sob:' + r['id'], source_group='reserved-sob:' + r['source_id'],
                   split='test', document=r['document']) for r in frozen]
    def parts(row):
        if row.get('source') == 'long_reading':
            return [row['oracle']['questions'][0]['context']] + row['oracle']['distractors']
        pieces = row['document'].split('\n\n')
        return pieces + [p.split(': ',1)[1] for p in pieces if ': ' in p]
    held_parts = {norm(p) for r in kept + guards if r['split'] != 'train' for p in parts(r) if len(p) >= 80}
    filtered = []
    for row in kept:
        if row['split'] == 'train' and any(norm(p) in held_parts for p in parts(row) if len(p) >= 80):
            counters['heldout_component_excluded'] += 1
            continue
        filtered.append(row)
    checked, removed = near_cross_split(filtered + guards, drop=True)
    rows = [r for r in checked if not r['id'].startswith('reserved-sob:')]
    output = Path(output)
    if output.exists():
        raise ValueError('New corpus output already exists')
    with tempfile.TemporaryDirectory(prefix='.grounded-', dir=output.parent) as temp:
        stage = Path(temp)
        stats = {}
        for split in ('train', 'development', 'test'):
            subset = sorted((r for r in rows if r['split'] == split), key=lambda r: rank(r['id']))
            arrays, masks, offsets = [], [], [0]
            for row in subset:
                ids, mask = validate(row, tokenizer)
                arrays.append(ids); masks.append(mask); offsets.append(offsets[-1] + len(ids))
            (stage / (split + '.jsonl')).write_text(''.join(canonical(r) + '\n' for r in subset))
            for name, value in [('tokens', np.concatenate(arrays)), ('mask', np.concatenate(masks)), ('offsets', np.asarray(offsets, np.uint64))]:
                np.save(stage / (split + '.' + name + '.npy'), value, allow_pickle=False)
            lengths = sorted(map(len, arrays))
            stats[split] = dict(records=len(subset), sources=dict(Counter(r['source'] for r in subset)), total_tokens=sum(lengths),
                median_total_tokens=lengths[len(lengths)//2], max_total_tokens=max(lengths), over_905_tokens=sum(n > 905 for n in lengths))
        for name in ('extraction-smoke-10.json', 'jsonschemabench-structural-smoke-10.json', 'reserved_public_eval_sources.json'):
            shutil.copyfile(BASE / name, stage / name)
        shutil.copyfile(SOB / 'sob__LICENSE', stage / 'LICENSE-SOB.txt')
        (stage / 'admission-audit.json').write_text(json.dumps(dict(counts=counters, near_split_exclusions=removed,
            scope='Annotation replay and sampled semantic review; no exhaustive factual certification. SOB test labels unused. Same-article SQuAD distractors are not synthetic facts.'), indent=2) + '\n')
        manifest = dict(format=FORMAT, tier='curated_annotations', serialization=PLAIN,
            status='experimental_long_reading_not_release_certified', splits=stats, source_receipt=source_receipt,
            inputs=dict(base_manifest=file_sha(BASE / 'manifest.json'), base=str(BASE)),
            tokenizer_sha256=file_sha(ROOT / 'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'),
            scope=['New natural-passage and native-schema training, frozen old holdouts and new article-disjoint reading holdouts.',
                   'SOB original synthetic references are screened, not uniformly human verified. Literal evidence is not proof of entity association.',
                   'No SOB test labels or benchmark reference values used in training. Benchmark source IDs and component texts are exclusion guards only.'],
            code={str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}, files={p.name:file_sha(p) for p in stage.iterdir()})
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(output)
    return manifest


def check(root, full=True):
    root = Path(root)
    m = strict_json((root / 'manifest.json').read_text())
    if m['format'] != FORMAT:
        return previous_check(root, full=full)
    for name, digest in m['files'].items():
        if Path(name).name != name or file_sha(root / name) != digest:
            raise ValueError('Grounded corpus file changed: ' + name)
    for name, digest in m['code'].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError('Grounded conversion code changed')
    if full:
        tokenizer = load_tokenizer()
        if file_sha(ROOT / 'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json') != m['tokenizer_sha256']:
            raise ValueError('Grounded tokenizer differs')
        questions, sob, _ = inputs()
        originals = {q['id']:q for q in questions}
        paragraphs = defaultdict(set)
        for q in questions:
            paragraphs[q['title']].add(q['context'])
        native = {r['record_id']:dict(r, split=s) for s in ('train', 'validation') for r in sob[s]}
        base = Path(m['inputs']['base'])
        if file_sha(base / 'manifest.json') != m['inputs']['base_manifest']:
            raise ValueError('Replay corpus identity changed')
        replay = {r['id']:r for split in ('train','development','test') for r in map(strict_json,(base / (split+'.jsonl')).read_text().splitlines())}
        allrows = []
        for split in ('train', 'development', 'test'):
            rows = list(map(strict_json, (root / (split + '.jsonl')).read_text().splitlines()))
            ids = np.load(root / (split + '.tokens.npy'), mmap_mode='r')
            mask = np.load(root / (split + '.mask.npy'), mmap_mode='r')
            offsets = np.load(root / (split + '.offsets.npy'), mmap_mode='r')
            if len(rows) != m['splits'][split]['records'] or len(offsets) != len(rows)+1 or offsets[0] != 0 or offsets[-1] != len(ids) or len(mask) != len(ids):
                raise ValueError('Grounded array/count mismatch')
            for i, row in enumerate(rows):
                if row['split'] != split:
                    raise ValueError('Grounded split differs')
                if row['source'] == 'long_reading':
                    qs = row['oracle']['questions']
                    if any(canonical(q) != canonical(originals[q['id']]) for q in qs):
                        raise ValueError('Question differs from pinned upstream')
                    if any(p not in paragraphs[qs[0]['title']] for p in row['oracle']['distractors']):
                        raise ValueError('Distractor differs from pinned same-article upstream')
                elif row['source'] == 'native_reading':
                    rebuilt = sob_row(native[row['oracle']['record_id']], split)
                    if canonical(row) != canonical(rebuilt):
                        raise ValueError('Native reference differs from pinned upstream')
                elif canonical(row['original_record']) != canonical(replay[row['id']]):
                    raise ValueError('Replay differs from frozen original')
                x, y = validate(row, tokenizer)
                a, b = map(int, offsets[i:i+2])
                if not np.array_equal(x, ids[a:b]) or not np.array_equal(y, mask[a:b]):
                    raise ValueError('Grounded serialized arrays differ')
            allrows.extend(rows)
        near_cross_split(allrows)
    return m


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.output)['splits'], indent=2))
