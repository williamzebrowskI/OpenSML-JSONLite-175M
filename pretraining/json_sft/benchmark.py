"""Frozen extraction benchmark integrity checks and annotation-based scoring.

Gold labels are used for admission and scoring only. Generation receives only
source text, the supplied schema, and the original extraction question.
"""
from collections import Counter
import hashlib
from pathlib import Path
import re

from jsonschema import FormatChecker, validators
from json_specialist.data import canonical, strict_json
from .corpus import file_sha, norm
from .diagnostics import field_counts, leaves
from .evaluate import equal_json, score
from .ordered_serialization import prompt
from .serving import validate_request

FORMAT = 'opensml-sob-text-extractive-v1'
STYLE = 'text-schema-plain-v1'
REVISION = 'c118e38abdef6a8e1beba183405c70b28ff7d5a8'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def literal_evidence(document, target):
    """Conservative admission screen, not a proof of correct field assignment."""
    result = {}
    for path, value in leaves(target).items():
        if isinstance(value, str) and value.strip():
            pattern = re.escape(value)
        elif type(value) in (int, float):
            pattern = r'(?<![\w.])' + re.escape(str(value)) + r'(?![\w.])'
        else:
            raise ValueError('Nonliteral/empty/boolean/null reference needs a separate benchmark track')
        spans = [[m.start(), m.end()] for m in re.finditer(pattern, document)]
        if not spans:
            raise ValueError('Reference value absent from source: ' + path)
        result[path] = spans
    return result


def model_input(row, tokenizer, style=STYLE):
    # Intentionally construct a new object, never pass the scored row to generation.
    return validate_request(row['schema'], row['document'], row['instruction'], tokenizer, style)


def verify_dataset(root, tokenizer):
    root = Path(root)
    manifest = strict_json((root / 'manifest.json').read_text())
    if manifest['format'] != FORMAT or manifest['serialization'] != STYLE:
        raise ValueError('Unsupported benchmark version')
    from .corpus import ROOT
    if manifest['tokenizer_sha256'] != file_sha(ROOT / 'reference/opensml-150m/tokenizer/bytebpe32k_v1/tokenizer.json'):
        raise ValueError('Benchmark tokenizer differs')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or file_sha(root / name) != expected:
            raise ValueError('Benchmark file changed: ' + name)
    review_rows = strict_json((root / 'reviews.json').read_text())['records']
    reviews = {r['id']: r for r in review_rows}
    if len(reviews) != len(review_rows):
        raise ValueError('Repeated review IDs')
    with (root / 'cases.jsonl').open() as f:
        rows = [strict_json(line) for line in f]
    if len(rows) != manifest['records'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Benchmark count/ID mismatch')
    seen = set()
    for row in rows:
        source = digest(norm(row['document']))
        if source in seen:
            raise ValueError('Repeated benchmark source')
        seen.add(source)
        if row['source_sha256'] != source:
            raise ValueError('Source digest differs')
        review = reviews.get(row['id'], {})
        if review.get('decision') != 'accept' or review.get('record_sha256') != digest(canonical(row)):
            raise ValueError('Missing or stale accepted review')
        model_input(row, tokenizer)
        validator = validators.validator_for(row['schema'])
        validator.check_schema(row['schema'])
        validator(row['schema'], format_checker=FormatChecker()).validate(row['target'])
        if literal_evidence(row['document'], row['target']) != row['evidence_spans']:
            raise ValueError('Evidence offsets differ')
        if len(tokenizer.encode(canonical(row['target']), add_special_tokens=False).ids) + 1 > 512:
            raise ValueError('Reference exceeds output budget')
        if len(tokenizer.encode(prompt(model_input(row, tokenizer)), add_special_tokens=False).ids) != row['prompt_tokens']:
            raise ValueError('Prompt token count differs')
    return manifest, rows


def reject_training_overlap(rows, training):
    """Check normalized sources and near-duplicate word shingles without labels."""
    from .corpus import near_cross_split
    training = Path(training)
    with (training / 'train.jsonl').open() as f:
        train = [strict_json(line) for line in f]
    actual_hashes = {digest(norm(r['document'])) for r in train}
    if any(r['source_sha256'] in actual_hashes for r in rows):
        raise ValueError('Benchmark source overlaps SFT training')
    held_out = [dict(r, split='test', source_group='sob:' + r['source_id']) for r in rows]
    near_cross_split(train + held_out)
    return dict(training_records=len(train), training_jsonl_sha256=file_sha(training / 'train.jsonl'),
                exact_source_overlap=0, word5_jaccard_ge_085_overlap=0,
                scope='Selected checkpoint SFT corpus only; complete pretraining decontamination is not established.')


def evaluate_answer(answer, row):
    checks = score(answer['text'], row, answer['stop_reason'])
    fields = field_counts(answer['text'], row['target'], row['document'])
    return checks, fields


def summarize(outputs):
    if not outputs:
        raise ValueError('No benchmark outputs')
    n = len(outputs)
    total = Counter()
    for row in outputs:
        total.update(row['fields'])
    correct = total['correct_fields']
    precision = correct / max(1, total['generated_fields'])
    recall = correct / max(1, total['expected_fields'])
    times = sorted(r['seconds'] for r in outputs)
    seconds = sum(times)
    tokens = sum(r['output_tokens'] for r in outputs)
    return dict(evaluated=n,
        counts={k: sum(bool(r['checks'][k]) for r in outputs)
                for k in ('exact_json', 'schema_valid', 'valid_json', 'natural_stop')},
        **{k: sum(bool(r['checks'][k]) for r in outputs) / n
           for k in ('exact_json', 'schema_valid', 'valid_json', 'natural_stop')},
        field_precision=precision, field_recall=recall,
        field_f1=2 * precision * recall / max(1e-12, precision + recall), field_counts=dict(total),
        stop_reasons=dict(Counter(r['stop_reason'] for r in outputs)),
        output_tokens=tokens, generation_seconds=seconds,
        output_tokens_per_second=tokens / max(seconds, 1e-12),
        latency_p50_seconds=times[(n - 1) // 2], latency_p95_seconds=times[int((n - 1) * .95)])
