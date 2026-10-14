"""Sparse current-model JSON-036 evaluation with three cache-only references.

Panels diagnose learning; only full development receipts select checkpoints.
Original observed labels, typed leaves and zero-loss guards stay unchanged.
Final tests can reject the committed candidate, never select a replacement.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from json_grounded_records import evaluate as previous
from json_curriculum.evaluate import rescore
from json_restart.evaluate import assess, load_model
from json_sft.corpus import file_sha, load_tokenizer
from json_specialist.data import canonical
from sml_v1.common import atomic_json

ROOT = previous.ROOT
CORPUS = ROOT / 'pretraining/data/json_036_evidence_binding_20261012'
OUTPUT = ROOT / 'pretraining/runs/json_036_evidence_binding_20261012'
REPORT = ROOT / 'reports/model-design/JSON-036_evidence_binding_20261012'
PREFIX = 'evidence036_'
MODES = previous.MODES
BLOCKS = ('foundation_atomic', 'natural_complete')
FRESH_FLOORS = {'foundation_atomic': .75, 'natural_complete': .50}
ANSWERABLE_ATOMIC_FLOOR = .75
NULL_PRECISION_FLOOR = NULL_RECALL_FLOOR = .80
COMMON_COMPLETE_GAIN = .05
PANEL_REQUEST_LIMIT = 192
PANEL_FRESH_BUDGET = PANEL_OBSERVED_BUDGET = 96
PANEL_POSITIVE_BUDGET = PANEL_MIXED_NULL_BUDGET = 48
retained = previous.retained
load = previous.load
metrics = previous.metrics
checkpoint_identity = previous.checkpoint_identity
REFERENCE035_RUN = ROOT / 'pretraining/runs/json_035_grounded_records_20261012'
REFERENCE035 = REFERENCE035_RUN / 'step_0001236_a994d7fc07b2'
REFERENCE035_SHA = '4c5ad3e5d7d81786735317b5fe290f5bf896ff9f891146ec2566cfc7693ec187'
REFERENCE035_METADATA_SHA = '84e290e57fa22f10ff592caa31fdf0f351f38c7b758373260697f8a5500b503f'
FROZEN035 = {
    'evaluations/development_step_0001236.json': '067ad16c227056363ef5d52688d46162a5dd262eb3bf4eb5cba6fab6046c9ffd',
    'evaluations/development_step_0001236.protocol.json': '4b98d440eea9e9976e22da7975d8069ccef1f91fbaea52a177410096fef3aa1c',
    'final-test/protocol.json': '4aaf76236743990b367d7a96fbedae7f22e4fa0ab0e7db737976ee690f4e4ee1',
    'final-test/selection.json': 'd27abb4f98b2fd3f155505e938fba736a5a0eda76058cd9dfe581e4c8de6acb8',
    'final-test/result.json': 'b88f8da3a9a8bb1f82eed70ddebf8f4eac9ab0bc00a21c134f49223c93c1b13b',
    'final-test/candidate-fresh035.json': '60ca0fcccb6f59b377bd1285dd27effe08ac5f964d6014f91af6ce3ef7eee5d2',
    'final-test/candidate-observed025.json': 'ab25650185224a7ad9736707f89bd7704ea5ed8cd6f2962ed2e3f9885ea4d10a',
    'final-test/candidate-observed030.json': 'c328120f52ebbab558b1a84d0439c20ad095a9a4d0ffd49117a5c4ebd7569c9a',
    'final-test/candidate-observed032.json': 'df70ffb67351077da09cc1e7a460d07f35282027e588b630dec4cdee154b50c7',
    'final-test/candidate-observed034fresh.json': 'db0a4b8e5d82ee3f4e75f1a4ff605f085fd5f4e66371d56fad467f5bf9e59c58',
    'step_0001236_a994d7fc07b2/model.safetensors.json': REFERENCE035_METADATA_SHA,
}
OBSERVED_DATA = {**previous.OBSERVED_DATA, 'observed035': previous.CORPUS / 'test.jsonl'}
REFERENCE_OUTPUTS = {
    'reference034': previous.REFERENCE_OUTPUTS,
    'reference032': previous.SECONDARY_OUTPUTS,
    'reference035': {
        'development': REFERENCE035_RUN / 'evaluations/development_step_0001236.json',
        **{name: REFERENCE035_RUN / ('final-test/candidate-' + name + '.json') for name in previous.OBSERVED_DATA},
        'observed035': REFERENCE035_RUN / 'final-test/candidate-fresh035.json',
    },
}
CACHE_FILES = {**{'upstream:' + key: value for key, value in previous.CACHE_FILES.items()},
    **{'035:' + key: REFERENCE035_RUN / key for key in FROZEN035},
    '035-manifest': previous.CORPUS / 'manifest.json',
    '035-development-data': previous.CORPUS / 'development.jsonl',
    '035-test-data': previous.CORPUS / 'test.jsonl'}


def is_new(row):
    return row['source'].startswith(PREFIX)


def qualification_exclusion(row):
    return None if is_new(row) else previous.qualification_exclusion(row)


def qualification_scope(rows):
    excluded = [dict(id=r['id'], source=r['source'], reason=qualification_exclusion(r))
                for r in rows if qualification_exclusion(r)]
    return dict(policy=previous.QUALIFICATION_POLICY, original_requests=len(rows),
        qualified_requests=len(rows)-len(excluded), excluded_requests=len(excluded), excluded=excluded,
        scope='Unchanged predeclared P577 whole-request quarantine only; all original labels and full scores remain visible.')


def subset(rows, result, predicate):
    selected = [r for r in rows if predicate(r)]
    if not selected:
        raise ValueError('Required evaluation block is empty')
    ids = {r['id'] for r in selected}
    return selected, rescore(selected, dict(outputs=[o for o in result['outputs'] if o['id'] in ids]))


def qualified_subset(rows, result):
    return subset(rows, result, lambda r: qualification_exclusion(r) is None)


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def validate_fresh(rows, split):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate JSON-036 fresh requests')
    identities = {}
    representations = defaultdict(lambda: [0, 0])
    for row in rows:
        if not is_new(row) or row['split'] != split or row.get('training_role') != 'holdout':
            raise ValueError('Fresh evaluation requires explicit JSON-036 holdouts')
        atomic = row.get('skill_block') == 'foundation_atomic'
        if row.get('skill_block') not in BLOCKS or atomic != (row.get('task_kind') == 'atomic') or row.get('is_complete_record') is not (not atomic):
            raise ValueError('Atomic/complete classification differs')
        if not row.get('source_group') or not row.get('contrast_group'):
            raise ValueError('Missing fresh world/group identity')
        bindings = row.get('field_bindings', [])
        paths = [tuple(b['field_path']) for b in bindings]
        if len(paths) != len(set(paths)) or set(paths) != set(retained.leaves(row['target'])):
            raise ValueError('Bindings do not cover exactly the requested gold leaves')
        if (atomic and len(bindings) != 1) or (not atomic and len(bindings) < 2):
            raise ValueError('Fresh complete requests need at least two questions')
        for binding in bindings:
            value = _at(row['target'], binding['field_path'])
            answerability = binding['answerability']
            start, end = binding['source_span_start'], binding['source_span_end']
            if answerability == 'answerable':
                if (not isinstance(value, str) or not value or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(row['document']) or row['document'][start:end] != value):
                    raise ValueError('Answerable binding differs from literal source span')
            elif answerability == 'unanswerable':
                if value is not None or start is not None or end is not None:
                    raise ValueError('Unanswerable gold must be null without an invented span')
            else:
                raise ValueError('Unknown answerability')
            key = row['source_group'], binding['question_id']
            representations[key][0 if atomic else 1] += 1
            identity = answerability, start, end, value
            if key in identities and identities[key] != identity:
                raise ValueError('Same world/question has contradictory view labels')
            identities[key] = identity
    if {r['skill_block'] for r in rows} != set(BLOCKS):
        raise ValueError('Fresh split must cover atomic and complete natural requests')
    if any(atomic != 1 or complete < 1 for atomic, complete in representations.values()):
        raise ValueError('Every fresh world/question requires one atomic anchor and complete views')
    return rows


def development_rows():
    fresh = validate_fresh(load(CORPUS / 'development.jsonl'), 'development')
    rows = fresh + previous.development_rows()
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Fresh/observed development IDs collide')
    return rows


def cached_development(label='reference034'):
    if label in ('reference034', '034', 'reference'):
        return previous.cached_development()
    if label in ('reference032', '032'):
        return previous.cached_development('032')
    if label in ('reference035', '035'):
        return rescore(previous.development_rows(), json.loads(REFERENCE_OUTPUTS['reference035']['development'].read_text()))
    raise ValueError('Unknown cached reference')


def reference_identity():
    return dict(format='json036-triple-retained-reference-v1',
        reference034=previous.reference_identity()['primary034'],
        reference032=previous.reference_identity()['secondary032'],
        reference035=dict(path=str(REFERENCE035), step=1236, model_sha256=REFERENCE035_SHA,
            metadata_sha256=REFERENCE035_METADATA_SHA,
            development_sha256=FROZEN035['evaluations/development_step_0001236.json']),
        temperature=0, repetition_penalty=1, max_tokens=512,
        decoder_revision=previous.DECODER_REVISION, truncation=False,
        scope='Cached trained outputs only. Every gold-correct answer/typed leaf remains protected separately for all three references; no base generation.')


def reference_runs():
    """Explicit immutable checkpoint directories for launcher lineage hashing."""
    return [previous.REFERENCE, previous.SECONDARY_REFERENCE, REFERENCE035]


def initialize_reference(output):
    output = Path(output)
    identity = reference_identity()
    session = output / 'session.json'
    if session.exists() and json.loads(session.read_text()).get('accepted_reference') != identity:
        raise ValueError('Session retained references differ')
    retained.commit(output / 'accepted_reference.json', identity)
    return identity


def verify_cache_binding():
    upstream = previous.verify_cache_binding()
    for name, digest in FROZEN035.items():
        if file_sha(REFERENCE035_RUN / name) != digest:
            raise ValueError('Frozen JSON-035 cache changed: ' + name)
    protocol = json.loads((REFERENCE035_RUN / 'final-test/protocol.json').read_text())
    metadata = json.loads((REFERENCE035 / 'model.safetensors.json').read_text())
    expected = dict(candidate_step=1236, candidate_sha256=REFERENCE035_SHA, temperature=0,
        repetition_penalty=1, max_tokens=512, decoder_revision=previous.DECODER_REVISION, truncation=False)
    if any(protocol[key] != value for key, value in expected.items()) or protocol['contract']['cache_files'] != upstream:
        raise ValueError('JSON-035 reference protocol/cache lineage differs')
    if metadata['step'] != 1236 or metadata['json_sft']['contract'] != protocol['contract'] or file_sha(REFERENCE035 / 'model.safetensors') != REFERENCE035_SHA:
        raise ValueError('Frozen JSON-035 checkpoint differs')
    for name, digest in protocol['contract']['code'].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError('Frozen JSON-035 generation dependency changed: ' + name)
    if file_sha(previous.CORPUS / 'manifest.json') != protocol['contract']['corpus_manifest']:
        raise ValueError('JSON-035 corpus manifest differs')
    dev = cached_development('035')
    saved = json.loads(REFERENCE_OUTPUTS['reference035']['development'].read_text())
    if saved['step'] != 1236 or saved['model_sha256'] != REFERENCE035_SHA or saved['metadata_sha256'] != REFERENCE035_METADATA_SHA or dev['request_count'] != 654:
        raise ValueError('JSON-035 development identity/coverage differs')
    for name, path in OBSERVED_DATA.items():
        old_name = 'fresh035' if name == 'observed035' else name
        if file_sha(path) != protocol['dataset_hashes'][old_name]:
            raise ValueError('JSON-035 observed prompt/target hash differs: ' + name)
        rescore(load(path), json.loads(REFERENCE_OUTPUTS['reference035'][name].read_text()))
    return {name: file_sha(path) for name, path in CACHE_FILES.items()}


def _round_robin_groups(rows, budget, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('contrast_group', row['id'])].append(row)
    strata = defaultdict(list)
    for gid, group in groups.items():
        group = sorted(group, key=lambda row: row['id'])
        strata[key(group[0])].append((hashlib.sha256(('json036-panel:' + gid).encode()).hexdigest(), group))
    for pool in strata.values():
        pool.sort(reverse=True)
    selected = []
    while len(selected) < budget:
        added = False
        for name in sorted(strata):
            while strata[name]:
                _, group = strata[name].pop()
                if len(selected) + len(group) <= budget:
                    selected.extend(group)
                    added = True
                    break
        if not added:
            break
    return selected


def observed_panel_category(row):
    """Coarse retained families keep large worlds from starving later lineages."""
    source = row['source']
    for prefix, lineage in (('grounded035_', '035'), ('consolidated034_', '034'),
                            ('intervention032_', '032')):
        if source.startswith(prefix):
            family = source[len(prefix):]
            if family.startswith('prose_'):
                return lineage, family
            if family.startswith('synthetic_relation_'):
                return lineage, 'relation'
            return lineage, family.split('_', 1)[0]
    if source.startswith('relation_'):
        family = 'relation'
    elif ('paragraph' in source or 'passage' in source or
          source.startswith(('binding_', 'fresh_'))):
        family = 'paragraph_complete' if previous.is_complete(row) else 'paragraph_atomic'
    elif source.startswith('natural_'):
        family = 'natural_instruction'
    elif source in ('shipping', 'support', 'line_items', 'observed_support'):
        family = 'typed_business'
    elif source in ('verbatim', 'dedicated_copying'):
        family = 'copy'
    elif source in ('associations', 'dedicated_associations'):
        family = 'association'
    elif source in ('corrections', 'dedicated_corrections'):
        family = 'correction'
    elif source in ('missing', 'dedicated_missing'):
        family = 'missing'
    elif source in ('dedicated_arrays', 'observed_empty_arrays'):
        family = 'array'
    elif source == 'dedicated_distractors':
        family = 'distractor'
    else:
        family = 'other'
    return 'legacy', family


def observed_panel(rows, budget=PANEL_OBSERVED_BUDGET):
    """Cover every retained lineage/family before filling remaining whole groups.

    A controlled world can cost 25 requests, so equal per-source iteration is
    insufficient. Reserve enough requests for a representative of each family,
    choose deterministic hash-ranked groups within that reservation, then fill
    from the least represented lineage/family that still fits the budget.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row.get('contrast_group') or row['id']].append(row)
    pools = defaultdict(list)
    for gid, group in groups.items():
        categories = {observed_panel_category(row) for row in group}
        if len(categories) != 1:
            raise ValueError('Observed contrast group crosses panel skill families')
        digest = hashlib.sha256(('json036-observed-panel:' + gid).encode()).hexdigest()
        pools[categories.pop()].append((digest, sorted(group, key=lambda row: row['id'])))
    for pool in pools.values():
        pool.sort(key=lambda item: item[0])
    categories = sorted(pools, key=lambda key: hashlib.sha256((':'.join(key)).encode()).hexdigest())
    minimum = {key: min(len(group) for _, group in pools[key]) for key in categories}
    if sum(minimum.values()) > budget:
        raise ValueError('Observed panel budget cannot cover every retained lineage/skill family')
    selected = []
    lineage_counts, family_counts = defaultdict(int), defaultdict(int)

    def take(key, index):
        _, group = pools[key].pop(index)
        selected.extend(group)
        lineage_counts[key[0]] += len(group)
        family_counts[key] += len(group)

    for position, key in enumerate(categories):
        reserved = sum(minimum[other] for other in categories[position + 1:])
        index = next(i for i, (_, group) in enumerate(pools[key])
                     if len(selected) + len(group) + reserved <= budget)
        take(key, index)
    while len(selected) < budget:
        feasible = [(lineage_counts[key[0]], family_counts[key], digest, key, index)
                    for key, pool in pools.items() for index, (digest, group) in enumerate(pool)
                    if len(selected) + len(group) <= budget]
        if not feasible:
            break
        _, _, _, key, index = min(feasible)
        take(key, index)
    return selected


def observed_panel_scope(rows):
    lineages, families, sources = defaultdict(int), defaultdict(int), defaultdict(int)
    for row in rows:
        lineage, family = observed_panel_category(row)
        lineages[lineage] += 1
        families[lineage + ':' + family] += 1
        sources[row['source']] += 1
    return dict(requests=len(rows), lineages=dict(sorted(lineages.items())),
        skill_families=dict(sorted(families.items())), sources=dict(sorted(sources.items())),
        scope='Fixed whole-group diagnostic sample; family coverage does not establish full-source retention.')


def development_panel(rows=None):
    """Fixed positive/null budgets, whole worlds and balanced source articles."""
    rows = development_rows() if rows is None else rows
    new_rows = [r for r in rows if is_new(r)]
    answerability = defaultdict(set)
    for row in new_rows:
        answerability[row['source_group']].update(b['answerability'] for b in row['field_bindings'])
    if any(values not in ({'answerable'}, {'answerable', 'unanswerable'}) for values in answerability.values()):
        raise ValueError('Panel requires positive or mixed answerable/null worlds, never all-null worlds')
    positive = [r for r in new_rows if answerability[r['source_group']] == {'answerable'}]
    mixed = [r for r in new_rows if answerability[r['source_group']] == {'answerable', 'unanswerable'}]
    if not positive or not mixed:
        raise ValueError('Diagnostic panel requires both declared positive and mixed-null strata')
    article = lambda r: r.get('provenance', {}).get('article', r['source'])
    chosen_positive = _round_robin_groups(positive, PANEL_POSITIVE_BUDGET, article)
    chosen_mixed = _round_robin_groups(mixed, PANEL_MIXED_NULL_BUDGET, article)
    if len(chosen_positive) != PANEL_POSITIVE_BUDGET or len(chosen_mixed) != PANEL_MIXED_NULL_BUDGET:
        raise ValueError('Fresh panel strata cannot fill their declared 48-request whole-world budgets')
    fresh = chosen_positive + chosen_mixed
    observed = observed_panel([r for r in rows if not is_new(r)])
    selected = fresh + observed
    if not selected or len(selected) > PANEL_REQUEST_LIMIT:
        raise ValueError('Empty or oversized diagnostic panel')
    return selected


def eval_rows(stage='full'):
    if stage == 'full':
        return development_rows()
    if stage == 'panel':
        return development_panel()
    raise ValueError('Unknown development stage')


def _matched_guard(rows, current, reference):
    ids = {o['id'] for o in reference['outputs']}
    selected, candidate = subset(rows, current, lambda r: r['id'] in ids)
    selected_ids = {r['id'] for r in selected}
    before = rescore(selected, dict(outputs=[o for o in reference['outputs'] if o['id'] in selected_ids]))
    qualified, candidate = qualified_subset(selected, candidate)
    _, before = qualified_subset(selected, before)
    return retained.strict_retention(qualified, before, candidate), qualification_scope(selected)


def common_complete_metrics(rows, result):
    selected, scored = subset(rows, result, lambda r: not is_new(r) and previous.is_complete(r) and qualification_exclusion(r) is None)
    return dict(**metrics(selected, scored), groups=retained.complete_groups(selected, scored),
        scope='Exact observed complete requests; each source has equal weight. Atomic diagnostics never enter complete ranking.')


def binding_diagnostics(rows, result):
    answers = {(o['id'], o['mode']): o for o in result['outputs']}
    by_mode = {}
    for mode in MODES:
        totals = dict(answerable_leaves=0, answerable_correct=0, unanswerable_leaves=0,
            null_true_positive=0, null_false_positive=0, null_false_negative=0,
            answerable_atomic=0, answerable_atomic_correct=0)
        questions = defaultdict(list)
        for row in rows:
            parsed = retained.parsed_leaves(answers[row['id'], mode]['text'])
            for binding in row['field_bindings']:
                path = tuple(binding['field_path'])
                expected = _at(row['target'], path)
                present = path in parsed
                value = parsed.get(path)
                correct = present and previous.retained.equal_json(value, expected)
                unanswerable = binding['answerability'] == 'unanswerable'
                totals['unanswerable_leaves' if unanswerable else 'answerable_leaves'] += 1
                if unanswerable:
                    totals['null_true_positive'] += present and value is None
                    totals['null_false_negative'] += not present or value is not None
                else:
                    totals['answerable_correct'] += correct
                    totals['null_false_positive'] += present and value is None
                    if row['task_kind'] == 'atomic':
                        totals['answerable_atomic'] += 1
                        totals['answerable_atomic_correct'] += correct
                questions[row['source_group'], binding['question_id']].append((row['task_kind'] == 'atomic', correct))
        paired = dict(questions=len(questions), paired_questions=0, atomic_correct=0,
            atomic_correct_but_complete_error=0, complete_errors_with_correct_atomic=0, all_views_correct=0)
        for values in questions.values():
            atomic = [correct for is_atomic, correct in values if is_atomic]
            complete = [correct for is_atomic, correct in values if not is_atomic]
            if not atomic or not complete:
                continue
            if len(atomic) != 1:
                raise ValueError('Paired diagnostic requires exactly one atomic anchor per world/question')
            paired['paired_questions'] += 1
            paired['atomic_correct'] += atomic[0]
            paired['atomic_correct_but_complete_error'] += atomic[0] and not all(complete)
            paired['complete_errors_with_correct_atomic'] += sum(atomic[0] and not correct for correct in complete)
            paired['all_views_correct'] += all(atomic + complete)
        tp, fp, fn = (totals[k] for k in ('null_true_positive', 'null_false_positive', 'null_false_negative'))
        totals.update(null_precision=tp / (tp + fp) if tp + fp else 0.,
            null_recall=tp / (tp + fn) if tp + fn else 0.,
            answerable_leaf_accuracy=totals['answerable_correct'] / totals['answerable_leaves'] if totals['answerable_leaves'] else 0.,
            answerable_atomic_accuracy=totals['answerable_atomic_correct'] / totals['answerable_atomic'] if totals['answerable_atomic'] else 0., paired_binding=paired)
        by_mode[mode] = totals
    return dict(by_mode=by_mode, scope='Original answerable and unanswerable questions scored separately; missing paths are failures, not null predictions. Paired errors never disappear inside an aggregate.')


def fresh_metrics(rows, result):
    fresh, scored = subset(rows, result, is_new)
    blocks = {}
    for block in BLOCKS:
        selected, block_result = subset(fresh, scored, lambda r: r['skill_block'] == block)
        blocks[block] = metrics(selected, block_result)
        counted = scope_metrics(selected, block_result)
        for mode in MODES:
            blocks[block]['by_mode'][mode]['macro_gold_leaf_accuracy'] = counted['by_mode'][mode]['macro_gold_leaf_accuracy']
    complete, complete_result = subset(fresh, scored, lambda r: r['is_complete_record'])
    return dict(blocks=blocks, balanced_complete_exact=blocks['natural_complete']['macro_exact'],
        complete_groups=retained.complete_groups(complete, complete_result), binding=binding_diagnostics(fresh, scored),
        atomic_scope='Single-question foundations never count as complete extraction.',
        independence_scope='New source worlds only; observed historical controls are separately identified. No old-model gain is claimed on fresh036 without matched old outputs.')


def _fresh_gates(fresh):
    gates = {}
    for mode in MODES:
        for block, floor in FRESH_FLOORS.items():
            gates[mode + ':' + block + '_floor'] = fresh['blocks'][block]['by_mode'][mode]['macro_exact'] >= floor - 1e-12
        binding = fresh['binding']['by_mode'][mode]
        gates[mode + ':answerable_atomic_floor'] = binding['answerable_atomic'] > 0 and binding['answerable_atomic_accuracy'] >= ANSWERABLE_ATOMIC_FLOOR - 1e-12
        gates[mode + ':null_precision_floor'] = binding['unanswerable_leaves'] > 0 and binding['null_precision'] >= NULL_PRECISION_FLOOR - 1e-12
        gates[mode + ':null_recall_floor'] = binding['unanswerable_leaves'] > 0 and binding['null_recall'] >= NULL_RECALL_FLOOR - 1e-12
    return gates


def scope_metrics(rows, result):
    """Count every current output without hiding failures inside macro rates."""
    answers = {(o['id'], o['mode']): o for o in result['outputs']}
    overall, sources, tasks, datasets, world_classes, pairs = {}, {}, {}, {}, {}, {}
    groups = defaultdict(list)
    fresh_answerability = defaultdict(set)
    for row in rows:
        groups[row['source'], row['source_group']].append(row)
        if is_new(row):
            fresh_answerability[row['source_group']].update(binding['answerability'] for binding in row['field_bindings'])
    for mode in MODES:
        counters = defaultdict(lambda: defaultdict(int))
        for row in rows:
            output = answers[row['id'], mode]
            checks = output['checks']
            gold = retained.leaves(row['target'])
            actual = retained.parsed_leaves(output['text'])
            correct = sum(path in actual and retained.equal_json(value, actual[path]) for path, value in gold.items())
            if is_new(row):
                dataset = row.get('provenance', {}).get('dataset', 'unspecified')
                world_class = 'mixed_null' if 'unanswerable' in fresh_answerability[row['source_group']] else 'positive_only'
            else:
                dataset = world_class = 'legacy'
            for key in (('overall', 'all'), ('source', row['source']), ('task', row.get('task_kind', 'legacy')),
                        ('dataset', dataset), ('world_class', world_class)):
                count = counters[key]
                count['requests'] += 1
                count['correct_exact'] += bool(checks['exact_json'])
                count['valid_json'] += bool(checks['valid_json'])
                count['schema_valid'] += bool(checks['schema_valid'])
                count['natural_stop'] += bool(checks['natural_stop'])
                count['expected_typed_leaves'] += len(gold)
                count['correct_typed_leaves'] += correct
        converted = {}
        for key, count in counters.items():
            value = dict(count)
            value['exact_rate'] = value['correct_exact'] / value['requests']
            value['typed_leaf_accuracy'] = value['correct_typed_leaves'] / value['expected_typed_leaves']
            converted[key] = value
        overall[mode] = converted['overall', 'all']
        sources[mode] = {name: value for (kind, name), value in converted.items() if kind == 'source'}
        overall[mode]['macro_gold_leaf_accuracy'] = sum(value['typed_leaf_accuracy'] for value in sources[mode].values()) / len(sources[mode])
        tasks[mode] = {name: value for (kind, name), value in converted.items() if kind == 'task'}
        datasets[mode] = {name: value for (kind, name), value in converted.items() if kind == 'dataset'}
        world_classes[mode] = {name: value for (kind, name), value in converted.items() if kind == 'world_class'}
        pair_counts = defaultdict(lambda: dict(groups=0, correct=0, noncontrasting_groups=0))
        for (source, _), group in groups.items():
            values = pair_counts[source]
            if (len(group) < 2 or len({row['document'] for row in group}) != 1
                    or len({canonical(row['target']) for row in group}) < 2):
                values['noncontrasting_groups'] += 1
            else:
                values['groups'] += 1
                values['correct'] += all(answers[row['id'], mode]['checks']['exact_json'] for row in group)
        pairs[mode] = {source: dict(values, exact_rate=values['correct']/values['groups'] if values['groups'] else None)
                       for source, values in pair_counts.items()}
    return dict(by_mode=overall, by_source=sources, by_task=tasks, by_dataset=datasets, by_world_class=world_classes,
        macro_exact_json=result['macro_exact_json'], macro_field_accuracy=result['macro_field_accuracy'],
        macro_paired_exact=result['macro_paired_exact'], paired_sources=pairs,
        legacy_sources=result['sources'])


def print_metrics(result, stage=None, step=None):
    """Full compact metric breakdown; deliberately omit individual error IDs."""
    stage = stage or result.get('stage', 'final')
    step = result.get('step', result.get('candidate_step')) if step is None else step
    def ratio(correct, denominator):
        return f'{correct}/{denominator} ({correct/denominator:.2%})' if denominator else 'n/a (0 eligible)'
    print(f'\n[JSON036 evaluation step={step} stage={stage}]', flush=True)
    print('Scope: diagnostic matched subset only; does not establish full retention.' if stage == 'panel'
          else 'Scope: full declared suite; finite research checks, no release/perfection claim.', flush=True)
    if 'balanced_complete_exact' in result:
        print(f'Ranking: balanced complete={result["balanced_complete_exact"]:.2%}; all-view exact={result["all_views_exact"]:.2%}; '
              f'qualified source-macro gold leaf={result["macro_gold_leaf_accuracy"]:.2%}; '
              f'requests={result.get("request_count")}; source groups={result.get("source_group_count")}', flush=True)
    if 'diagnostic_panel_scope' in result:
        print('Observed diagnostic coverage: ' + json.dumps(result['diagnostic_panel_scope'], sort_keys=True), flush=True)
    counted = result.get('scope_metrics')
    if counted:
        print('mode | exact/requests | typed leaves/expected | JSON | schema | natural stops', flush=True)
        for mode, values in counted['by_mode'].items():
            n = values['requests']
            print(f'{mode} | {values["correct_exact"]}/{n} ({values["exact_rate"]:.2%}) | '
                  f'{values["correct_typed_leaves"]}/{values["expected_typed_leaves"]} ({values["typed_leaf_accuracy"]:.2%}) | '
                  f'{values["valid_json"]}/{n} | {values["schema_valid"]}/{n} | {values["natural_stop"]}/{n}', flush=True)
        pair_denominator = sum(value['groups'] for mode in counted['paired_sources'].values() for value in mode.values())
        paired_macro = f'{counted["macro_paired_exact"]:.2%}' if pair_denominator else 'n/a (0 eligible groups)'
        print(f'Source macro exact={counted["macro_exact_json"]:.2%}; legacy field={counted["macro_field_accuracy"]:.2%}; paired exact={paired_macro}', flush=True)
        for table_name, table in (('source', counted['by_source']), ('task', counted['by_task']),
                                 ('dataset', counted['by_dataset']), ('world class', counted['by_world_class'])):
            print(f'{table_name} | mode | exact/n | typed leaves/n | legacy field | JSON/schema/stop', flush=True)
            for name in sorted(set().union(*(values.keys() for values in table.values()))):
                for mode in MODES:
                    values = table[mode][name]
                    n = values['requests']
                    legacy = counted['legacy_sources'][mode].get(name, {}).get('field_accuracy') if table_name == 'source' else None
                    legacy_text = f'{legacy:.2%}' if legacy is not None else '-'
                    print(f'{name} | {mode} | {values["correct_exact"]}/{n} ({values["exact_rate"]:.2%}) | '
                          f'{values["correct_typed_leaves"]}/{values["expected_typed_leaves"]} ({values["typed_leaf_accuracy"]:.2%}) | '
                          f'{legacy_text} | {values["valid_json"]}/{values["schema_valid"]}/{values["natural_stop"]} of {n}', flush=True)
        print('paired source | mode | all requests correct/groups | noncontrasting groups', flush=True)
        for mode in MODES:
            for source, values in sorted(counted['paired_sources'][mode].items()):
                print(f'{source} | {mode} | {ratio(values["correct"], values["groups"])} | {values.get("noncontrasting_groups", 0)}', flush=True)
    fresh = result.get('fresh_metrics', result.get('fresh_candidate_metrics'))
    if fresh:
        print('fresh block | mode | exact/n | source macro exact | source-macro gold leaf', flush=True)
        for name, values in fresh['blocks'].items():
            for mode, item in values['by_mode'].items():
                print(f'{name} | {mode} | {item["correct"]}/{item["requests"]} | {item["macro_exact"]:.2%} | {item["macro_gold_leaf_accuracy"]:.2%}', flush=True)
        for mode, item in fresh['binding']['by_mode'].items():
            precision = ratio(item['null_true_positive'], item['null_true_positive']+item['null_false_positive'])
            recall = ratio(item['null_true_positive'], item['null_true_positive']+item['null_false_negative'])
            print(f'{mode} answerable: leaves={item["answerable_correct"]}/{item["answerable_leaves"]}; '
                  f'atomic={item["answerable_atomic_correct"]}/{item["answerable_atomic"]}; '
                  f'null TP/FP/FN={item["null_true_positive"]}/{item["null_false_positive"]}/{item["null_false_negative"]}, '
                  f'precision={precision}, recall={recall}, '
                  f'unanswerable leaves={item["unanswerable_leaves"]}; paired={json.dumps(item["paired_binding"], sort_keys=True)}', flush=True)
        for mode, values in fresh['complete_groups']['by_mode'].items():
            print(f'{mode} fresh all-view worlds={ratio(values["correct"], values["groups"])}', flush=True)
    comparisons = result.get('common_cached_comparison', result.get('common_complete_comparison', {}))
    print('matched complete source macros: ' + json.dumps({label: {who: value['by_mode'] for who, value in values.items()}
        for label, values in comparisons.items()} if comparisons and 'reference034' in comparisons and 'by_mode' not in comparisons['reference034']
        else {label: values['by_mode'] for label, values in comparisons.items()}, sort_keys=True), flush=True)
    guards = result.get('strict_retention', result.get('strict_reference_retention', {}))
    for name, scope in result.get('reference_qualification_scope', {}).items():
        print(f'qualification {name}: original={scope["original_requests"]}, protected={scope["qualified_requests"]}, '
              f'predeclared exclusions={scope["excluded_requests"]}', flush=True)
    for reference, value in guards.items():
        suites = {'development': value} if 'by_mode' in value else value
        for suite, guard in suites.items():
            for mode, values in guard['by_mode'].items():
                sources = guard['per_source'][mode].values()
                gained_leaves = sum(v['candidate_correct_leaves']-v['baseline_correct_leaves'] for v in sources)+len(values['lost_leaves'])
                groups = guard['complete_groups']
                print(f'{reference}/{suite}/{mode}: exact gained/lost={len(values["gained_exact"])}/{len(values["lost_exact"])} '
                      f'protected={values["protected_exact"]}; typed leaves gained/lost={gained_leaves}/{len(values["lost_leaves"])} '
                      f'protected={values["protected_leaves"]}; JSON/schema/stop lost='
                      f'{len(values["lost_valid_json"])}/{len(values["lost_schema_valid"])}/{len(values["lost_natural_stop"])}; '
                      f'all-view groups before={ratio(groups["parent"]["by_mode"][mode]["correct"], groups["parent"]["by_mode"][mode]["groups"])} '
                      f'after={ratio(groups["candidate"]["by_mode"][mode]["correct"], groups["candidate"]["by_mode"][mode]["groups"])}; guard={guard["passed"]}', flush=True)
    gates = dict(result.get('common_complete_improvement_checks', {}), **result.get('fresh_skill_floor_checks', {}), **result.get('research_gates', {}))
    for name, passed in sorted(gates.items()):
        print(f'gate {name}: {"PASS" if passed else "FAIL"}', flush=True)
    print(f'DEV qualified={result.get("development_gate_pass", False)}; strict={result.get("strict_retention_pass", result.get("strict_no_regression", False))}; '
          f'research improved={result.get("research_improved", False)}; release_ready=False; promoted=False\n', flush=True)


def development_metrics(rows, result, step, references, stage='full'):
    if step <= 0:
        raise ValueError('Only positive trained checkpoints can be evaluated')
    if set(references) != set(REFERENCE_OUTPUTS):
        raise ValueError('All three trained references are required')
    result = rescore(rows, result)
    observed, current = subset(rows, result, lambda r: not is_new(r))
    if stage == 'full':
        # JSON035 alone covers the entire observed654; earlier caches have smaller exact scopes.
        rescore(observed, references['reference035'])
        if not {o['id'] for o in references['reference034']['outputs']} <= {r['id'] for r in observed}:
            raise ValueError('Full development omitted primary reference requests')
        if not {o['id'] for o in references['reference032']['outputs']} <= {r['id'] for r in observed}:
            raise ValueError('Full development omitted secondary reference requests')
    guards, scopes = {}, {}
    for name, reference in references.items():
        guards[name], scopes[name] = _matched_guard(observed, current, reference)
    primary_ids = {o['id'] for o in references['reference034']['outputs']}
    primary_rows, primary_current = subset(observed, current, lambda r: r['id'] in primary_ids)
    selected_ids = {r['id'] for r in primary_rows}
    before = rescore(primary_rows, dict(outputs=[o for o in references['reference034']['outputs'] if o['id'] in selected_ids]))
    common = dict(reference034=common_complete_metrics(primary_rows, before), candidate=common_complete_metrics(primary_rows, primary_current))
    fresh = fresh_metrics(rows, result)
    gains = {mode + ':common_complete_gain_5pp': common['candidate']['by_mode'][mode]['macro_exact'] >= common['reference034']['by_mode'][mode]['macro_exact'] + COMMON_COMPLETE_GAIN - 1e-12 for mode in MODES}
    floors = _fresh_gates(fresh)
    strict = all(g['passed'] for g in guards.values())
    qualified, qualified_result = qualified_subset(rows, result)
    return dict(step=step, split='development', stage=stage, diagnostic_only=stage == 'panel',
        **({'diagnostic_panel_scope': observed_panel_scope(observed)} if stage == 'panel' else {}),
        fresh_metrics=fresh, common_cached_comparison=common, strict_retention=guards,
        strict_retention_pass=strict, reference_qualification_scope=scopes,
        common_complete_improvement_checks=gains, fresh_skill_floor_checks=floors,
        development_gate_pass=stage == 'full' and strict and all(gains.values()) and all(floors.values()),
        balanced_complete_exact=(fresh['balanced_complete_exact'] + common['candidate']['macro_exact']) / 2,
        all_views_exact=fresh['complete_groups']['macro_exact'],
        macro_gold_leaf_accuracy=retained.gold_leaf_accuracy(qualified, qualified_result),
        scope_metrics=scope_metrics(rows, result),
        qualification_scope='Panel results protect only their explicit matched panel scope. Only full DEV can qualify. Finite zero-loss research qualification, not release certification.')


def selection(result):
    if result.get('stage') != 'full' or result['step'] <= 0:
        raise ValueError('Only positive full-development receipts select checkpoints')
    return [int(result['development_gate_pass']), result['balanced_complete_exact'], result['all_views_exact'], result['macro_gold_leaf_accuracy']]


def _stage(output, step, stage=None):
    plan = json.loads((Path(output) / 'session.json').read_text())['plan']
    full = set(plan.get('full_dev_steps', []))
    early = set(plan.get('early_evaluation_steps', []))
    actual = 'full' if step in full else 'panel' if step in early else None
    if actual is None or (stage is not None and stage != actual):
        raise ValueError('Evaluation step/stage differs from frozen plan')
    return actual


def development_protocol(rows, identity, step, stage):
    return dict(**identity, step=step, split='development', stage=stage, requests=len(rows),
        rows_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
        temperature=0, repetition_penalty=1, max_tokens=512,
        decoder_revision=previous.DECODER_REVISION, truncation=False, accepted_reference=reference_identity())


def checked_development(rows, saved, references):
    rescored = rescore(rows, saved)
    actual = development_metrics(rows, rescored, saved['step'], references, saved['stage'])
    if any(saved.get(key) != value for key, value in actual.items()) or saved['macro_field_accuracy'] != rescored['macro_field_accuracy']:
        raise ValueError('Development metrics/selection do not reproduce from saved strings')
    return dict(rescored, **actual)


def _receipt(output, step, stage, prior, identity, rows, references):
    if any(key in prior and prior[key] != value for key, value in identity.items()):
        raise ValueError('Existing development checkpoint identity/hash differs')
    if 'step' in prior:
        if prior['step'] != step or prior.get('stage') != stage or any(prior.get(key) != value for key, value in identity.items()):
            raise ValueError('Completed development receipt identity differs')
        return checked_development(rows, prior, references)
    result = rescore(rows, prior)
    result.update(**development_metrics(rows, result, step, references, stage), **identity)
    atomic_json(Path(output) / f'evaluations/development_step_{step:07d}.json', result)
    return result


def _checked_saved(output, step, required_stage):
    output = Path(output)
    stage = _stage(output, step, required_stage)
    identity = checkpoint_identity(output, step)
    rows = eval_rows(stage)
    path = output / f'evaluations/development_step_{step:07d}.json'
    expected = development_protocol(rows, identity, step, stage)
    protocol = path.with_suffix('.protocol.json')
    if not path.exists() or not protocol.exists() or json.loads(protocol.read_text()) != expected:
        raise ValueError('Saved development protocol differs or is missing')
    refs = {name: cached_development(name) for name in REFERENCE_OUTPUTS}
    return _receipt(output, step, stage, json.loads(path.read_text()), identity, rows, refs)


def checked_panel(output, step):
    return _checked_saved(output, step, 'panel')


def checked_selection(output, step):
    return selection(_checked_saved(output, step, 'full'))


def evaluate(model, tok, step, output, *, stage=None):
    if step <= 0:
        raise ValueError('JSON-036 skips initial base generation')
    output = Path(output)
    stage = _stage(output, step, stage)
    initialize_reference(output)
    identity = checkpoint_identity(output, step)
    rows = eval_rows(stage)
    path = output / f'evaluations/development_step_{step:07d}.json'
    protocol = path.with_suffix('.protocol.json')
    if path.exists() and not protocol.exists():
        raise ValueError('Existing outputs have no pre-generation checkpoint protocol')
    retained.commit(protocol, development_protocol(rows, identity, step, stage))
    refs = {name: cached_development(name) for name in REFERENCE_OUTPUTS}
    prior = json.loads(path.read_text()) if path.exists() else assess(model, tok, rows, path)
    result = _receipt(output, step, stage, prior, identity, rows, refs)
    atomic_json(output / 'last_eval.json', {key: value for key, value in result.items() if key != 'outputs'})
    print_metrics(result, stage, step)
    return selection(result) if stage == 'full' else None


def _check_run(output):
    from .data import check, PARENT_SHA
    session = json.loads((output / 'session.json').read_text())
    if json.loads((output / 'result.json').read_text())['status'] != 'complete':
        raise ValueError('Final testing requires a complete run')
    checked = check()
    manifest_sha = checked.get('manifest_sha256')
    if manifest_sha is None:
        manifest_sha = file_sha(CORPUS / 'manifest.json')
    if manifest_sha != session['contract']['corpus_manifest']:
        raise ValueError('Frozen JSON-036 corpus changed')
    for name, digest in session['contract']['code'].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError('Frozen code changed: ' + name)
    if verify_cache_binding() != session['contract']['cache_files']:
        raise ValueError('Frozen retained-reference caches changed')
    if file_sha(output / 'parent/model.safetensors') != PARENT_SHA:
        raise ValueError('Staged pretrained parent changed')
    return session


def final_test(output, diagnostic_final=False):
    """Qualified candidate only by default; failed diagnostic finals are opt-in."""
    output = Path(output)
    session = _check_run(output)
    initialize_reference(output)
    choices = []
    for step in session['plan']['full_dev_steps']:
        path = output / f'evaluations/development_step_{step:07d}.json'
        if path.exists():
            rank = checked_selection(output, step)
            saved = json.loads(path.read_text())
            identity = checkpoint_identity(output, step)
            choices.append((rank, -step, step, identity, saved))
    if not choices:
        raise ValueError('No positive full-development checkpoint receipt')
    best = max(choices)
    eligible = [choice for choice in choices if choice[4]['development_gate_pass']]
    selected = max(eligible) if eligible else best
    _, _, step, checkpoint, dev = selected
    bundle = output / checkpoint['bundle_name']
    pointer = dict(format='sml-pretrain-bundle-v1', bundle=bundle.name)
    previous._pointer(output, 'best_trained.json', pointer)
    if eligible:
        previous._pointer(output, 'best.json', pointer)
    elif (output / 'best.json').exists():
        raise ValueError('Ineligible run has a qualified best pointer')
    suites = {'fresh036': validate_fresh(load(CORPUS / 'test.jsonl'), 'test'), **{name: load(path) for name, path in OBSERVED_DATA.items()}}
    counts = {name: len(rows) for name, rows in suites.items()}
    selection_path = output / 'final-test/selection.json'
    existing_selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    diagnostic_authorized = bool(diagnostic_final or existing_selection.get('diagnostic_final'))
    identity = dict(candidate=str(bundle / 'model.safetensors'), candidate_step=step,
        candidate_sha256=checkpoint['model_sha256'], metadata_sha256=checkpoint['metadata_sha256'],
        accepted_reference=reference_identity(), development_gate_pass=dev['development_gate_pass'],
        development_strict_retention_pass=dev['strict_retention_pass'], diagnostic_final=diagnostic_authorized and not eligible)
    if not eligible and not diagnostic_authorized:
        result = dict(**identity, status='skipped_development_not_qualified', counts_expected=counts,
            generated_outputs=0, coverage_complete=False, research_improved=False,
            release_ready=False, promoted=False,
            scope='No final predictions requested because full development did not qualify. Explicit --diagnostic-final may test this same failed candidate later; TEST never selects.')
        atomic_json(output / 'final-test/result.json', result)
        return result
    hashes = {'fresh036': file_sha(CORPUS / 'test.jsonl'), **{name: file_sha(path) for name, path in OBSERVED_DATA.items()}}
    if hashes != session['contract']['final_test_files']:
        raise ValueError('Final test files differ from frozen contract')
    protocol = dict(**identity, contract=session['contract'], counts=counts, dataset_hashes=hashes,
        temperature=0, repetition_penalty=1, max_tokens=512,
        decoder_revision=previous.DECODER_REVISION, truncation=False,
        selection='Full development alone committed this candidate. Final scores can reject it, never select another checkpoint.')
    protocol_path = output / 'final-test/protocol.json'
    existing_candidates = list((output / 'final-test').glob('candidate-*.json'))
    if existing_candidates and (not selection_path.exists() or not protocol_path.exists()):
        raise ValueError('Existing final outputs have no committed pre-generation selection/protocol')
    retained.commit(output / 'final-test/selection.json', identity)
    retained.commit(output / 'final-test/protocol.json', protocol)
    results = {}
    for name, rows in suites.items():
        for label, paths in REFERENCE_OUTPUTS.items():
            if name in paths:
                results[label + '-' + name] = rescore(rows, json.loads(paths[name].read_text()))
    model = None
    for name, rows in suites.items():
        key = 'candidate-' + name
        path = output / f'final-test/{key}.json'
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            if model is None:
                import mlx.core as mx
                mx.set_default_device(mx.gpu)
                mx.set_cache_limit(4 * 1024 ** 3)
                tok = load_tokenizer()
                model = load_model(bundle / 'model.safetensors')
            saved = assess(model, tok, rows, path)
        results[key] = rescore(rows, saved)
    if model is not None:
        del model
        mx.clear_cache()
    guards = {name: {} for name in REFERENCE_OUTPUTS}
    qualification = {}
    for name, rows in suites.items():
        if name == 'fresh036':
            continue
        qualification[name] = qualification_scope(rows)
        for label in guards:
            if label + '-' + name in results:
                guards[label][name], _ = _matched_guard(rows, results['candidate-' + name], results[label + '-' + name])
    common = {}
    for label, paths in REFERENCE_OUTPUTS.items():
        names = [name for name in OBSERVED_DATA if name in paths]
        observed = [row for name in names for row in suites[name]]
        common[label] = {who: common_complete_metrics(observed, rescore(observed, dict(outputs=[o for name in names for o in results[who + '-' + name]['outputs']]))) for who in (label, 'candidate')}
    fresh = fresh_metrics(suites['fresh036'], results['candidate-fresh036'])
    gates = {label + ':' + name + ':strict_no_regression': guard['passed'] for label, sets in guards.items() for name, guard in sets.items()}
    primary = common['reference034']
    gates.update({mode + ':common_complete_gain_5pp': primary['candidate']['by_mode'][mode]['macro_exact'] >= primary['reference034']['by_mode'][mode]['macro_exact'] + COMMON_COMPLETE_GAIN - 1e-12 for mode in MODES})
    gates.update(_fresh_gates(fresh))
    gates['development_gate_pass'] = dev['development_gate_pass']
    all_rows = [row for rows in suites.values() for row in rows]
    all_current = rescore(all_rows, dict(outputs=[o for name in suites for o in results['candidate-' + name]['outputs']]))
    result = dict(**identity, status='complete', counts=counts,
        coverage_complete=all(value['coverage_complete'] for value in results.values()),
        strict_reference_retention=guards, reference_qualification_scope=qualification,
        strict_no_regression=all(guard['passed'] for sets in guards.values() for guard in sets.values()),
        common_complete_comparison=common, fresh_candidate_metrics=fresh,
        research_gates=gates, research_improved=all(gates.values()), release_ready=False, promoted=False,
        scope_metrics=scope_metrics(all_rows, all_current),
        metrics={key: previous._summary(value) for key, value in results.items()},
        scope='Current candidate only; all historical exact observed suites plus unseen036. Triple-reference zero-loss and research floors are finite protocol checks, not perfection, unseen-input guarantees, or release certification.')
    atomic_json(output / 'final-test/result.json', result)
    print_metrics(result, 'final', step)
    return result
