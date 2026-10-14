"""One anchored pass: human question anchors before their joint compositions."""
import math
import numpy as np

SAMPLER = 'evidence-binding-anchor-then-compose-v1'
SKILL_BLOCKS = frozenset(('foundation_atomic', 'natural_complete', 'controlled_complete',
                          'typed_instruction_complete', 'relation_complete'))


def _pass(rows, seed):
    rng = np.random.default_rng(seed)
    grouped = {}
    for index, row in enumerate(rows):
        block = row.get('skill_block')
        if block not in SKILL_BLOCKS:
            raise ValueError('Unknown TRAIN skill block')
        group = row.get('contrast_group') or row.get('source_group')
        if not isinstance(group, str) or not group:
            raise ValueError('Missing TRAIN source/contrast group')
        grouped.setdefault(block, {}).setdefault(group, []).append(index)
    streams = {}
    for block in sorted(grouped):
        keys = list(grouped[block])
        rng.shuffle(keys)
        stream = []
        for key in keys:
            ids = list(grouped[block][key])
            rng.shuffle(ids)
            stream.extend(ids)
        streams[block] = stream
    # Cumulative largest-deficit scheduling uses the corpus's declared skill
    # proportions, without oversampling a small block or leaving a skill tail.
    blocks = list(streams)
    rng.shuffle(blocks)
    counts = {block: len(streams[block]) for block in blocks}
    used = dict.fromkeys(blocks, 0)
    order = []
    for position in range(len(rows)):
        available = [block for block in blocks if used[block] < counts[block]]
        block = max(available, key=lambda key: (position + 1) * counts[key] / len(rows) - used[key])
        order.append(streams[block][used[block]])
        used[block] += 1
    if sorted(order) != list(range(len(rows))):
        raise ValueError('TRAIN pass membership changed')
    return order


def batches_for(rows, seed, batch_size=64, epochs=1):
    if batch_size != 64 or epochs != 1 or not rows:
        raise ValueError('Expected nonempty corpus, batch64 and one prepared pass')
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate TRAIN request IDs')
    if {row.get('training_phase') for row in rows} != {'foundation', 'binding'}:
        raise ValueError('Expected explicit foundation and binding TRAIN phases')
    if {row.get('skill_block') for row in rows} != SKILL_BLOCKS:
        raise ValueError('Expected natural anchors plus every retained skill block')
    batches = []
    phase_ends = {}
    for number, phase in enumerate(('foundation', 'binding')):
        members = [i for i, row in enumerate(rows) if row['training_phase'] == phase]
        local = _pass([rows[i] for i in members], seed + number)
        ordered = [members[i] for i in local]
        batches.extend(np.asarray(ordered[i:i + batch_size], np.int64)
                       for i in range(0, len(ordered), batch_size))
        phase_ends[phase] = len(batches)
    if sorted(int(i) for batch in batches for i in batch) != list(range(len(rows))):
        raise ValueError('One-pass TRAIN membership changed')
    early = sorted({min(64, len(batches)-1), phase_ends['foundation']} - {len(batches)})
    return batches, dict(final=len(batches), phase_ends=phase_ends,
                         early_evaluation_steps=early, full_dev_steps=[len(batches)])


def evaluation_steps(boundaries):
    return set(boundaries['early_evaluation_steps']) | set(boundaries['full_dev_steps'])
