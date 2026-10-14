"""JSON-036: one anchored SFT pass with measured microbatch padding."""
import argparse
from dataclasses import asdict
import gzip
import json
import math
from pathlib import Path
import signal
import time
import numpy as np
from sml_v1.common import atomic_json, fingerprint
from sft.data import Corpus, learning_rate
from json_sft.corpus import load_tokenizer
from .data import load
from .evaluate import evaluate, final_test, initialize_reference, checked_selection, checked_panel
from .objective import loss, OBJECTIVE
from .order import batches_for, evaluation_steps, SAMPLER
from .storage import prune_transient
from .runtime import iter_microbatches, RUNTIME_FORMAT


def checkpoint_for_step(output, step):
    """Every evaluated step has exactly one committed bundle from this run."""
    output = Path(output)
    contract = json.loads((output / 'session.json').read_text())['contract']
    candidates = []
    for path in output.glob(f'step_{step:07d}_*'):
        metadata = path / 'model.safetensors.json'
        if path.is_symlink() or not metadata.exists():
            continue
        saved = json.loads(metadata.read_text())
        if saved['step'] == step and saved.get('json_sft', {}).get('contract') == contract:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError('Expected exactly one committed checkpoint for evaluated step')
    return candidates[0]


def ranked_receipts(output, step, points, full_points):
    """Rescore actual strings; a missing current receipt is recoverable in place."""
    diagnostic = eligible = None
    diagnostic_step = eligible_step = None
    recover = False
    for point in sorted(p for p in points if p <= step):
        try:
            if point not in full_points:
                checked_panel(output, point)
                continue
            selection = checked_selection(output, point)
        except (OSError, KeyError, ValueError) as error:
            if point == step:
                recover = True
                continue
            raise ValueError('Missing or invalid earlier development receipt') from error
        if diagnostic is None or selection > diagnostic:
            diagnostic, diagnostic_step = selection, point
        if selection[0] and (eligible is None or selection > eligible):
            eligible, eligible_step = selection, point
    return diagnostic, eligible, diagnostic_step, eligible_step, recover


def recover_committed_latest(output, contract, order_hash, total_steps, resolve):
    """Recover a complete bundle committed just before pointer publication.

    The shared writer atomically commits all checkpoint bytes before latest.json.
    A process kill between those actions must restore the committed optimizer and
    cursor, rather than replay that update and create a duplicate evaluated step.
    Never infer a complete checkpoint from temporary files or a directory name.
    """
    output = Path(output).absolute()
    latest = output / 'latest.json'
    published = Path(resolve(latest)).parent if latest.exists() else None
    complete = []
    for bundle in sorted(output.glob('step_*_*')):
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError('Checkpoint bundle must be a real directory')
        weight = Path(resolve(bundle))
        if weight.parent != bundle:
            raise ValueError('Checkpoint resolver escaped current run')
        metadata = json.loads(Path(str(weight) + '.json').read_text())
        saved, point = metadata['json_sft'], metadata['step']
        if (type(point) is not int or not 0 <= point <= total_steps
                or not bundle.name.startswith(f'step_{point:07d}_')
                or saved['contract'] != contract or saved['order_hash'] != order_hash
                or saved['objective'] != OBJECTIVE):
            raise ValueError('Committed checkpoint differs from current run contract/order')
        with gzip.open(str(weight) + '.rank0.data_state.json.gz', 'rt') as stream:
            cursor = json.load(stream)
        if cursor != {'step': point, 'stream_state': {'next_batch': point}}:
            raise ValueError('Committed checkpoint cursor differs')
        complete.append((point, bundle, str(weight)))
    if not complete:
        raise ValueError('Resume requires a complete committed checkpoint')
    if published is not None and published not in {bundle for _, bundle, _ in complete}:
        raise ValueError('Published checkpoint is outside this run')
    point = max(item[0] for item in complete)
    newest = [item for item in complete if item[0] == point]
    if len(newest) != 1:
        raise ValueError('Ambiguous duplicate committed checkpoints at latest step')
    _, bundle, weight = newest[0]
    if published != bundle:
        atomic_json(latest, dict(format='sml-pretrain-bundle-v1', bundle=bundle.name))
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output
    session = json.loads((output / 'session.json').read_text())
    plan, contract = session['plan'], session['contract']
    if (plan['epochs'] != 1 or plan['sampler'] != SAMPLER or plan['adaptation'] != 'full'
            or plan['batch_size'] != 64 or plan['microbatch'] not in (2, 4, 8) or plan['parent_kind'] != 'pretrained'):
        raise ValueError('Unexpected fresh JSON-036 training policy')
    if 'teacher' in session or any(key.startswith('teacher_') for key in plan):
        raise ValueError('Fresh-base JSON-036 is CE only and has no teacher')
    corpus = Corpus(plan['corpus'], 'train')
    rows = load(Path(plan['corpus']) / 'train.jsonl')
    if len(corpus) != len(rows):
        raise ValueError('TRAIN rows/token arrays differ')
    order, boundaries = batches_for(rows, plan['seed'], epochs=plan['epochs'])
    order_hash = fingerprint([batch.tolist() for batch in order])
    points = evaluation_steps(boundaries)
    full_points = set(boundaries['full_dev_steps'])
    if plan['early_evaluation_steps'] != boundaries['early_evaluation_steps'] or plan['full_dev_steps'] != boundaries['full_dev_steps']:
        raise ValueError('Frozen evaluation points differ from prepared phase schedule')
    initialize_reference(output)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from sml_v1.checkpoint_bundle import resolve_bundle, save_bundle
    from sml_v1.model import TransformerConfig, TransformerLM
    from sml_v1.precision import MasterAdamW
    parent = json.loads((output / 'parent/model.safetensors.json').read_text())
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(4 * 1024 ** 3)
    mx.random.seed(plan['seed'])
    model = TransformerLM(TransformerConfig(**parent['recipe']['model']))
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), model.parameters()))
    model.load_weights(str(output / 'parent/model.safetensors'))
    optimizer = MasterAdamW(learning_rate=plan['peak_lr'], betas=tuple(plan['betas']),
                            eps=plan['eps'], weight_decay=plan['weight_decay'])
    step = tokens = records = 0
    best = accepted = None
    recover = False
    if args.resume:
        weight = recover_committed_latest(output, contract, order_hash, len(order), resolve_bundle)
        meta = json.loads(Path(weight + '.json').read_text())
        saved = meta['json_sft']
        if saved['contract'] != contract or saved['order_hash'] != order_hash or saved['objective'] != OBJECTIVE:
            raise ValueError('Resume contract/order/objective differs')
        step, tokens, records = meta['step'], meta['tokens'], saved['records']
        if type(step) is not int or not 0 <= step <= len(order):
            raise ValueError('Invalid resume step')
        done = np.concatenate(order[:step]) if step else np.array([], np.int64)
        expected = sum(int(corpus.offsets[i + 1] - corpus.offsets[i]) for i in done)
        with gzip.open(weight + '.rank0.data_state.json.gz', 'rt') as stream:
            cursor = json.load(stream)
        if records != len(done) or tokens != expected or cursor != {'step': step, 'stream_state': {'next_batch': step}}:
            raise ValueError('Resume cursor/counters differ')
        best, accepted, best_step, accepted_step, recover = ranked_receipts(output, step, points, full_points)
        model.load_weights(weight)
        optimizer.init(model.trainable_parameters())
        optimizer.state = tree_unflatten(list(mx.load(weight + '.optimizer.safetensors').items()))
        mx.eval(model.parameters(), optimizer.state)
        if int(optimizer.step.item()) != step:
            raise ValueError('Optimizer step differs')
    else:
        optimizer.init(model.trainable_parameters())
    model.train()
    mx.eval(model.parameters(), optimizer.state)
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    one = nn.value_and_grad(model, lambda x, y: loss(model, x, y))

    def pointer(name, point):
        bundle = checkpoint_for_step(output, point)
        resolve_bundle(bundle)
        atomic_json(output / f'{name}.json', dict(format='sml-pretrain-bundle-v1', bundle=bundle.name))

    def save():
        meta = dict(step=step, tokens=tokens, recipe={'model': asdict(model.cfg)}, json_sft=dict(
            format='opensml-json-sft-run-v1', contract=contract, parent=session['parent'],
            accepted_reference=session['accepted_reference'], records=records, total_steps=len(order),
            order_hash=order_hash, best_selection=best, eligible_selection=accepted,
            adaptation='full', sampler=SAMPLER, serialization='text-schema-plain-v1',
            objective=OBJECTIVE, loss_weighting='record-mean gold output/EOS CE; prompt/padding zero',
            evaluation_steps=sorted(points), phase_ends=boundaries['phase_ends'], runtime_format=RUNTIME_FORMAT, padding_policy=plan['padding_policy'], microbatch=plan['microbatch']))
        # Shared count pruning is disabled; own contract-aware policy owns rotation.
        keep = sum(1 for _ in output.glob('step_*_*')) + 1
        save_bundle(output, model, optimizer, meta, {'next_batch': step}, best=False, keep=keep, reserve_gib=10)
        prune_transient(output, points | {0}, step)

    def assess():
        nonlocal best, accepted
        candidate = evaluate(model, load_tokenizer(), step, output)
        if step not in full_points:
            if candidate is not None:
                raise ValueError('A diagnostic panel cannot select a checkpoint')
            checked_panel(output, step)
            prune_transient(output, points | {0}, step)
            return
        if candidate is None:
            raise ValueError('Full development evaluation must return a selection rank')
        # Trust neither cached summary flags nor the generator's return value.
        checked = checked_selection(output, step)
        if candidate != checked:
            raise ValueError('Development selection differs from rescored outputs')
        if best is None or candidate > best:
            best = candidate
            pointer('best_trained', step)
        if candidate[0] and (accepted is None or candidate > accepted):
            accepted = candidate
            pointer('best', step)
        prune_transient(output, points | {0}, step)

    if not args.resume:
        save()  # Recovery-only base checkpoint: no base generation or best pointer.
    else:
        if best_step is not None:
            pointer('best_trained', best_step)
        if accepted_step is not None:
            pointer('best', accepted_step)
        elif (output / 'best.json').exists():
            raise ValueError('Accepted pointer exists without an eligible positive receipt')
        if recover:
            assess()  # Exact midpoint/final resume only; current checkpoint is committed.
    began = time.monotonic()
    last_saved = step
    print(f'[ready-json-evidence-binding] fresh Stage-A; step={step}/{len(order)}; one anchored pass; microbatch={plan["microbatch"]} padding={plan["padding_policy"]}; diagnostic panels={boundaries["early_evaluation_steps"]}; full DEV={sorted(full_points)}', flush=True)
    while step < len(order):
        if stop[0] or (output / 'STOP').exists():
            break
        indices = order[step]
        total = None
        loss_total = 0.
        for x, y in iter_microbatches(corpus, indices, size=64,
                microbatch=plan['microbatch'], policy=plan['padding_policy']):
            value, gradients = one(mx.array(x), mx.array(y))
            gradients = tree_map(lambda gradient: gradient.astype(mx.float32), gradients)
            total = gradients if total is None else tree_map(lambda a, b: a + b, total, gradients)
            mx.eval(total, value)
            loss_total += float(value.item())
        total = tree_map(lambda gradient: gradient / len(indices), total)
        norm = mx.sqrt(sum(mx.sum(gradient * gradient) for _, gradient in sorted(tree_flatten(total))))
        mx.eval(total, norm)
        if not math.isfinite(float(norm.item())) or not math.isfinite(loss_total):
            raise RuntimeError('Nonfinite CE loss/gradient')
        total = tree_map(lambda gradient: gradient * mx.minimum(1., plan['grad_clip'] / (norm + 1e-6)), total)
        lr = learning_rate(step + 1, len(order), plan['peak_lr'], plan['final_lr'], plan['warmup_fraction'])
        optimizer.learning_rate = lr
        optimizer.update(model, total)
        mx.eval(model.parameters(), optimizer.state)
        tokens += sum(int(corpus.offsets[i + 1] - corpus.offsets[i]) for i in indices)
        records += len(indices)
        step += 1
        if step == 1 or step % 10 == 0:
            print(f'[json-evidence-binding {step}/{len(order)}] output_eos_record_ce={loss_total / len(indices):.4f} lr={lr:.3e} records={records} tokens={tokens} elapsed={time.monotonic() - began:.1f}s', flush=True)
        if step in points:
            save()  # Bind actual predictions to a durable, immutable model first.
            last_saved = step
            assess()  # Diagnostic panels report progress; full DEV qualification occurs once at the end.
        elif step % plan['checkpoint_every_steps'] == 0:
            save()
            last_saved = step
    if last_saved != step:
        save()
    status = 'complete' if step == len(order) else 'stopped'
    atomic_json(output / 'result.json', dict(status=status, step=step, total_steps=len(order), records=records, tokens=tokens))
    if status == 'complete':
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)
        print(json.dumps(final_test(output), indent=2), flush=True)


if __name__ == '__main__':
    main()
