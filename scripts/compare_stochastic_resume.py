#!/usr/bin/env python3
"""Compare every rank/micro tape of the first fresh-process R resume update.

This is the stochastic evidence component only. Full restored/checkpoint model,
optimizer, counters and CPU/XLA RNG comparisons remain separately required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now


ATOL, RTOL = 1e-7, 1e-6
TENSOR_FIELDS = ('actions', 'action_mask', 'lengths', 'behavior_logps',
                 'clean', 'thought', 'advantages')
FLOAT_FIELDS = ('behavior_logps', 'clean', 'thought', 'advantages')
META_FIELDS = ('schema', 'run_id', 'spec_hash', 'order_hash', 'arm', 'seed',
               'step', 'rank', 'world_size', 'micro', 'global_batch',
               'accumulation', 'record_id', 'ordered_record_index',
               'policy_version', 'temperature', 'decode_iterations')


def read_json(path):
    if path.is_symlink():
        raise ValueError(f'Symlink input: {path}')
    return json.loads(path.read_text())


def location(mask):
    return mask.nonzero()[0].tolist()


def compare_tapes(actual, expected):
    """Return the first mismatch in a fixed order, retaining numeric maxima."""
    result = dict(pass_=True, exact=True, first_divergence=None, numeric={})

    def fail(field, **detail):
        result['pass_'] = False
        if result['first_divergence'] is None:
            result['first_divergence'] = dict(field=field, **detail)

    for field in META_FIELDS:
        if type(actual[field]) is not type(expected[field]) or actual[field] != expected[field]:
            result['exact'] = False
            fail(field, kind='metadata', expected=expected[field], actual=actual[field])
    for field in TENSOR_FIELDS:
        a, e = actual[field], expected[field]
        if a.shape != e.shape or a.dtype != e.dtype:
            result['exact'] = False
            fail(field, kind='shape_or_dtype', expected_shape=list(e.shape),
                 actual_shape=list(a.shape), expected_dtype=str(e.dtype), actual_dtype=str(a.dtype))
            continue
        different = a.ne(e)
        if bool(different.any()):
            result['exact'] = False
        if field not in FLOAT_FIELDS:
            if bool(different.any()):
                index = location(different)
                fail(field, kind='exact', index=index, expected=e[tuple(index)].item(),
                     actual=a[tuple(index)].item())
            continue
        a, e = a.double(), e.double()
        finite = torch.isfinite(a) & torch.isfinite(e)
        if not bool(finite.all()):
            fail(field, kind='nonfinite', index=location(~finite))
            result['numeric'][field] = dict(nonfinite_values=int((~finite).sum()))
            continue
        absolute, tolerance = (a-e).abs(), ATOL + RTOL * e.abs()
        bad = absolute > tolerance
        result['numeric'][field] = dict(max_abs=float(absolute.max()),
            max_tolerance_fraction=float((absolute/tolerance).max()),
            different_values=int(different.sum()), violations=int(bad.sum()))
        if bool(bad.any()):
            index = location(bad)
            fail(field, kind='numeric', index=index, expected=e[tuple(index)].item(),
                 actual=a[tuple(index)].item(), tolerance=tolerance[tuple(index)].item())
    result['pass'] = result.pop('pass_')
    return result


def validate_tape(tape, *, run, order_hash, step, rank, micro):
    if not isinstance(tape, dict) or set(tape) != set(META_FIELDS + TENSOR_FIELDS):
        raise ValueError('Tape schema fields differ')
    spec = run['spec']
    world, batch = spec['world_size'], spec['global_prefixes']
    accumulation, cap = batch // world, spec['cap']
    expected = dict(schema='qualification-rollout-v1', run_id=run['run_id'],
        spec_hash=run['spec_hash'], order_hash=order_hash, arm='thought',
        seed=run['seed'], step=step, rank=rank, world_size=world, micro=micro,
        global_batch=batch, accumulation=accumulation, policy_version=step-1,
        ordered_record_index=(step-1)*batch + micro*world + rank, temperature=.7)
    for key, value in expected.items():
        if type(tape[key]) is not type(value) or tape[key] != value:
            raise ValueError(f'Tape identity mismatch: {key}')
    if not isinstance(tape['record_id'], str) or not tape['record_id']:
        raise ValueError('Empty tape record identity')
    iterations = tape['decode_iterations']
    if type(iterations) is not int or not 1 <= iterations <= cap:
        raise ValueError('Invalid decode iterations')
    shapes = dict(actions=(1,4,cap), action_mask=(1,4,cap), lengths=(1,4),
                  behavior_logps=(1,4,cap), clean=(1,), thought=(1,4), advantages=(1,4))
    for key, shape in shapes.items():
        value = tape[key]
        dtype = torch.float32 if key in FLOAT_FIELDS else (
            torch.bool if key == 'action_mask' else torch.int64)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f'Tape shape/dtype mismatch: {key}')
    mask = tape['action_mask']
    if not bool(mask[:,:,0].all()) or bool((~mask[:,:,:-1] & mask[:,:,1:]).any()):
        raise ValueError('Action mask is not a nonempty contiguous prefix')
    if not torch.equal(tape['lengths'], mask.sum(-1)):
        raise ValueError('Tape lengths differ from masks')
    if bool((tape['lengths'] > iterations).any()):
        raise ValueError('Action length exceeds decode iterations')


def compare_runs(baseline, resumed, *, resume_step, criteria_path):
    baseline, resumed = Path(baseline), Path(resumed)
    if baseline.resolve() == resumed.resolve():
        raise ValueError('Baseline and resumed outputs must be distinct')
    criteria = read_json(criteria_path)
    if criteria.get('atol') != ATOL or criteria.get('rtol') != RTOL:
        raise ValueError('Resume criteria must retain 1e-7 absolute and 1e-6 relative bounds')
    run = read_json(baseline/'run.json')
    if run != read_json(resumed/'run.json'):
        raise ValueError('Baseline and resume run identities differ')
    spec = run['spec']
    if (spec.get('stage') != 'qualification' or run.get('arm') != 'thought'
            or spec.get('arm') != 'thought' or spec.get('precision') != 'fp32'
            or run['spec_hash'] != canonical_hash(spec)
            or spec.get('seed') != run.get('seed')
            or baseline.name != resumed.name or baseline.name != run.get('run_id')):
        raise ValueError('Require the same immutable FP32 thought qualification run')
    world, batch = spec['world_size'], spec['global_prefixes']
    if (type(world) is not int or type(batch) is not int or world < 1 or batch % world
            or type(resume_step) is not int or not 1 <= resume_step < spec['steps']):
        raise ValueError('Invalid resume step/topology')
    checkpoint = f'step-{resume_step:07d}/COMPLETE.json'
    manifest = read_json(baseline/checkpoint)
    if manifest != read_json(resumed/checkpoint):
        raise ValueError('Resume fork checkpoint manifests differ')
    if any(manifest.get(k) != v for k,v in dict(schema='neuron-v1', step=resume_step,
            spec_hash=run['spec_hash'], run_id=run['run_id'], world_size=world).items()):
        raise ValueError('Resume checkpoint identity mismatch')
    metadata = manifest['metadata']
    if any(metadata.get(k) != v for k,v in dict(arm='thought', seed=run['seed'],
            global_batch=batch, accumulation=batch//world, spec_hash=run['spec_hash']).items()):
        raise ValueError('Resume checkpoint metadata mismatch')
    order_hash = metadata['order_hash']
    if not isinstance(order_hash,str) or len(order_hash) != 64:
        raise ValueError('Missing resume order identity')
    step, accumulation = resume_step+1, batch//world
    relative = Path('qualification-tapes')/f'step-{step:07d}'
    names = {f'rank-{r:05d}-micro-{m:03d}.pt'
             for r in range(world) for m in range(accumulation)}
    out = dict(schema='stochastic-resume-tapes-v1', created_utc=utc_now(),
        scope='First resumed update tapes only; full model/optimizer/RNG checks separately required',
        full_state_resume_qualified=False, baseline=str(baseline.resolve()),
        resumed=str(resumed.resolve()), resume_step=resume_step, compared_step=step,
        spec_hash=run['spec_hash'], order_hash=order_hash, world_size=world,
        accumulation=accumulation, criteria=criteria, criteria_sha256=sha256_file(criteria_path),
        source_sha256=sha256_file(Path(__file__)),
        baseline_run_sha256=sha256_file(baseline/'run.json'),
        resumed_run_sha256=sha256_file(resumed/'run.json'),
        checkpoint_manifest_sha256=sha256_file(baseline/checkpoint),
        expected_tape_count=world*accumulation, rows=[], first_divergence=None)
    for side, root in [('baseline',baseline),('resumed',resumed)]:
        directory = root/relative
        actual = {p.name for p in directory.iterdir()} if directory.is_dir() else set()
        if directory.is_symlink() or actual != names:
            out.update(status='FAIL', **{'pass':False}, first_divergence=dict(
                kind='coverage', side=side, missing=sorted(names-actual), unexpected=sorted(actual-names)))
            return out
    for rank in range(world):
        for micro in range(accumulation):
            name = f'rank-{rank:05d}-micro-{micro:03d}.pt'
            paths = [root/relative/name for root in (baseline,resumed)]
            row = dict(rank=rank, micro=micro, baseline_sha256=sha256_file(paths[0]),
                       resumed_sha256=sha256_file(paths[1]))
            try:
                if any(p.is_symlink() for p in paths) or paths[0].samefile(paths[1]):
                    raise ValueError('Resume tape aliases baseline evidence')
                tapes = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
                for tape in tapes:
                    validate_tape(tape, run=run, order_hash=order_hash, step=step, rank=rank, micro=micro)
                row.update(compare_tapes(tapes[1],tapes[0]))
            except (ValueError, RuntimeError, EOFError, KeyError, TypeError,
                    OSError, pickle.UnpicklingError) as error:
                row.update(**{'pass':False}, first_divergence=dict(kind='invalid_tape',error=str(error)))
            out['rows'].append(row)
            if not row['pass'] and out['first_divergence'] is None:
                out['first_divergence'] = dict(step=step,rank=rank,micro=micro,**row['first_divergence'])
    out['pass'] = out['first_divergence'] is None
    out['status'] = 'PASS' if out['pass'] else 'FAIL'
    out['all_tapes_bitwise_equal'] = out['pass'] and all(row['exact'] for row in out['rows'])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--resumed', type=Path, required=True)
    parser.add_argument('--resume-step', type=int, required=True)
    parser.add_argument('--criteria', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.baseline,args.resumed,resume_step=args.resume_step,
                          criteria_path=args.criteria)
    atomic_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'},indent=2))
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
