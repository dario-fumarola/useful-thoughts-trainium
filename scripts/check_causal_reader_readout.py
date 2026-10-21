"""Independently reconstruct a completed frozen-selector predictive readout.

No backbone loader, note generator, optimizer or producer retry. Uses original
native caches, terminal head tensors and independent NumPy gate arithmetic.
Within-repository verification, not external replication or semantic validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np
import torch

from weekend_thinking.artifacts import atomic_json, sha256_file

ORDERS = (2026102131, 2026102132)


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, name):
    if not np.allclose(actual, expected, rtol=0, atol=1e-9):
        raise ValueError('Independent numerical readout mismatch: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'features', 'fitted', 'readout', 'runtime', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    a = parser.parse_args()
    torch.set_num_threads(4)
    deadline = datetime.fromisoformat(a.deadline_utc)

    def check():
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError('Finite independent readout-check deadline')

    p = read(a.protocol); digest = sha256_file(a.protocol)
    fitted = read(a.fitted/'complete.json'); report = read(a.readout/'readout.json')
    runtime = read(a.runtime)
    if (p['status'] != 'FROZEN_CAUSAL_READER_PROTOCOL'
            or fitted['status'] != 'COMPLETE_FOUR_TRAIN_ONLY_TERMINAL800_HEADS'
            or not fitted['no_DEV_cache_opened'] or len(fitted['heads']) != 4
            or report['status'] != 'COMPLETE_FROZEN_HEAD_PAIRED_READOUT'
            or runtime['status'] != 'ALL_SCIENCE_FEATURE_AND_TRAIN_HEAD_PROCESSES_CLOSED'
            or any(x['protocol_sha256'] != digest for x in (fitted, report, runtime))
            or report['fit_complete_sha256'] != sha256_file(a.fitted/'complete.json')
            or runtime['total_reserved_chip_hours_including_prior_qualification'] > 8):
        raise ValueError('Complete original frozen execution required')
    states = {}
    for role in ('T_note', 'T_raw'):
        for seed in ORDERS:
            key = f'{role}:{seed}'; binding = fitted['heads'][key]
            if sha256_file(binding['file']) != binding['sha256']:
                raise ValueError('Terminal head changed')
            state = torch.load(binding['file'], weights_only=True)
            states[key] = (state['weight'].numpy(), state['bias'].item())
            logs = [json.loads(line) for line in (Path(binding['file']).parent/'training.jsonl').open()]
            if [r['step'] for r in logs] != list(range(1, 801)):
                raise ValueError('Exactly800 original head updates required')

    def cache(cohort, role):
        n = p['data'][cohort]['rows']; h = p['hidden_size']
        hidden = np.empty((n, 16, h), np.float32)
        entropy = np.empty((n, 16), np.float32); lp = np.empty((n, 16), np.float64)
        records = [None]*n
        for part in range(8):
            check(); directory = a.features/cohort/role/f'part-{part}'
            meta = read(directory/'complete.json')
            if (meta['status'] != 'COMPLETE_FROZEN_READER_FEATURES'
                    or meta['protocol_sha256'] != digest or meta['cohort'] != cohort
                    or meta['role'] != role or meta['phase'] != 'science'
                    or not meta['execution_device'].startswith('xla')):
                raise ValueError('Original actual native feature cache required')
            for name, value in meta['files'].items():
                if sha256_file(directory/name) != value:
                    raise ValueError('Original native array changed')
            index = [json.loads(line) for line in (directory/'index.jsonl').open()]
            arrays = [np.load(directory/(name+'.npy'), allow_pickle=False, mmap_mode='r')
                      for name in ('hidden', 'entropy', 'target_logps')]
            if len(index) != n//8:
                raise ValueError('Incomplete native partition')
            for j, row in enumerate(index):
                i = row['global_index']
                if not 0 <= i < n or i % 8 != part or records[i] is not None:
                    raise ValueError('Repeated or incorrect native source index')
                records[i] = row; hidden[i] = arrays[0][j]
                entropy[i] = arrays[1][j]; lp[i] = arrays[2][j]
        if any(r is None for r in records) or not all(np.isfinite(v).all() for v in (hidden, entropy, lp)):
            raise ValueError('Incomplete/nonfinite native cache')
        return hidden, entropy, lp, records

    def blend(first, second, weight):
        weights = np.broadcast_to(np.asarray(weight, np.float64), first.shape)
        with np.errstate(divide='ignore'):
            return np.logaddexp(first+np.log1p(-weights), second+np.log(weights))

    rng = np.random.default_rng(2026102133); checks = {}; requirements = {}
    deltas = {}
    for cohort in ('DEV_calculation', 'DEV_ordinary'):
        raw = cache(cohort, 'B0_raw'); totals = {'B0': raw[2].sum(1)}
        for role in ('T_note', 'T_raw'):
            other = cache(cohort, role)
            for first, second in zip(raw[3], other[3]):
                if any(first[k] != second[k] for k in ('global_index', 'item_id', 'document_id', 'target_sha256')):
                    raise ValueError('Unpaired original source targets')
            controls = fitted['TRAIN_controls'][role]
            totals[role] = other[2].sum(1)
            totals[role+':global'] = blend(raw[2], other[2], controls['global_mixture']['weight']).sum(1)
            totals[role+':position'] = blend(raw[2], other[2], controls['position_schedule']['weights']).sum(1)
            for seed in ORDERS:
                check(); key = f'{role}:{seed}'; w, b = states[key]
                lp = np.empty_like(raw[2]); gates = np.empty_like(lp)
                for start in range(0, len(lp), 128):
                    sl = slice(start, start+128)
                    def normalized(x):
                        x = x[sl].astype(np.float64)
                        return x/np.sqrt(np.mean(x*x, axis=-1, keepdims=True)+1e-6)
                    x0, x1 = normalized(raw[0]), normalized(other[0])
                    u = x0@w[:p['hidden_size']] + x1@w[p['hidden_size']:2*p['hidden_size']] + b
                    u += raw[1][sl].astype(np.float64)/math.log(p['vocabulary_size'])*w[-5]
                    u += other[1][sl].astype(np.float64)/math.log(p['vocabulary_size'])*w[-4]
                    if role == 'T_note':
                        meta = other[3][sl]
                        u += np.array([v['note_actions']/128 for v in meta])[:, None]*w[-3]
                        u += np.array([float(v['capped']) for v in meta])[:, None]*w[-2]
                    u += np.arange(16)[None, :]/16*w[-1]
                    g = np.maximum(0., (np.exp(-np.logaddexp(0., -u))-.01)/.99)
                    gates[sl] = g; lp[sl] = blend(raw[2][sl], other[2][sl], g)
                close(lp, np.load(a.readout/f'{cohort}-{role}-{seed}-target_logps.npy'), key+' token likelihood')
                close(gates, np.load(a.readout/f'{cohort}-{role}-{seed}-gates.npy'), key+' causal gate')
                totals[key] = lp.sum(1)
            del other
        saved = np.load(a.readout/(cohort+'-document_scores.npz'))
        if set(saved.files) != set(totals):
            raise ValueError('All original absolute arms required')
        for key, values in totals.items():
            close(values, saved[key], cohort+':'+key)
            close(values.mean(), report['absolute_mean_log_probability_per16'][cohort][key], key+' mean')
        columns = {}
        for seed in ORDERS:
            own = totals[f'T_note:{seed}']
            for name, control in [('B0', 'B0'), ('global', 'T_note:global'),
                                  ('raw-ensemble', f'T_raw:{seed}'), ('position-secondary', 'T_note:position')]:
                columns[f'{seed}:own-minus-{name}'] = own-totals[control]
        matrix = np.column_stack(list(columns.values())); draws = []
        for start in range(0, 5000, 64):
            check(); index = rng.integers(0, 4096, size=(min(64, 5000-start), 4096))
            draws.append(matrix[index].mean(1))
        draws = np.concatenate(draws); reconstructed = {}
        for j, (key, values) in enumerate(columns.items()):
            ci = np.quantile(draws[:, j], [.025, .975]); old = report['contrasts'][cohort][key]
            close([values.mean(), *ci], [old['mean'], *old['CI95']], cohort+':'+key+' paired bootstrap')
            reconstructed[key] = dict(mean=float(values.mean()), CI95=ci.tolist())
        checks[cohort] = reconstructed; deltas[cohort] = columns
        del raw
    for seed in ORDERS:
        c = checks['DEV_calculation']; base = c[f'{seed}:own-minus-B0']
        ordinary = checks['DEV_ordinary'][f'{seed}:own-minus-B0']
        requirements[str(seed)] = dict(base=base['mean'] >= .25 and base['CI95'][0] > 0,
            global_control=c[f'{seed}:own-minus-global']['CI95'][0] > 0,
            raw_ensemble=c[f'{seed}:own-minus-raw-ensemble']['CI95'][0] > 0,
            ordinary_guard=-ordinary['CI95'][0]/16 <= .03)
    passed = all(all(v.values()) for v in requirements.values())
    if requirements != report['eight_conjunctive_requirements'] or passed != report['overall_pass']:
        raise ValueError('Independent scientific decision mismatch')
    atomic_json(a.output, dict(status='PASS_INDEPENDENT_PRIMARY_READOUT_RECONSTRUCTION',
        protocol_sha256=digest, original_readout_sha256=sha256_file(a.readout/'readout.json'),
        source_sha256=sha256_file(__file__), documents_per_DEV_stratum=4096,
        terminal_heads=4, paired_bootstrap_draws=5000, requirements=requirements, overall_pass=passed,
        limits='Original native arrays, separate NumPy causal-feature/gate/mixture reconstruction and paired statistics. Does not rerun backbones, establish faithfulness, externally replicate, or independently validate secondary interventions.'))
    print('PASS_INDEPENDENT_PRIMARY_READOUT_RECONSTRUCTION', passed, flush=True)


if __name__ == '__main__':
    main()
