"""Bounded cross-domain retention using the qualified original-text worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from train import partial_state_driver as driver
from train import uniform_lm_evaluate as original_evaluator
from train.uniform_lm_evaluate import worker
from weekend_thinking.artifacts import sha256_file

BASELINE_MODELS = {
    f'warmup_{schedule}-lr{rate}'
    for schedule in ('constant', 'cosine') for rate in ('3e-06', '1e-05')
}
SCAFFOLD_MODELS = {
    f'scaffold_{arm}-seed{seed}'
    for arm in ('raw', 'format', 'trace') for seed in (11, 12)
}


def validate(p, release, output):
    if (p.get('schema') != 'frozen-text-retention-v1'
            or p.get('cohort') != 'ENGLISH_RETENTION'
            or p.get('world_size') != 8 or p.get('micro_batch') != 4
            or p.get('items') != 1024
            or p.get('source_files') != driver.source_files()
            or p.get('evaluator_sha256') != sha256_file(__file__)
            or p.get('worker_evaluator_sha256') != sha256_file(original_evaluator.__file__)
            or not 0 < release.get('maximum_reserved_chip_hours', 0) <= 1.5):
        raise ValueError('The fixed cohort, original worker and finite budget are required')
    for name in ('data', 'data_manifest', 'independent_source_check'):
        if sha256_file(p[name+'_file']) != p[name+'_sha256']:
            raise ValueError('Frozen source or independent check differs')
    manifest = json.loads(Path(p['data_manifest_file']).read_text())
    check = json.loads(Path(p['independent_source_check_file']).read_text())
    if (manifest.get('status') != 'COMPLETE_GENERAL_WEB_SOURCE_PREPARATION'
            or manifest.get('data_sha256') != p['data_sha256']
            or manifest.get('documents') != 1024
            or check.get('status') != 'PASS_INDEPENDENT_GENERAL_WEB_SOURCES'
            or check.get('data_sha256') != p['data_sha256']):
        raise ValueError('Complete independently checked original sources are required')
    job = release.get('jobs', {}).get(str(output))
    if not job or not job.get('protocol_sha256'):
        raise ValueError('An explicit named evaluation job is required')
    terminals = release['all_terminal_models']
    required = BASELINE_MODELS | (SCAFFOLD_MODELS if p['model_name'] in SCAFFOLD_MODELS else set())
    if not required <= terminals.keys() or any(terminals[k]['step'] != 128 for k in required):
        raise ValueError('All prespecified terminal models must freeze before evaluation')
    name = p['model_name']
    if name == 'B0':
        if p['checkpoint'] is not None:
            raise ValueError('B0 must use unchanged released weights')
    elif (name not in required or name not in terminals
            or p['checkpoint_sha256'] != terminals[name]['checkpoint_sha256']
            or sha256_file(p['checkpoint']) != p['checkpoint_sha256']):
        raise ValueError('The exact frozen terminal checkpoint is required')


def read_sources(path):
    rows = [json.loads(line) for line in Path(path).open()]
    if (len(rows) != 1024
            or len({row['document_id'] for row in rows}) != 1024
            or len({row['canonical_page'] for row in rows}) != 1024
            or any(len(row['ids']) != 528 for row in rows)):
        raise ValueError('Complete unique original windows are required')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'execution-release', 'output', 'stop-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    release = json.loads(args.execution_release.read_text())
    actual = sha256_file(args.protocol)
    if release['jobs'].get(str(args.output), {}).get('protocol_sha256') != actual:
        raise ValueError('The frozen protocol differs')
    validate(p, release, args.output)
    if args.output.exists():
        raise ValueError('A fresh output is required')
    rows = read_sources(p['data_file'])
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(worker, args=(args, p, rows), nprocs=None)


if __name__ == '__main__':
    main()
