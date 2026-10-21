"""Plain next-token windows, optimizer-step schedules and accumulation layout.

All windows have the same native length. One mean over every shifted token is
therefore both the token-weighted and document-weighted objective. There are no
inserted delimiters, answer labels, auxiliary passes or sampled target masks.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

from train.partial_state_packing import batch_packs
from train.useful_note_sft_data import packed
from weekend_thinking.artifacts import sha256_file

SCHEMA = 'uniform-next-token-v1'
ARM = 'uniform_lm'
SCHEDULES = ('warmup_constant', 'warmup_cosine')


def validate_recipe(p):
    integers = ('items', 'steps', 'world_size', 'micro_batch', 'global_batch',
                'accumulation', 'width', 'target_slots', 'warmup_steps')
    if any(type(p.get(k)) is not int for k in integers):
        raise ValueError('Explicit integer exposure, topology and schedule required')
    if (p.get('schema') != SCHEMA or p.get('loss') != 'uniform_next_token'
            or p.get('schedule') not in SCHEDULES
            or p['items'] != p['steps'] * p['global_batch']
            or not 1 <= p['accumulation'] <= 32
            or p['global_batch'] != p['world_size'] * p['micro_batch'] * p['accumulation']
            or p['world_size'] < 1 or p['micro_batch'] < 1
            or p['width'] < 2 or p['target_slots'] != p['width'] - 1
            or not 0 < p['warmup_steps'] < p['steps']):
        raise ValueError('One complete source pass and all shifted targets required')
    for key in ('lr', 'minimum_lr_ratio'):
        value = p.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError('Finite real learning-rate parameters required')
    if not 0 < p['lr'] <= 1e-4 or not 0 <= p['minimum_lr_ratio'] <= 1:
        raise ValueError('Bounded positive learning rate and decay floor required')


def learning_rate(p, step):
    """LR for zero-based optimizer step, including resumed steps, not microsteps."""
    validate_recipe(p)
    if type(step) is not int or not 0 <= step < p['steps']:
        raise ValueError('An actual optimizer step is required')
    completed = step + 1
    if completed <= p['warmup_steps']:
        return p['lr'] * completed / p['warmup_steps']
    if p['schedule'] == 'warmup_constant':
        return p['lr']
    progress = (completed-p['warmup_steps']) / (p['steps']-p['warmup_steps'])
    floor = p['minimum_lr_ratio']
    return p['lr'] * (floor + (1-floor) * (1+math.cos(math.pi*progress))/2)


def training_pack(row, tokenizer, *, width, target_slots):
    ids = row.get('ids')
    vocabulary_size = len(tokenizer)
    if (row.get('split') != 'train' or row.get('schema') != SCHEMA
            or not isinstance(ids, list) or len(ids) != width
            or target_slots != width-1 or width < 2
            or any(type(t) is not int or not 0 <= t < vocabulary_size for t in ids)):
        raise ValueError('Complete original native TRAIN window required')
    # A real source EOS remains supervised. Nothing is appended or truncated.
    return packed(ids, range(1, width), note_positions=(),
                  width=width, target_slots=target_slots)


def batch_at(rows, tokenizer, p, order, step, rank, micro):
    validate_recipe(p)
    if (len(rows) != p['items'] or len(order) != p['items']
            or not 0 <= step < p['steps'] or not 0 <= rank < p['world_size']
            or not 0 <= micro < p['accumulation']):
        raise ValueError('Complete one-pass order and bounded accumulation indices required')
    first = (step*p['global_batch'] + micro*p['world_size']*p['micro_batch']
             + rank*p['micro_batch'])
    return batch_packs([training_pack(rows[order[i]], tokenizer,
        width=p['width'], target_slots=p['target_slots'])
        for i in range(first, first+p['micro_batch'])])


def set_learning_rate(optimizer, p, step):
    """ZeRO copies these public groups into its local optimizer before stepping."""
    rate = learning_rate(p, step)
    for group in optimizer.param_groups:
        group['lr'] = rate
    return rate


def validate_sources(p, rows):
    """Bind TRAIN/qualification data; development and confirmation are never fit."""
    path = Path(p['data_manifest_file'])
    if sha256_file(path) != p['data_manifest_sha256']:
        raise ValueError('Uniform-LM source manifest changed')
    manifest = json.loads(path.read_text())
    cohort = 'QUALIFICATION' if p['purpose'] == 'uniform_lm_qualification' else 'TRAIN'
    if (manifest.get('status') != 'COMPLETE_UNIFORM_LM_SOURCE_PREPARATION'
            or manifest['files'][cohort]['sha256'] != p['data_sha256']
            or manifest['files'][cohort]['rows'] != len(rows)
            or any(r.get('cohort') != cohort or r.get('split') != 'train'
                   or r.get('schema') != SCHEMA for r in rows)
            or len({r['document_id'] for r in rows}) != len(rows)):
        raise ValueError('Only complete, unique, declared fitting sources are allowed')


def validate_execution(p, args, source_files):
    """CPU preparation is distinct from an actually budgeted native release."""
    if args.arm != ARM or args.execution_release is None:
        raise ValueError('An explicit uniform-LM native stage release is required')
    release = json.loads(Path(args.execution_release).read_text())
    job = release.get('jobs', {}).get(str(args.output))
    if (release.get('status') != 'RELEASED_UNIFORM_LM_NATIVE_STAGE'
            or release.get('source_files') != source_files or job is None
            or job.get('protocol_sha256') != sha256_file(args.protocol)
            or not isinstance(release.get('maximum_reserved_chip_hours'), (int, float))
            or isinstance(release['maximum_reserved_chip_hours'], bool)
            or not 0 < release['maximum_reserved_chip_hours'] < math.inf):
        raise ValueError('A frozen job and finite native resource release are required')
    if p['purpose'] != 'uniform_lm_qualification':
        receipt = release.get('native_qualification', {})
        path = Path(receipt.get('file', ''))
        if not path.is_file() or sha256_file(path) != receipt.get('sha256'):
            raise ValueError('Actual uniform-objective accumulation/recovery qualification required')
        qualified = json.loads(path.read_text())
        if (qualified.get('status') != 'PASS_UNIFORM_LM_NATIVE_QUALIFICATION'
                or qualified.get('source_files') != source_files
                or qualified.get('accumulation') != p['accumulation']
                or qualified.get('world_size') != p['world_size']
                or qualified.get('width') != p['width']):
            raise ValueError('Numerical qualification must cover this accumulation and shape')
