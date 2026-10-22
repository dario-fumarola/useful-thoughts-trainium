"""Post-document computation supervision with an unchanged original-text loss.

The complete original 528 tokens precede any supplied work header. Thus no
original-token prediction is conditioned on work, and every source target keeps
its coefficient. Verified work is derived only from the first 512 tokens.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch

from train import uniform_lm
from train.partial_state_packing import batch_packs
from train.useful_note_sft_data import packed
from weekend_thinking.artifacts import sha256_file

SCHEMA = 'post-document-scaffold-learning-v1'
ARMS = ('scaffold_raw', 'scaffold_format', 'scaffold_trace', 'scaffold_qualification')
SOURCE_LENGTH = 528
SOURCE_TARGETS = 527


def _schedule(p):
    return {**p, 'schema': uniform_lm.SCHEMA, 'loss': 'uniform_next_token',
            'target_slots': p['width']-1}


def validate_recipe(p):
    uniform_lm.validate_recipe(_schedule(p))
    if (p.get('schema') != SCHEMA or p.get('loss') != 'original_source_plus_post_document_work'
            or p.get('purpose') not in ('scaffold_learning_qualification', 'scaffold_learning_study')
            or p.get('source_loss_weight') != 1. or p.get('work_loss_weight') != 1.
            or p.get('scaffold_arm') not in ARMS
            or p['target_slots'] < SOURCE_TARGETS+128
            or p['width'] < SOURCE_LENGTH+144):
        raise ValueError('An explicit post-document scaffold recipe is required')
    if ((p['purpose'] == 'scaffold_learning_qualification') !=
            (p['scaffold_arm'] == 'scaffold_qualification')):
        raise ValueError('Qualification and fixed scientific arms must remain distinct')


def set_learning_rate(optimizer, p, step):
    rate = uniform_lm.learning_rate(_schedule(p), step)
    for group in optimizer.param_groups:
        group['lr'] = rate
    return rate


def training_pack(row, tokenizer, arm, *, width=704, target_slots=656):
    if arm not in ARMS[:-1] or row.get('split') != 'train':
        raise ValueError('One declared training arm and TRAIN source required')
    ids = list(row['ids'])
    vocab = len(tokenizer)
    if (len(ids) != SOURCE_LENGTH or any(type(t) is not int or not 0 <= t < vocab for t in ids)
            or row.get('kind') not in ('natural', 'calculation')):
        raise ValueError('An intact original 528-token source is required')
    source_positions = list(range(1, SOURCE_LENGTH))
    work_positions = []
    if row['kind'] == 'calculation' and arm != 'scaffold_raw':
        opening = list(row['work_header_ids'])
        trace = list(row['verified_work_ids']) if arm == 'scaffold_trace' else []
        if (not opening or len(trace) > 127 or row['stop_id'] != tokenizer.eos_token_id
                or any(type(t) is not int or not 0 <= t < vocab for t in opening+trace)):
            raise ValueError('Bounded native header, verified work and explicit closure required')
        # The header is supplied, never a target after the real source.
        first = len(ids)+len(opening)
        ids += opening+trace+[row['stop_id']]
        work_positions = list(range(first, len(ids)))
    if len(ids) > width or len(source_positions)+len(work_positions) > target_slots:
        raise ValueError('Retain the source; never truncate to make a pack fit')
    positions = source_positions+work_positions
    result = packed(ids, positions, note_positions=(), width=width, target_slots=target_slots)
    result.pop('note_close_mask'); result.pop('future_mask')
    result['source_mask'] = torch.tensor([[i < len(source_positions) for i in range(target_slots)]])
    result['work_mask'] = torch.tensor([[len(source_positions) <= i < len(positions)
                                        for i in range(target_slots)]])
    return result


def batch_at(rows, tokenizer, p, order, arm, step, rank, micro):
    validate_recipe(p)
    if arm != p['scaffold_arm'] or not (0 <= step < p['steps'] and
            0 <= rank < p['world_size'] and 0 <= micro < p['accumulation']):
        raise ValueError('The actual released arm, update and rank are required')
    actual = ('scaffold_raw', 'scaffold_format', 'scaffold_trace', 'scaffold_trace')[step % 4]
    actual = actual if arm == 'scaffold_qualification' else arm
    first = step*p['global_batch']+micro*p['world_size']*p['micro_batch']+rank*p['micro_batch']
    return batch_packs([training_pack(rows[order[i]], tokenizer, actual,
        width=p['width'], target_slots=p['target_slots'])
        for i in range(first, first+p['micro_batch'])])


def validate_sources(p, rows):
    manifest = Path(p['data_manifest_file'])
    if sha256_file(manifest) != p['data_manifest_sha256']:
        raise ValueError('Frozen scaffold source manifest changed')
    m = json.loads(manifest.read_text())
    role = 'QUALIFICATION' if p['purpose'] == 'scaffold_learning_qualification' else 'TRAIN'
    if (m.get('status') != 'COMPLETE_POST_DOCUMENT_SCAFFOLD_SOURCES'
            or m['files'][role]['sha256'] != p['data_sha256']
            or len(rows) != m['files'][role]['rows']
            or len({r['document_id'] for r in rows}) != len(rows)
            or any(r['split'] != 'train' or len(r['ids']) != SOURCE_LENGTH for r in rows)):
        raise ValueError('Complete disjoint original fitting documents required')


def validate_execution(p, args, source_files):
    if args.execution_release is None or args.arm != p['scaffold_arm']:
        raise ValueError('An explicit finite scaffold execution release is required')
    release = json.loads(Path(args.execution_release).read_text())
    job = release.get('jobs', {}).get(str(args.output))
    if (release.get('status') != 'RELEASED_SCAFFOLD_LEARNING_NATIVE_STAGE'
            or release.get('source_files') != source_files or not job
            or job.get('protocol_sha256') != sha256_file(args.protocol)
            or not 0 < release.get('maximum_reserved_chip_hours', math.inf) <= 32):
        raise ValueError('Frozen source, job and finite scaffold budget required')
    if p['purpose'] != 'scaffold_learning_qualification':
        q = release['native_qualification']; path = Path(q['file'])
        if sha256_file(path) != q['sha256']:
            raise ValueError('Actual scaffold qualification receipt changed')
        receipt = json.loads(path.read_text())
        if (receipt.get('status') != 'PASS_SCAFFOLD_LEARNING_NATIVE_QUALIFICATION'
                or receipt['source_files'] != source_files
                or any(receipt.get(k) != p[k] for k in
                       ('world_size', 'micro_batch', 'accumulation', 'width', 'target_slots'))
                or not receipt.get('exact_full_state_recovery')):
            raise ValueError('Same-shape source-qualified native recovery is required')
