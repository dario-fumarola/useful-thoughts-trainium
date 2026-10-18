"""Read-only terminal bindings for the explicitly nonfinal TRAIN128 models.

Never treat a qualification, model-only snapshot or unfinished run as the
reader. Training still starts from B0; this module is for frozen evaluation.
"""
import json
from pathlib import Path

from train.partial_state_driver import ARMS, checkpoint_identity, order_for, validate
from weekend_thinking.artifacts import canonical_hash, sha256_file

SCOPE = 'partial-state-TRAIN128-terminal128'


def terminal_scope(protocol, run, completion):
    validate(protocol)
    if (protocol['purpose'] != 'TRAIN_overfit_only' or protocol['steps'] != 128
            or run.get('purpose') != protocol['purpose'] or run.get('arm') not in ARMS
            or run.get('pipeline_schema') != protocol['schema']
            or run.get('spec_hash') != canonical_hash(protocol)
            or run.get('planned_steps') != 128
            or completion.get('purpose') != protocol['purpose']
            or completion.get('status') != 'complete'
            or completion.get('actual_steps') != 128 or completion.get('planned_steps') != 128):
        raise ValueError('Completed TRAIN128 terminal128 model required; qualification is excluded')


def identity(checkpoint, protocol_file, closure_file):
    """Hash the full publication and check every shard's original lineage."""
    import torch
    from train.neuron_driver import verify_checkpoint
    paths = [Path(p) for p in (checkpoint, protocol_file, closure_file)]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Ordinary trusted checkpoint, protocol and closure files required')
    checkpoint, protocol_file, closure_file = [p.resolve() for p in paths]
    if checkpoint.name != 'model.pt' or checkpoint.parent.name != 'step-0000128':
        raise ValueError('Use the declared full terminal128 publication')
    root = checkpoint.parent.parent
    read = lambda p: json.loads(p.read_text())
    protocol, run, completion = read(protocol_file), read(root/'run.json'), read(root/'completion.json')
    terminal_scope(protocol, run, completion)
    closure = read(closure_file)
    if (closure.get('status') != 'COMPLETE128_TRAIN_ONLY_PHYSICALLY_CLOSED_CHECKPOINTS_VERIFIED'
            or closure.get('all_original_session_processes_absent') is not True
            or closure.get('completion_sha256') != sha256_file(root/'completion.json')
            or closure.get('terminal_COMPLETE_sha256') != sha256_file(checkpoint.parent/'COMPLETE.json')):
        raise ValueError('Original physical closure and terminal hashes required')
    protocol_sha = sha256_file(protocol_file)
    if (run['protocol_sha256'] != protocol_sha or run['order_hash'] != canonical_hash(order_for(protocol))
            or sha256_file(protocol['data_file']) != protocol['data_sha256']):
        raise ValueError('Original data/order/protocol bytes changed')
    manifest = verify_checkpoint(checkpoint.parent, canonical_hash(protocol), 4)
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    for rank in range(4):
        saved = torch.load(checkpoint.parent/f'rank-{rank:05d}.pt', map_location='cpu', weights_only=True, mmap=True)
        checkpoint_identity(manifest, weights, saved, protocol, run['arm'], root.name, rank, run['order_hash'], protocol_sha)
    metrics_file = root/'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_file.read_text().splitlines() if line.strip()]
    if ([m['step'] for m in metrics] != list(range(1,129))
            or metrics[-1]['counters'] != manifest['metadata']['counters']
            or metrics[-1]['counters']['expressions'] != 4096):
        raise ValueError('All original128 complete updates and counters required')
    provenance = {str(p):sha256_file(p) for p in
        (protocol_file, closure_file, root/'run.json', root/'completion.json', metrics_file, checkpoint.parent/'COMPLETE.json')}
    return dict(id=run['arm']+'-TRAIN128-step128', arm=run['arm'], checkpoint=str(checkpoint),
        checkpoint_sha256=manifest['files']['model.pt']['sha256'],
        checkpoint_scope=SCOPE, developmental_only=True, step=128, run_id=root.name,
        spec_hash=canonical_hash(protocol), training_protocol=str(protocol_file),
        closure_file=str(closure_file), provenance=provenance,
        checkpoint_files={str(checkpoint.parent/k):v['sha256'] for k,v in manifest['files'].items()})


def verify(value):
    if value.get('checkpoint_scope') != SCOPE or value.get('developmental_only') is not True:
        raise ValueError('Explicit TRAIN128 developmental scope required')
    actual = identity(value['checkpoint'], value['training_protocol'], value['closure_file'])
    if actual != value:
        raise ValueError('Frozen TRAIN128 reader identity changed')
    return actual
