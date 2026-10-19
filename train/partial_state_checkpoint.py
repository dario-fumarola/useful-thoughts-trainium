"""Read-only terminal bindings for separately declared fitting/study models.

Never treat a qualification, model-only snapshot or unfinished run as the
reader. Training still starts from B0; this module is for frozen evaluation.
"""
import json
from pathlib import Path

from train.partial_state_driver import ARMS, checkpoint_identity, order_for, validate
from weekend_thinking.artifacts import canonical_hash, sha256_file

SCOPE = 'partial-state-full-terminal-v7'


def terminal_scope(protocol, run, completion):
    validate(protocol)
    steps=protocol['steps']
    fitting=protocol['purpose']=='TRAIN_overfit_only' and protocol['items']==128 and steps==128
    science=protocol['purpose']=='TRAIN5120_partial_state_study' and protocol['items']==5120 and steps in (320,640)
    pilot=(protocol['schema']=='variable-depth-writer-v11'
           and protocol['purpose']=='variable_depth_pilot'
           and protocol['items']==640 and steps==160 and run.get('arm')=='writer')
    if (not (fitting or science or pilot)
            or run.get('purpose') != protocol['purpose'] or run.get('arm') not in ARMS
            or run.get('pipeline_schema') != protocol['schema']
            or run.get('spec_hash') != canonical_hash(protocol)
            or run.get('planned_steps') != steps
            or completion.get('purpose') != protocol['purpose']
            or completion.get('status') != 'complete'
            or completion.get('actual_steps') != steps or completion.get('planned_steps') != steps):
        raise ValueError('Declared full terminal model required; qualification is excluded')


def identity(checkpoint, protocol_file, closure_file, producer_source_root=None):
    """Hash the full publication and check every shard's original lineage."""
    # Older studies retain their exact validator in an immutable source tree.
    # Execute it there rather than weakening its source-closure check to fit
    # current code. This is an explicit identity field, not a guessed fallback.
    if producer_source_root is not None:
        import os
        import subprocess
        import sys
        root = Path(producer_source_root).resolve()
        current = Path(__file__).resolve().parents[1]
        if root != current:
            if not (root/'train/partial_state_checkpoint.py').is_file():
                raise ValueError('Original producer validator source required')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root) + os.pathsep + env.get('PYTHONPATH', '')
            command = [sys.executable, '-c',
                'import json,sys; from train.partial_state_checkpoint import identity; '
                'print(json.dumps(identity(*sys.argv[1:])))',
                str(checkpoint), str(protocol_file), str(closure_file)]
            result = subprocess.run(command, cwd=root, env=env, text=True,
                                    capture_output=True, check=True, timeout=600)
            value = json.loads(result.stdout)
            value['producer_source_root'] = str(root)
            return value
    import torch
    from train.neuron_driver import verify_checkpoint
    paths = [Path(p) for p in (checkpoint, protocol_file, closure_file)]
    if any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError('Ordinary trusted checkpoint, protocol and closure files required')
    checkpoint, protocol_file, closure_file = [p.resolve() for p in paths]
    if checkpoint.name != 'model.pt':
        raise ValueError('Use the declared full terminal publication')
    root = checkpoint.parent.parent
    read = lambda p: json.loads(p.read_text())
    protocol, run, completion = read(protocol_file), read(root/'run.json'), read(root/'completion.json')
    terminal_scope(protocol, run, completion)
    steps=protocol['steps'];fitting=protocol['purpose']=='TRAIN_overfit_only'
    if checkpoint.parent.name!=f'step-{steps:07d}':
        raise ValueError('Use the declared full terminal publication')
    closure = read(closure_file)
    closure_status='COMPLETE128_TRAIN_ONLY_PHYSICALLY_CLOSED_CHECKPOINTS_VERIFIED' if fitting else 'COMPLETE_TRAINING_PHYSICALLY_CLOSED_CHECKPOINTS_VERIFIED'
    if (closure.get('status') != closure_status
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
    if ([m['step'] for m in metrics] != list(range(1,steps+1))
            or metrics[-1]['counters'] != manifest['metadata']['counters']
            or metrics[-1]['counters']['expressions'] != steps*protocol['global_batch']):
        raise ValueError('All original complete updates and counters required')
    provenance = {str(p):sha256_file(p) for p in
        (protocol_file, closure_file, root/'run.json', root/'completion.json', metrics_file, checkpoint.parent/'COMPLETE.json')}
    label = f"{run['arm']}-TRAIN{protocol['items']}-step{steps}"
    if protocol['schema']=='variable-depth-writer-v11':
        label = f"{run['arm']}-{protocol['curriculum']}-TRAIN{protocol['items']}-step{steps}-seed{protocol['seed']}"
    value = dict(id=label, arm=run['arm'], checkpoint=str(checkpoint),
        checkpoint_sha256=manifest['files']['model.pt']['sha256'],
        checkpoint_scope=SCOPE, developmental_only=fitting, step=steps, run_id=root.name,
        spec_hash=canonical_hash(protocol), training_protocol=str(protocol_file),
        closure_file=str(closure_file), provenance=provenance,
        checkpoint_files={str(checkpoint.parent/k):v['sha256'] for k,v in manifest['files'].items()})
    if producer_source_root is not None:
        value['producer_source_root'] = str(Path(producer_source_root).resolve())
    return value


def verify(value):
    if value.get('checkpoint_scope') != SCOPE or type(value.get('developmental_only')) is not bool:
        raise ValueError('Explicit v7 terminal scope required')
    actual = identity(value['checkpoint'], value['training_protocol'], value['closure_file'],
                      value.get('producer_source_root'))
    if actual != value:
        raise ValueError('Frozen TRAIN128 reader identity changed')
    return actual
