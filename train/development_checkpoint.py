"""Strict read-only identity for a saved step64 developmental model.

This accepts only the separately published, nonresumable model-only milestone.
It never substitutes for trained_identity's full terminal/master-state gate.
"""
import json
from pathlib import Path

from weekend_thinking.artifacts import canonical_hash, sha256_file

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'development_model_only_step64'
PROTOCOL_SHA256 = '584ed0f9177d81a7c10ef3891b87b1ad7777169404cdb8b44b81b9897d6d2b62'


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Ordinary immutable development input required')
    return json.loads(path.read_text())


def identity(checkpoint, checkpoint_sha256, complete_sha256, training_protocol):
    checkpoint, training_protocol = Path(checkpoint), Path(training_protocol)
    if checkpoint.is_symlink() or training_protocol.is_symlink():
        raise ValueError('Ordinary immutable development input required')
    checkpoint, training_protocol = checkpoint.resolve(), training_protocol.resolve()
    if sha256_file(training_protocol) != PROTOCOL_SHA256:
        raise ValueError('Exact original seed20261018 science protocol required')
    complete = checkpoint.parent/'COMPLETE.json'
    if (checkpoint.name != 'model.pt' or checkpoint.parent.name != 'step-0000064'
            or checkpoint.parent.parent.name != 'development' or checkpoint.is_symlink()
            or sha256_file(checkpoint) != checkpoint_sha256 or sha256_file(complete) != complete_sha256):
        raise ValueError('Exact model-only development64 model/COMPLETE hashes required')
    run_dir = checkpoint.parent.parent.parent
    p, r, c = read(training_protocol), read(run_dir/'run.json'), read(complete)
    if (p.get('schema') != 'useful-note-sft-v4' or p.get('purpose') != 'science'
            or (p.get('steps'), p.get('global_batch'), p.get('world_size'), p.get('items')) != (320,32,32,5120)
            or r.get('schema') != p['schema'] or r.get('purpose') != 'science'
            or r.get('qualification') is not False or r.get('arm') not in ('W','Cprime','T')
            or r.get('run_id') != run_dir.name or r.get('spec_hash') != canonical_hash(p)
            or r.get('protocol_sha256') != sha256_file(training_protocol)
            or c.get('schema') != 'useful-note-model-only-v1' or c.get('resumable') is not False
            or type(c.get('step')) is not int or c['step'] != 64
            or c.get('run_id') != run_dir.name or c.get('spec_hash') != r['spec_hash']
            or set(c.get('files',{})) != {'model.pt'}
            or c['files']['model.pt'] != dict(bytes=checkpoint.stat().st_size,sha256=checkpoint_sha256)):
        raise ValueError('Actual nonresumable64 milestone of original320-update science required')
    m = c['metadata']
    if (m.get('qualification') is not False or m.get('purpose') != 'science'
            or m.get('arm') != r['arm'] or m.get('seed') != p['seed']
            or m.get('spec_hash') != r['spec_hash'] or m.get('order_hash') != r['order_hash']
            or m.get('protocol_sha256') != r['protocol_sha256']
            or m.get('data_sha256') != p['data_sha256'] or m.get('dataset_sha256') != p['dataset_sha256']
            or (m.get('parameter_dtype'),m.get('gradient_dtype'),m.get('optimizer_dtype')) != ('bf16','bf16','fp32')
            or m.get('global_batch') != 32 or m.get('items') != 5120):
        raise ValueError('Developmental model source/data/order/precision lineage differs')
    metrics_file = run_dir/'metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_file.open() if line.strip()]
    if (len(metrics) < 64 or [v['step'] for v in metrics[:64]] != list(range(1,65))
            or metrics[63]['counters'] != m['counters']
            or m['counters']['source_records'] != 2048 or m['counters']['source_tokens'] != 2048*528
            or not all(v['sampled_replicas_bitwise_equal'] and v['sampled_active_model_matches_cast_master'] for v in metrics[:64])):
        raise ValueError('Complete genuine first64 ordered update counters required')
    for name, digest in p['source_files'].items():
        source = ROOT/name
        if source.is_symlink() or sha256_file(source) != digest:
            raise ValueError('Original training source closure changed')
    if (sha256_file(p['data_file']) != p['data_sha256']
            or sha256_file(p['dataset_file']) != p['dataset_sha256']):
        raise ValueError('Original training data publication changed')
    provenance = {str(path):sha256_file(path) for path in
                  (complete,run_dir/'run.json',metrics_file,training_protocol,Path(__file__).resolve())}
    return dict(id=r['arm'], checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256,
        step=64, run_id=run_dir.name, spec_hash=r['spec_hash'], seed=p['seed'], order_hash=r['order_hash'],
        checkpoint_scope=SCOPE, developmental_only=True, resumable=False, training_plan_terminal_step=320,
        provenance=provenance, checkpoint_files={str(checkpoint):checkpoint_sha256},
        training_source_files=p['source_files'], training_protocol=str(training_protocol),
        complete_sha256=complete_sha256)


def verify(value):
    if value.get('checkpoint_scope') != SCOPE or value.get('developmental_only') is not True:
        raise ValueError('Explicit developmental model-only scope required')
    actual = identity(value['checkpoint'],value['checkpoint_sha256'],value['complete_sha256'],value['training_protocol'])
    if value != actual:
        raise ValueError('Exact developmental64 identity differs from actual saved lineage')
    return actual
