"""Canonical ARC/LAMBADA projection only after an explicitly pinned pre-gold lock.

The guard reads lock and code metadata before canonical benchmark columns.
Root verifies actual model payloads when building the lock. This projector does
no model loading, accelerator work, generation or evaluation scoring.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


PINNED = {
    'arc_challenge': dict(sha256='62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9',
        revision='210d026faf9955653af8916fad021475a3f00453', items=1172),
    'lambada': dict(sha256='99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7',
        revision='900124bf3b8235c6daf21033af9948b3f07346c4', items=5153),
}
PAIRS = {('B0', None), ('vanilla', 1234), ('vanilla', 5678), ('thought', 1234), ('thought', 5678)}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'Ordinary file required: '+str(path))
    return path


def sha(path):
    digest = hashlib.sha256()
    with ordinary(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def deadline(lock):
    value = lock['hard_deadline_utc']
    require(value == lock['training_spec']['deadline_utc'], 'Deadline differs from frozen training specification')
    end = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(end.tzinfo is not None and datetime.now(timezone.utc) < end, 'Pre-gold lock deadline missing or expired')


def validate_lock(path, expected_sha):
    """Only lock/code metadata is read here; never canonical rows or model files."""
    raw = ordinary(path).read_bytes()
    require(digest(expected_sha) and hashlib.sha256(raw).hexdigest() == expected_sha, 'Explicit pre-gold lock SHA mismatch')
    lock = json.loads(raw)
    require(lock.get('schema') == 'final-evaluation-pre-gold-lock-v1' and lock.get('status') == 'locked', 'Locked pre-gold schema required')
    spec = lock['training_spec']
    require(spec.get('status') == 'FROZEN_ASSETS_NOT_HARDWARE_QUALIFIED'
        and spec.get('spec_hash') == canonical({k: v for k, v in spec.items() if k != 'spec_hash'}), 'Frozen training specification mismatch')
    require(spec.get('terminal_steps') == 256 and spec['site']['world_size_per_arm'] == 32
        and spec['proposal']['optimization']['seeds'] == [1234, 5678]
        and spec['proposal']['optimization']['compute_dtype'] == 'fp32', 'Expected frozen terminal256/two-seed FP32 study')
    deadline(lock)
    models = lock['models']
    require(isinstance(models, list) and len(models) == 5, 'All five actual model identities required')
    pairs, base_hashes, run_ids = set(), set(), set()
    for model in models:
        arm, seed = model.get('arm'), model.get('seed')
        require(arm == 'B0' and seed is None or arm in {'vanilla', 'thought'} and type(seed) is int and seed in {1234, 5678}, 'Invalid model arm/seed')
        require((arm, seed) not in pairs, 'Duplicate model identity')
        pairs.add((arm, seed))
        require(digest(model.get('base_assets_sha256')), 'Missing base-model identity')
        base_hashes.add(model['base_assets_sha256'])
        for key, value in [('model_manifest_sha256', spec['assets']['model_manifest_sha256']),
                           ('model_id', spec['assets']['model_id']), ('model_revision', spec['assets']['model_revision'])]:
            require(model.get(key) == value, 'Model pretrained identity mismatch: '+key)
        if arm == 'B0':
            require(all(model.get(key) is None for key in ('step', 'spec_hash', 'run_id', 'model_sha256', 'complete_sha256')), 'B0 must remain unchanged released assets')
        else:
            require(type(model.get('step')) is int and model['step'] == 256 and model.get('spec_hash') == spec['spec_hash'], 'Nonterminal or wrong-spec model identity')
            require(all(digest(model.get(key)) for key in ('order_hash', 'model_sha256', 'complete_sha256')), 'Incomplete terminal model identity')
            require(isinstance(model.get('run_id'), str) and bool(model['run_id']) and model['run_id'] not in run_ids, 'Missing/duplicate terminal run identity')
            run_ids.add(model['run_id'])
    require(pairs == PAIRS and len(base_hashes) == 1, 'Exact B0 plus four terminal arm/seed identities required')
    if 'training_run_order' in spec['site']:
        require(run_ids == set(spec['site']['training_run_order']), 'Final models differ from frozen run roster')
    if 'model_inventory_sha256' in lock:
        require(lock['model_inventory_sha256'] == canonical(models), 'Model inventory hash mismatch')
    if 'models_by_label' in lock:
        labels = lock['models_by_label']
        require(isinstance(labels, dict) and len(labels) == 5
            and all(isinstance(label, str) and label for label in labels)
            and sorted(canonical(value) for value in labels.values()) == sorted(canonical(value) for value in models), 'Model label map differs from inventory')
    if 'base_assets' in lock:
        require(lock['base_assets']['sha256'] in base_hashes
            and lock['base_assets']['manifest_sha256'] == spec['assets']['model_manifest_sha256'], 'Base assets differ from model inventory')
    require(lock['projector_sha256'] == sha(Path(__file__)), 'Projector differs from pre-gold lock')
    root = Path(lock['source_root'])
    require(root.is_absolute() and not root.is_symlink() and root.is_dir(), 'Absolute sealed source directory required')
    manifest_path = Path(lock['source_manifest_file'])
    require(manifest_path.is_absolute() and sha(manifest_path) == lock['source_manifest_sha256'], 'Source manifest changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == 'science-source-archive-v1' and manifest.get('archive_dir') == str(root), 'Wrong sealed source manifest')
    files = {}
    for directory in ('weekend_thinking', 'scripts', 'train'):
        for file in sorted((root/directory).rglob('*')):
            if file.suffix in {'.py', '.sh'} and file.is_file():
                files[str(file.relative_to(root))] = sha(file)
    require(files and files == manifest['source']['files']
        and canonical(files) == manifest['source']['sha256'] == spec['source']['sha256'], 'Current sealed source closure changed')
    for task, pinned in PINNED.items():
        entry = lock['canonical_sources'][task]
        require(all(entry.get(key) == value for key, value in pinned.items())
            and type(entry['items']) is int and Path(entry['path']).is_absolute(), 'Canonical source identity mismatch: '+task)
    return lock


def pure_definitions(path, names, namespace=None):
    """Execute only named pure definitions from verified code, avoiding Torch imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require(len(selected) == len(names) and {node.name for node in selected} == set(names), 'Missing/duplicate pure definitions')
    space = dict(Path=Path, json=json, sha256_file=sha, __file__=str(path), **(namespace or {}))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), space)
    return {name: space[name] for name in names}


def project_rows(task, rows, helpers):
    questions, gold, orders, seen = [], [], [], set()
    for index, row in enumerate(rows):
        if task == 'arc_challenge':
            identifier = row.get('id')
            require(isinstance(identifier, str) and identifier.strip() and isinstance(row.get('question'), str) and row['question'].strip(), 'Malformed ARC id/question')
            choices = row.get('choices')
            require(isinstance(choices, dict), 'Malformed ARC choices')
            labels, texts = choices.get('label'), choices.get('text')
            require(isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts) >= 2
                and all(isinstance(label, str) and label for label in labels)
                and len(set(labels)) == len(labels) and row.get('answerKey') in labels
                and all(isinstance(text, str) and text.strip() for text in texts), 'Malformed ARC choice/answer mapping')
            key = 'arc_challenge:test:'+identifier
            question = dict(item_id=key, document_id=key, task=task, split='test', context=helpers['arc_context'](row['question']), choices=[' '+text for text in texts])
            answer = dict(item_id=key, gold=labels.index(row['answerKey']))
            orders.append(dict(item_id=key, labels=labels))
        else:
            require(task == 'lambada' and isinstance(row.get('text'), str), 'Malformed LAMBADA text')
            context, target = helpers['lambada_parts'](row['text'])
            key = f'lambada:test:{index:05d}'
            question = dict(item_id=key, document_id=key, task=task, split='test', context=context)
            answer = dict(item_id=key, target=target)
        require(key not in seen, 'Duplicate canonical item ID')
        seen.add(key)
        questions.append(question)
        gold.append(answer)
    require(bool(questions), 'Empty canonical source')
    return questions, gold, orders


def read_table(path, columns):
    # Import and canonical array access are strictly after validate_lock().
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns)


def execute(lock_path, lock_sha256, task, output):
    lock_path, output = Path(lock_path), Path(output)
    lock = validate_lock(lock_path, lock_sha256)
    require(task in PINNED, 'Unknown final task')
    require(not output.exists() and not output.is_symlink(), 'New output directory required; never overwrite')
    entry = lock['canonical_sources'][task]
    source = Path(entry['path'])
    require(sha(source) == entry['sha256'], 'Canonical source bytes changed')
    root = Path(lock['source_root'])
    helpers = pure_definitions(root/'train/evaluate_likelihood.py', ('arc_context', 'lambada_parts'))
    serialize = pure_definitions(root/'weekend_thinking/numeric_projection.py', ('jsonl_bytes',))['jsonl_bytes']
    dependencies = pure_definitions(root/'train/evaluation_integrity.py', ('evaluation_dependencies',))['evaluation_dependencies'](root)
    deadline(lock)
    columns = ['id', 'question', 'choices', 'answerKey'] if task == 'arc_challenge' else ['text']
    rows = read_table(source, columns=columns).to_pylist()
    require(len(rows) == entry['items'], 'Canonical row count differs; dropping rows is forbidden')
    questions, gold, orders = project_rows(task, rows, helpers)
    deadline(lock)
    output.mkdir(parents=True, exist_ok=False)
    q, g = output/'questions.jsonl', output/'gold.jsonl'
    for path, values in ((q, questions), (g, gold)):
        with path.open('xb') as stream:
            stream.write(serialize(values))
    result = dict(schema='post-lock-secondary-projection-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        task=task, split='test', items=len(questions), source_file=str(source), source_sha256=entry['sha256'], source_revision=entry['revision'],
        pre_gold_lock_file=str(lock_path.resolve()), pre_gold_lock_sha256=lock_sha256, projector_sha256=lock['projector_sha256'],
        training_spec_hash=lock['training_spec']['spec_hash'], model_inventory_sha256=canonical(lock['models']),
        source_manifest_file=lock['source_manifest_file'], source_manifest_sha256=lock['source_manifest_sha256'],
        sealed_source_sha256=lock['training_spec']['source']['sha256'], dependencies=dependencies,
        question_file=str(q.resolve()), gold_file=str(g.resolve()), questions_sha256=sha(q), gold_sha256=sha(g),
        item_ids=[row['item_id'] for row in questions], item_ids_sha256=canonical([row['item_id'] for row in questions]),
        candidate_order=orders if task == 'arc_challenge' else None, rows_dropped=0,
        note='Deterministic projection after the pre-gold lock; no generation, scores, model loading or qualification claim.')
    with (output/'projection.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pre-gold-lock', type=Path, required=True)
    parser.add_argument('--lock-sha256', required=True)
    parser.add_argument('--task', choices=tuple(PINNED), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.pre_gold_lock, args.lock_sha256, args.task, args.output)
    print(json.dumps(dict(task=result['task'], items=result['items'], projection=str(args.output/'projection.json'))))


if __name__ == '__main__':
    main()
