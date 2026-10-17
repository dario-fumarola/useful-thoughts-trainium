"""Prepare final metadata, then lock actual models before projecting official gold.

This controller never runs model inference. Numerical evaluators stay unchanged.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys

if not __debug__:
    raise RuntimeError('Final preparation requires enabled assertions; use Python without -O or PYTHONOPTIMIZE.')

ROOT = Path('/home/ubuntu/dario')
HERE = ROOT/'local/evaluation/final-protocol-v1'
SOURCE = ROOT/'local/science/source-256-001'
PYTHON = '/home/ubuntu/ema-trainium/.venv/bin/python'
FROZEN = ROOT/'local/science/experiment-256.json'
MANIFEST = ROOT/'local/science/source-256-001.manifest.json'
FROZEN_SHA = '2215772156ff3fbc9fdd41b450ac3263fde2c10c5aa32c35c23981902fc0dcda'
SOURCE_SHA = '244e2f13f91dfc746f0829b236a35b8f2c5452f5d62da7a1bc3b3bbe31287ff7'
MANIFEST_SHA = '8a0641c679b0e1bf61ed984d74a14a0385e19976470d1ab96acbe7a75b77faa4'

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def canonical(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def load(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError('Ordinary input required: '+str(p))
    return json.loads(p.read_text())

def save(path, obj):
    data = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    Path(path).chmod(0o444)

def source_check():
    f, m = load(FROZEN), load(MANIFEST)
    assert sha(FROZEN) == FROZEN_SHA
    assert sha(MANIFEST) == MANIFEST_SHA
    assert canonical({k:v for k,v in f.items() if k != 'spec_hash'}) == f['spec_hash']
    assert f['source']['sha256'] == m['source']['sha256'] == SOURCE_SHA
    assert f['terminal_steps'] == 256
    assert datetime.now(timezone.utc) < datetime.fromisoformat(f['deadline_utc'])
    files = {str(path.relative_to(SOURCE)):sha(path)
        for directory in ['train','weekend_thinking','scripts']
        for path in (SOURCE/directory).rglob('*') if path.is_file() and path.suffix in ['.py','.sh']}
    assert files == m['source']['files'] and canonical(files) == SOURCE_SHA
    return f

def literal(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing sealed literal: '+name)

def expected_models():
    entries = dict(B0=dict(arm='B0', seed=None, checkpoint=None))
    for letter, arm in [('C','vanilla'),('R','thought')]:
        for seed in [1234,5678]:
            rid = f'qwen-{letter.lower()}-seed{seed}-science-256-001'
            entries[f'{letter}{seed}'] = dict(arm=arm, seed=seed, run_id=rid,
                checkpoint=str(ROOT/'local/nvme/runs'/rid/'step-0000256/model.pt'))
    return entries

def declare(output):
    f = source_check()
    prepared = ROOT/'local/evaluation/local-dev512-v1/b0-plan-v1/prepared.json'
    assert sha(prepared) == '0b54fd2d9568a7eb50877dd3ac2d698b0b6712c23d027a49fcec599e5b7bf798'
    dependencies = load(prepared)['evaluation_dependencies']
    assert all(sha(SOURCE/name) == digest for name,digest in dependencies.items())
    # Reuse actual, unchanged development protocol dictionaries. Scores are not
    # consulted, and DEV bundles will never be relabeled as TEST outputs.
    gsm = {}
    for mode in ['direct','empty_format','one_thought']:
        run = load(ROOT/'local/nvme/evaluation/held-dev128-v1-B0-001'/mode/'run.json')
        assert run['status'] == 'complete'
        gsm[mode] = run['protocol']
        assert canonical(gsm[mode]) == run['protocol_sha256']
        assert gsm[mode]['compute'] == gsm[mode]['head_precision'] == 'fp32'
        assert gsm[mode]['max_answer_tokens'] == 256
        assert gsm[mode]['dependencies'] == dependencies
    local = ROOT/'local/nvme/evaluation/local-dev512-v1-B0-001'
    generate, score = [load(local/part/'run.json') for part in ['notes','scores']]
    assert generate['status'] == score['status'] == 'complete'
    assert load(local/'COMPLETE.json')['spec_hash'] == f['spec_hash']
    for run in [generate,score]:
        assert canonical(run['protocol']) == run['protocol_sha256']
        assert run['protocol']['dependencies'] == dependencies
    assert score['protocol']['modes'] == ['raw','empty','own','filler','shuffled','other']
    assert score['protocol']['horizons'] == [1,4,16]
    selection = ROOT/'local/data/finemath/fixed-local-evaluation-v1.ids.json'
    selected = load(selection)['splits']['test']
    assert selected['count'] == len(selected['item_ids']) == 2048
    sources = load(ROOT/'local/data/benchmark_exclusion/questions.manifest.json')
    canonical_sources = {}
    keys = {'openai/gsm8k':('gsm8k','main/test-00000-of-00001.parquet'),
        'allenai/ai2_arc':('arc_challenge','ARC-Challenge/test-00000-of-00001.parquet'),
        'EleutherAI/lambada_openai':('lambada','en/test/en.parquet')}
    for dataset in sources['sources']:
        task, relative = keys[dataset['dataset_id']]
        raw = next(entry for entry in dataset['files'] if entry['path'] == relative)
        canonical_sources[task] = dict(path=str(ROOT/'local/data/benchmark_exclusion/sources'/dataset['dataset_id'].replace('/','--')/relative),
            sha256=raw['sha256'], revision=dataset['revision'], items=raw['rows'])
    inputs = [FROZEN, MANIFEST, selection, ROOT/'local/resolved_format.json',
        ROOT/'local/science/resolved_format.absolute.json', HERE/'postlock_project.py',
        ROOT/'local/evaluation/held-dev128-v1/prompt.json',
        ROOT/'local/data/benchmark_exclusion/questions.manifest.json']
    plan = dict(schema='final-evaluation-plan-v1',status='PREPARED_NOT_LOCKED',declared_utc=now(),
        planner_sha256=sha(__file__), frozen_file=str(FROZEN), frozen_file_sha256=FROZEN_SHA,
        source_root=str(SOURCE), source_sha256=SOURCE_SHA,
        source_manifest_file=str(MANIFEST), source_manifest_sha256=sha(MANIFEST),
        projector_sha256=sha(HERE/'postlock_project.py'), hard_deadline_utc=f['deadline_utc'],
        small_input_sha256={str(p):sha(p) for p in inputs}, canonical_sources=canonical_sources,
        expected_models=expected_models(),gsm_protocol_templates=gsm,
        local_generate_template=generate['protocol'],local_score_template=score['protocol'],
        local_selection=str(selection),local_test_ids_sha256=canonical(selected['item_ids']),
        secondary=dict(max_context_tokens=768,future_width=128,thought_cap=1024,compute='fp32',head_precision='fp32',
            conventions=literal(SOURCE/'train/evaluate_likelihood.py','CONVENTIONS'),
            schema=literal(SOURCE/'train/evaluate_likelihood.py','SCHEMA')),
        primary='Terminal256 R.one_thought minus C.one_thought on full official GSM8K test; seeds separately.',
        secondary_contrasts=['R.direct-C.direct','R.one_thought-R.empty_format','each trained model minus B0 in same mode'],
        failure_policy='Every original item and error/truncation remains; no score-based retries, checkpoint selection, filtering or best-of inference.',
        sharding='Root allocates finite disjoint shards; every global item exactly once; local donors use all saved global notes before any score shard.',
        final_outcomes_accessed=False)
    save(output,plan)

def read_plan(path, expected):
    assert sha(path) == expected
    p = load(path)
    assert p['schema'] == 'final-evaluation-plan-v1' and p['status'] == 'PREPARED_NOT_LOCKED'
    assert p['planner_sha256'] == sha(__file__)
    assert p['expected_models'] == expected_models()
    for name,digest in p['small_input_sha256'].items():
        assert sha(name) == digest, name
    return p

def proc(pid):
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return int(fields[19]), int(fields[2])
    except (FileNotFoundError,ProcessLookupError):
        return None

def lock_models(plan_path, plan_sha, output):
    p, f = read_plan(plan_path,plan_sha), source_check()
    backup = load(ROOT/'local/science/shared-backups.state.json')
    all_runs = {v['run_id'] for v in expected_models().values() if v['checkpoint']}
    assert all_runs.issubset(backup['terminal_verified_runs'])
    for model in p['expected_models'].values():
        if not model['checkpoint']:
            continue
        c = ROOT/'local/science/control'/model['run_id']
        assert load(c/'exit.json')['driver_exit_code'] == 0
        original = load(c/'processes.json')
        assert original['run_id'] == model['run_id'] and original['spec_hash'] == f['spec_hash']
        assert original['source_sha256'] == SOURCE_SHA and len(original['workers']) == 32
        assert len({worker['pid'] for worker in original['workers']}) == 32
        identities = [original[k] for k in ['launcher','timeout','driver']] + original['workers']
        for identity in identities:
            current = proc(identity['pid'])
            assert current is None or current[0] != identity['starttime_ticks']
        groups = {original[k]['pgid'] for k in ['launcher','timeout']}
        assert not any((current := proc(int(d.name))) and current[1] in groups
                       for d in Path('/proc').iterdir() if d.name.isdigit())
        complete = Path(model['checkpoint']).parent/'COMPLETE.json'
        dest = ROOT/'local/backups/science'/model['run_id']/'step-0000256'
        assert complete.read_bytes() == (dest/'COMPLETE.json').read_bytes()
        assert load(dest/'BACKUP.json')['manifest_sha256'] == sha(complete)
    # These existing checks hash actual released weights and all33 own checkpoint
    # payloads. They run only now, after all real terminal/exit/backup closures.
    os.environ['USE_TORCH_XLA']='0'
    os.environ['PJRT_DEVICE']='CPU'
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(SOURCE))
    from train.evaluate import asset_provenance
    from train.evaluation_integrity import final_model_identity, evaluation_dependencies
    assets = asset_provenance(Path(f['assets']['model_dir']))
    models = {}
    for label,expected in p['expected_models'].items():
        identity = final_model_identity(Path(expected['checkpoint']) if expected['checkpoint'] else None,
            assets,f,arm=expected['arm'],seed=expected['seed'])
        if expected['checkpoint']:
            assert identity['run_id'] == expected['run_id'], label
        models[label] = identity
    result = dict(schema='final-evaluation-pre-gold-lock-v1',status='locked',locked_utc=now(),
        recipe_file=str(Path(plan_path).resolve()),recipe_sha256=plan_sha,planner_sha256=sha(__file__),
        training_spec=f,models=list(models.values()),models_by_label=models,
        model_inventory_sha256=canonical(list(models.values())),base_assets=assets,
        source_root=str(SOURCE),source_manifest_file=str(MANIFEST),source_manifest_sha256=sha(MANIFEST),
        source_sha256=SOURCE_SHA,projector_sha256=p['projector_sha256'],
        canonical_sources=p['canonical_sources'],hard_deadline_utc=f['deadline_utc'],
        dependencies=evaluation_dependencies(),final_outcomes_accessed=False)
    assert datetime.now(timezone.utc) < datetime.fromisoformat(f['deadline_utc'])
    save(output,result)

def read_lock(path, expected):
    assert sha(path) == expected
    x = load(path)
    assert x['schema'] == 'final-evaluation-pre-gold-lock-v1' and x['status'] == 'locked'
    assert canonical(x['models']) == x['model_inventory_sha256']
    assert set(x['models_by_label']) == set(expected_models())
    assert list(x['models_by_label'].values()) == x['models']
    p = read_plan(x['recipe_file'],x['recipe_sha256'])
    assert x['training_spec'] == source_check()
    return x,p

def project_gsm(lock_path, lock_sha, output):
    x,p = read_lock(lock_path,lock_sha)
    seconds = int((datetime.fromisoformat(x['hard_deadline_utc'])-datetime.now(timezone.utc)).total_seconds())
    assert seconds > 0
    env = dict(os.environ,USE_TORCH_XLA='0',PJRT_DEVICE='CPU',PYTHONPATH=str(SOURCE),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s',PYTHON,
        str(SOURCE/'scripts/prepare_numeric_scoring.py'),'--canonical-source',p['canonical_sources']['gsm8k']['path'],
        '--frozen-experiment',str(FROZEN),'--output',str(output)],cwd=SOURCE,env=env,check=True,timeout=seconds+35)
    save(Path(output)/'pre-gold-lock-binding.json',dict(pre_gold_lock_sha256=lock_sha,
        projected_utc=now(),scoring_manifest_sha256=sha(Path(output)/'scoring-manifest.json')))

def assemble(lock_path, lock_sha, projections, output):
    x,p = read_lock(lock_path,lock_sha)
    projections = Path(projections)
    numeric = load(projections/'gsm8k/scoring-manifest.json')
    binding = load(projections/'gsm8k/pre-gold-lock-binding.json')
    assert binding['pre_gold_lock_sha256'] == lock_sha
    assert binding['scoring_manifest_sha256'] == sha(projections/'gsm8k/scoring-manifest.json')
    assert numeric['canonical_source_sha256'] == p['canonical_sources']['gsm8k']['sha256']
    assert numeric['items'] == 1319 and numeric['training_spec_hash'] == x['training_spec']['spec_hash']
    assert sha(projections/'gsm8k/questions.jsonl') == numeric['question_file_sha256']
    assert sha(projections/'gsm8k/gold.jsonl') == numeric['gold_file_sha256']
    protocols=[]
    for mode,template in p['gsm_protocol_templates'].items():
        protocol=copy.deepcopy(template)
        protocol.update(task_file_sha256=numeric['question_file_sha256'],item_ids_sha256=numeric['item_ids_sha256'])
        protocols.append(protocol)
    for task in ['arc_challenge','lambada']:
        d=projections/task
        receipt=load(d/'projection.json')
        assert receipt['pre_gold_lock_sha256'] == lock_sha and receipt['projector_sha256'] == p['projector_sha256']
        assert receipt['source_sha256'] == p['canonical_sources'][task]['sha256']
        assert receipt['items'] == p['canonical_sources'][task]['items']
        assert sha(d/'questions.jsonl') == receipt['questions_sha256'] and sha(d/'gold.jsonl') == receipt['gold_sha256']
        for mode in ['direct','one_thought']:
            s=p['secondary']
            protocol=dict(schema=s['schema'],task=task,task_revision=p['canonical_sources'][task]['revision'],
                task_file_sha256=receipt['questions_sha256'],gold_file_sha256=receipt['gold_sha256'],
                item_ids_sha256=receipt['item_ids_sha256'],mode=mode,max_context_tokens=s['max_context_tokens'],
                future_width=s['future_width'],thought_cap=1024 if mode=='one_thought' else 0,
                thought_format=p['local_generate_template']['thought_format'] if mode=='one_thought' else None,
                compute='fp32',head_precision='fp32',conventions=s['conventions'],
                model_id=x['base_assets']['model_id'],model_revision=x['base_assets']['revision'],
                tokenizer_and_config_sha256=p['local_generate_template']['tokenizer_and_config_sha256'],
                dependencies=x['dependencies'],evaluator_sha256=sha(SOURCE/'train/evaluate_likelihood.py'),
                backend_sha256=sha(SOURCE/'train/neuron_backend.py'))
            protocols.append(protocol)
    generate=copy.deepcopy(p['local_generate_template'])
    generate.update(split='test',selection_item_ids_sha256=p['local_test_ids_sha256'])
    protocols.append(generate)
    for identity in x['models']:
        protocol=copy.deepcopy(p['local_score_template'])
        protocol.update(split='test',selection_item_ids_sha256=p['local_test_ids_sha256'],
            writer_protocol=generate,writer_model_identity=identity)
        protocols.append(protocol)
    assert len(protocols)==13 and len({canonical(v) for v in protocols})==13
    result=dict(schema='evaluation-freeze-v2',status='frozen',frozen_utc=now(),training_spec=x['training_spec'],
        models=x['models'],protocols=protocols,scoring_manifests=[numeric],
        pre_gold_lock_file=str(Path(lock_path).resolve()),pre_gold_lock_sha256=lock_sha,
        recipe_sha256=x['recipe_sha256'],model_inventory_sha256=x['model_inventory_sha256'],
        projection_receipts={str(file):sha(file) for file in [projections/'gsm8k/scoring-manifest.json',
            projections/'arc_challenge/projection.json',projections/'lambada/projection.json']},
        final_generation_or_scoring_started=False)
    assert datetime.fromisoformat(result['frozen_utc']) >= datetime.fromisoformat(x['locked_utc'])
    save(output,result)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['declare','lock','project-gsm8k','assemble'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--plan-sha256')
    parser.add_argument('--pre-gold-lock',type=Path)
    parser.add_argument('--lock-sha256')
    parser.add_argument('--projections',type=Path)
    a=parser.parse_args()
    if a.command=='lock' and (a.plan is None or a.plan_sha256 is None):
        parser.error('lock requires --plan and --plan-sha256')
    if a.command in ['project-gsm8k','assemble'] and (a.pre_gold_lock is None or a.lock_sha256 is None):
        parser.error('projection/assembly requires --pre-gold-lock and --lock-sha256')
    if a.command=='assemble' and a.projections is None:
        parser.error('assemble requires --projections')
    if a.command=='declare':declare(a.output)
    elif a.command=='lock':lock_models(a.plan,a.plan_sha256,a.output)
    elif a.command=='project-gsm8k':project_gsm(a.pre_gold_lock,a.lock_sha256,a.output)
    else:assemble(a.pre_gold_lock,a.lock_sha256,a.projections,a.output)

if __name__=='__main__':
    main()
