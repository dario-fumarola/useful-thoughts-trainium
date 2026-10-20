"""Publish the three audited terminals and the unchanged second-seed release."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, shutil

R = Path('/home/ubuntu/dario')
C = R / 'local/diagnostics/permutation-program-v28-study-v1'
P = R / 'validation/permutation_program_v28_first_wave_20261020'
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
plan = read(C / 'plan.json')
release_path = C / 'stage64-release-seed2026102029.json'
release = read(release_path)
assert release['status'] == 'RELEASED_V28_SCIENCE_STAGE64_ONLY'
models = {}
inputs = [(release_path, 'second-seed-stage64-release.json')]
reserved_seconds = 0.
for key, job in plan['jobs'].items():
    if job['seed'] != 2026102028:
        continue
    out = Path(job['output'])
    audit_path = out / 'terminal-full-audit.json'
    audit = read(audit_path)
    assert audit['status'] == 'PASS_V28_TERMINAL_ALL_STATE' and audit['step'] == 4096
    assert audit['source_files'] == plan['source_files']
    assert sha(audit_path) == release['prior_seed_terminal_audits'][key]['sha256']
    assert sha(out / 'step-0004096/model.pt') == audit['model_sha256']
    models[key] = {k: audit[k] for k in ('step', 'model_sha256', 'COMPLETE_sha256', 'development64_sha256')}
    inputs.append((audit_path, key + '-terminal-full-audit.json'))
    for phase in ('first64', 'terminal'):
        stage = C / key / phase
        assert read(stage / 'exit.json')['exit_code'] == 0
        closure = read(stage / 'closure.json')
        assert closure['all_original_session_processes_absent'] and closure['assigned_physical_chip_empty']
        reserved_seconds += (datetime.fromisoformat(closure['utc']) - datetime.fromisoformat(read(stage / 'launch.json')['utc'])).total_seconds()
        inputs.append((stage / 'closure.json', key + '-' + phase + '-closure.json'))
    for name in ('COMPLETE.json', 'READBACK.json'):
        inputs.append((out / 'step-0004096' / name, key + '-terminal-' + name))
assert len(models) == 3
P.mkdir(exist_ok=False)
files = {}
for src, name in inputs + [(Path(__file__), Path(__file__).name)]:
    dst = P / name
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst)
    with dst.open('rb') as stream:
        os.fsync(stream.fileno())
    files[str(dst.relative_to(R))] = sha(dst)
report = dict(status='PASS_ALL_THREE_FIRST_SEED_TERMINAL4096_AUDITS_SECOND_SEED_RELEASED',
    utc=datetime.now(timezone.utc).isoformat(), experiment_id='controlled-computation-v28',
    source_commit=plan['source_commit'], models=models, updates_completed=12288,
    first_wave_reserved_physical_chip_hours=reserved_seconds / 3600,
    second_seed_stage64_deadline_utc=release['deadline_utc'],
    second_seed_whole_wave_deadline_utc=release['wave_terminal_deadline_utc'],
    held_model_outcomes=0, files=files,
    limits=['Run integrity and finite observed stability, not learned generalization.',
        'Same fixed source/data/optimizer/terminal endpoint; the second data-order seed starts from fresh B0.',
        'No held model inference before all six terminal state audits. No performance stopping or checkpoint selection.',
        'First-seed terminal models and development64 models retained; older new-study optimizer payloads may rotate only after the declared audited replacement.',
        'Same-instance EBS protection, not off-instance backup. Historical original checkpoints and cache untouched.'])
rp = R / 'validation/permutation_program_v28_first_wave_20261020.json'
with rp.open('x') as stream:
    json.dump(report, stream, indent=2)
backup = R / 'local/backups/diagnostics/permutation-program-v28-first-wave-v1'
backup.mkdir(parents=True, exist_ok=False)
for src in [rp, *P.iterdir()]:
    dst = backup / src.name
    shutil.copyfile(src, dst)
    assert sha(src) == sha(dst)
    with dst.open('rb') as stream:
        os.fsync(stream.fileno())
for directory in (P, backup):
    fd = os.open(directory, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
print(json.dumps(dict(status=report['status'], files=len(files), report_sha256=sha(rp),
    first_wave_reserved_physical_chip_hours=reserved_seconds / 3600)))
