"""Freeze actual six-model identities publicly before any held evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil

from weekend_thinking.artifacts import sha256_file

R = Path('/home/ubuntu/dario')
C = R / 'local/diagnostics/permutation-program-v28-study-v1'
E = R / 'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
P = R / 'validation/permutation_program_v28_training_complete_20261020'
read = lambda path: json.loads(Path(path).read_text())


def main():
    release = read(E / 'release.json')
    assert release['status'] == 'RELEASED_V28_FIXED_ALL_TERMINAL_EVALUATION'
    assert release['maximum_evaluation_reserved_chip_hours'] == 30
    assert not (E / 'EXECUTION').exists()
    assert not (R / 'local/nvme-cpt/evaluation/permutation-program-v28-fixed-v1').exists()
    assert not (R / 'local/nvme-cpt/evaluation/permutation-program-v28-ordinary-CPU-v1').exists()
    plan = read(C / 'plan.json')
    models = {}
    inputs = [(E / 'release.json', 'evaluation-release30.json'),
              (E / 'original-release18.json', 'original-evaluation-release18.json'),
              (C / 'second-seed-recovery-independent-audit.json', 'second-seed-actual-restores.json'),
              (Path(__file__), Path(__file__).name)]
    for key, job in sorted(plan['jobs'].items()):
        output = Path(job['output'])
        audit_path = output / 'terminal-full-audit.json'
        audit = read(audit_path)
        assert audit['status'] == 'PASS_V28_TERMINAL_ALL_STATE' and audit['step'] == 4096
        assert audit['source_files'] == plan['source_files']
        assert sha256_file(audit_path) == release['model_specs'][key]['audit_sha256']
        assert audit['model_sha256'] == release['model_specs'][key]['checkpoint_sha256']
        final = json.loads((output / 'metrics.jsonl').read_text().splitlines()[-1])
        assert final['step'] == 4096 and final['counters']['expressions'] == 131072
        phases = {}
        for phase in ('first64', 'terminal'):
            path = C / key / phase
            launch, closure = read(path / 'launch.json'), read(path / 'closure.json')
            assert read(path / 'exit.json')['exit_code'] == 0
            assert closure['all_original_session_processes_absent'] and closure['assigned_physical_chip_empty']
            duration = (datetime.fromisoformat(closure['utc']) - datetime.fromisoformat(launch['utc'])).total_seconds()
            assert duration > 0
            phases[phase] = dict(launched_utc=launch['utc'], physically_closed_utc=closure['utc'],
                reserved_chip_hours=duration / 3600, launch_sha256=sha256_file(path / 'launch.json'))
        models[key] = dict(step=4096, model_sha256=audit['model_sha256'],
            terminal_audit_sha256=sha256_file(audit_path), actual_final_counters=final['counters'], phases=phases)
        inputs.extend([(audit_path, key + '-terminal-audit.json'),
                       (C / key / 'terminal/closure.json', key + '-physical-closure.json')])
    assert len(models) == 6
    assert read(C / 'second-seed-recovery-independent-audit.json')['status'] == 'PASS_ALL_THREE_ACTUAL64_RESTORES'
    P.mkdir(exist_ok=False)
    files = {}
    for source, name in inputs:
        target = P / name
        shutil.copyfile(source, target)
        assert sha256_file(source) == sha256_file(target)
        files[str(target.relative_to(R))] = sha256_file(target)
    extra = read(C / 'second-seed-stage64-path-repair.json')['failed_reserved_physical_chip_hours']
    reserved = sum(phase['reserved_chip_hours'] for model in models.values() for phase in model['phases'].values())
    report = dict(status='ALL_SIX_FIXED_TERMINALS_AUDITED_EVALUATION_RELEASED_NO_HELD_RESULTS',
        utc=datetime.now(timezone.utc).isoformat(), experiment_id='controlled-computation-v28',
        models=models, actual_updates=24576, held_model_outcomes=0,
        actual_training_reserved_chip_hours=reserved, preserved_preload_path_failure_chip_hours=extra,
        total_training_including_failed_preload_chip_hours=reserved + extra,
        numerical_training_source_commit=plan['source_commit'],
        prospective_design_sha256=plan['prospective_design_sha256'],
        evaluation_time_amendment=release['operational_budget_amendment'], files=files,
        limits=['Actual saved state/order/native/recovery evidence; no free-answer or generalization result yet.',
                'Charge each original phase from launch through physical closure, including initialization and checkpoint time; CPU-only audits are separate.',
                'Earlier training and decoder hardware qualifications retain their separate cost reports; they are outside this science-training total.',
                'All six terminal models and development64 models remain. Latest complete optimizer per arm retained by the prospective rolling policy.',
                'Same-instance EBS protection, not off-instance backup. No checkpoint, prompt, cohort or gate selection.'])
    path = R / 'validation/permutation_program_v28_training_complete_20261020.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2)
    backup = R / 'local/backups/diagnostics/permutation-program-v28-training-complete-v1'
    backup.mkdir(parents=True, exist_ok=False)
    for source in [path, *P.iterdir()]:
        target = backup / source.name
        shutil.copyfile(source, target)
        assert sha256_file(source) == sha256_file(target)
        for item in (source, target):
            with item.open('rb') as stream:
                os.fsync(stream.fileno())
    for directory in (P, backup):
        fd = os.open(directory, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    print(json.dumps(dict(status=report['status'], report_sha256=sha256_file(path),
                          training_reserved_chip_hours=reserved + extra, files=len(files))))


if __name__ == '__main__':
    main()
