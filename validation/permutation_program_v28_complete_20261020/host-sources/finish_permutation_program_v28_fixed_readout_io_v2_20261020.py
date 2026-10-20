"""Run the frozen offline analysis only after all original jobs physically close.

No model calls, retries, new cohorts, parser changes, or automatic publication.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/controlled-computation-v28-eval'
E = R / 'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
O = R / 'local/diagnostics/permutation-program-v28-complete-readout-io-v2'
read = lambda p: json.loads(Path(p).read_text())


def main():
    assert not O.exists(), 'Keep original offline analyses; no silent replacement'
    plan = read(E / 'EXECUTION/plan.json')
    deadline = datetime.fromisoformat(plan['worker_deadline_utc']).timestamp() + 300
    print('WAITING_FOR_ALL23_ORIGINAL_PHYSICAL_CLOSURES_NO_SCORES_READ', flush=True)
    while not (E / 'EXECUTION/completion.json').exists():
        assert time.time() < deadline, 'Finite wait expired; no model retry'
        time.sleep(30)
    completion = read(E / 'EXECUTION/completion.json')
    assert completion['status'] == 'COMPLETE_ALL_FIXED23_GROUPS_PHYSICALLY_CLOSED_NO_SCORES_REVIEWED'
    registry = read(E / 'registry.json')
    assert len(registry['jobs']) == len(read(E / 'index.json')) == 23
    for job in registry['jobs']:
        key = job['key']
        assert read(E / 'EXECUTION' / (key + '-closed.json'))['exit_code'] == 0
        close = read(E / 'EXECUTION' / key / 'closure.json')
        assert close['all_original_session_processes_absent'] and close['assigned_physical_device_empty']
        assert close['actual_attempts'] == job['attempts']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=W, text=True).strip() == '2eb3c1cf09d14604cc9b5dd0f97562656e95b6a5'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True).strip()
    O.mkdir()
    env = dict(os.environ, USE_TORCH_XLA='0', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
               PYTHONPATH=str(W) + ':' + str(R / 'local/test-deps'))
    index = str(E / 'index.json')
    result = str(O / 'readout.json')
    check = str(O / 'independent-check.json')
    stages = [
        ('primary', ['-m', 'train.permutation_program_readout', '--index', index, '--output', result]),
        ('independent', [str(R / 'local/analysis/check_permutation_program_v28_readout_20261020.py'),
                         '--index', index, '--readout', result, '--output', check]),
        ('secondary', [str(R / 'local/analysis/permutation_program_v28_secondary_reference_20261020.py'),
                       '--index', index, '--primary-readout', result, '--output', str(O / 'secondary-reference.json')]),
        ('fixed-examples', [str(R / 'local/analysis/collect_permutation_program_v28_examples_20261020.py'),
                            '--index', index, '--readout', result, '--independent-check', check,
                            '--output', str(O / 'fixed-examples.json')]),
        ('plots', [str(R / 'local/analysis/plot_permutation_program_v28_20261020.py'),
                   '--readout', result, '--output', str(O / 'figures')]),
    ]
    history = []
    for name, arguments in stages:
        print('OFFLINE_STAGE_START ' + name, flush=True)
        started = datetime.now(timezone.utc).isoformat()
        with (O / (name + '.log')).open('x') as log:
            process = subprocess.run([sys.executable, *arguments], cwd=W, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        history.append(dict(stage=name, exit_code=process.returncode, started=started,
                            closed=datetime.now(timezone.utc).isoformat()))
        atomic_json(O / 'stage-history.json', history, overwrite=True)
        assert process.returncode == 0, 'Offline stage failed; preserve its original output, no model retry'
        print('OFFLINE_STAGE_CLOSED0 ' + name, flush=True)
    assert read(O / 'independent-check.json')['status'] == 'PASS_INDEPENDENT_NATIVE_ACTIONS_NODES_WITNESSES_FAMILY_BOOTSTRAPS_AND_GATES'
    atomic_json(O / 'COMPLETE.json', dict(status='COMPLETE_OFFLINE_READOUT_INDEPENDENT_CHECK_REFERENCES_EXAMPLES_FIGURES',
                utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha256_file(__file__),
                execution_completion_sha256=sha256_file(E / 'EXECUTION/completion.json'),
                readout_sha256=sha256_file(O / 'readout.json'), model_calls=0,
                publication='Pending interpretation and visual inspection; no automatic scientific claim.'))
    print('ALL_FIXED_OFFLINE_ANALYSES_COMPLETE_NO_NEW_MODEL_CALLS', flush=True)


if __name__ == '__main__':
    main()
