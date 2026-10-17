"""Small host-only fixtures; all copied checkpoint payloads are synthetic bytes."""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from weekend_thinking.artifacts import atomic_json, sha256_file
from scripts.watch_checkpoint_backups import cycle, event

spec = importlib.util.spec_from_file_location('shared_backups', Path(__file__).with_name('shared_checkpoint_backups.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def roster(tmp_path, count=4):
    return dict(schema='shared-checkpoint-backup-v1', spec_hash='a'*64, source_sha256='b'*64,
        source_root=str(tmp_path/'immutable-source'), terminal_step=3, roster_sha256='c'*64,
        deadline_utc=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat(),
        lock_file=str(tmp_path/'control/leader.lock'), log_file=str(tmp_path/'control/events.jsonl'),
        state_file=str(tmp_path/'control/state.json'), runs=[dict(run_id=f'run-{i}',
            source=str(tmp_path/f'scratch/run-{i}'), destination=str(tmp_path/f'ebs/run-{i}'),
            stop_file=str(tmp_path/f'control/run-{i}.STOP')) for i in range(count)])


def checkpoint(source, step):
    path = source/f'step-{step:07d}'
    path.mkdir(parents=True)
    for name in ('model.pt', 'rank-00000.pt'):
        (path/name).write_bytes(bytes([step])*1024)
    atomic_json(path/'COMPLETE.json', dict(schema='neuron-v1', run_id=source.name, step=step,
        world_size=1, spec_hash='a'*64, files={p.name:dict(bytes=p.stat().st_size,
        sha256=sha256_file(p)) for p in path.iterdir()}))


def run(config, function, **kwargs):
    return module.supervise(config, function, atomic_json=atomic_json, event=event,
                            install_signals=False, **kwargs)


def test_two_run_real_copies_are_sequential_retain_two_and_preserve_sources(tmp_path):
    # Runtime roster parsing requires exactly four; two runs isolate this core loop fixture.
    config = roster(tmp_path, count=2)
    log = Path(config['log_file']);log.parent.mkdir()
    for entry in config['runs']:
        source = Path(entry['source']);checkpoint(source, 1)
        cycle(source, entry['destination'], spec_hash=config['spec_hash'], terminal_step=3,
              log=log, allow_same_filesystem=True)
        checkpoint(source, 2);checkpoint(source, 3)
    trace, active = [], 0

    def copying(source, destination, **kwargs):
        nonlocal active
        active += 1
        assert active == 1
        trace.append(('start', source.name))
        try:
            return cycle(source, destination, **kwargs, allow_same_filesystem=True)
        finally:
            trace.append(('finish', source.name));active -= 1

    with module.leader_lock(config['lock_file']):
        run(config, copying)
    assert trace == [('start', 'run-0'), ('finish', 'run-0'), ('start', 'run-1'), ('finish', 'run-1')]*2
    for entry in config['runs']:
        assert len(list(Path(entry['source']).glob('step-*'))) == 3
        assert {p.name for p in Path(entry['destination']).glob('step-*')} == {'step-0000002','step-0000003'}
        assert not Path(entry['stop_file']).exists()
    assert json.loads(Path(config['state_file']).read_text())['status'] == 'COMPLETE'


def test_copy_failure_stops_all_four_preserves_incomplete_and_future_outputs(tmp_path):
    config = roster(tmp_path)
    incomplete = Path(config['runs'][0]['destination']).with_name('.step-0000001.incomplete')

    def failed_copy(*args, **kwargs):
        incomplete.mkdir(parents=True)
        (incomplete/'.model.pt.part').write_bytes(b'preserve interrupted copy')
        raise OSError('fixture disk failure')

    with module.leader_lock(config['lock_file']), pytest.raises(OSError, match='fixture disk failure'):
        run(config, failed_copy)
    assert (incomplete/'.model.pt.part').read_bytes() == b'preserve interrupted copy'
    assert all(Path(entry['stop_file']).is_file() for entry in config['runs'])
    assert not any(Path(entry['source']).exists() for entry in config['runs'])
    assert json.loads(Path(config['state_file']).read_text())['status'] == 'FAILED_STOP_REQUESTED'


@pytest.mark.parametrize('during_cycle', [False, True])
def test_deadline_before_or_after_cycle_stops_without_terminal_success(tmp_path, during_cycle):
    config = roster(tmp_path)
    deadline = datetime.now(timezone.utc)
    config['deadline_utc'] = deadline.isoformat()
    clock = [deadline-timedelta(seconds=1) if during_cycle else deadline]
    calls = []

    def finish_after_deadline(*args, **kwargs):
        calls.append(True);clock[0] = deadline+timedelta(seconds=1)
        return True

    with module.leader_lock(config['lock_file']), pytest.raises(TimeoutError, match='deadline'):
        run(config, finish_after_deadline, clock=lambda:clock[0])
    assert len(calls) == int(during_cycle)
    assert all(Path(entry['stop_file']).is_file() for entry in config['runs'])
    assert not any(Path(entry['source']).exists() for entry in config['runs'])
    assert json.loads(Path(config['state_file']).read_text())['status'] != 'COMPLETE'


def test_duplicate_real_advisory_lock_does_not_stop_valid_leader(tmp_path):
    config = roster(tmp_path)
    with module.leader_lock(config['lock_file']):
        with pytest.raises(module.LeaderBusy):
            with module.leader_lock(config['lock_file']):
                pytest.fail('Duplicate leader entered supervision')
    assert not any(Path(entry['stop_file']).exists() for entry in config['runs'])
    assert not any(Path(entry['source']).exists() for entry in config['runs'])
    # Lock inode is retained and becomes available after the elected leader exits.
    with module.leader_lock(config['lock_file']):
        pass


def test_frozen_roster_requires_four_runs_and_control_paths_outside_outputs(tmp_path):
    config = roster(tmp_path)
    path = tmp_path/'roster.json'

    def publish(value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    digest = publish(config)
    assert len(module.read_roster(path, digest)['runs']) == 4
    assert not any(Path(entry['source']).exists() for entry in config['runs'])
    config['runs'].pop()
    with pytest.raises(ValueError, match='four runs'):
        module.read_roster(path, publish(config))
    config = roster(tmp_path);config['runs'][0]['stop_file'] = config['runs'][0]['source']+'/STOP'
    with pytest.raises(ValueError, match='outside'):
        module.read_roster(path, publish(config))
