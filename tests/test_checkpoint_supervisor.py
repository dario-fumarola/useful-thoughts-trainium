"""Small real-file backup/retention faults; no accelerator claims."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from weekend_thinking.artifacts import atomic_json, sha256_file

spec = importlib.util.spec_from_file_location(
    'backup_supervisor', Path(__file__).resolve().parents[1] / 'scripts/watch_checkpoint_backups.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(tmp_path):
    source = tmp_path / 'scratch' / 'run'
    source.mkdir(parents=True)
    target = tmp_path / 'ebs' / 'run'
    arguments = dict(spec_hash='a' * 64, terminal_step=4,
                     log=tmp_path / 'events.jsonl', allow_same_filesystem=True)
    return source, target, arguments


def checkpoint(source, step):
    path = source / f'step-{step:07d}'
    path.mkdir()
    for name in ['model.pt', 'rank-00000.pt']:
        (path / name).write_bytes(bytes([step]) * 1024)
    manifest = dict(schema='neuron-v1', run_id=source.name, step=step, world_size=1,
                    spec_hash='a' * 64, files={p.name: dict(
                        bytes=p.stat().st_size, sha256=sha256_file(p)) for p in path.iterdir()})
    atomic_json(path / 'COMPLETE.json', manifest)
    return path


def test_rotation_preserves_source_and_two_latest_complete_backups(tmp_path):
    source, target, arguments = fixture(tmp_path)
    for step in range(1, 5):
        checkpoint(source, step)
        assert module.cycle(source, target, **arguments) == (step == 4)
    assert {p.name for p in target.glob('step-*')} == {'step-0000003', 'step-0000004'}
    assert len(list(source.glob('step-*'))) == 4
    assert module.cycle(source, target, **arguments)


def test_source_corruption_prevents_old_copy_removal(tmp_path):
    source, target, arguments = fixture(tmp_path)
    for step in range(1, 3):
        checkpoint(source, step)
        module.cycle(source, target, **arguments)
    (source / 'step-0000001' / 'model.pt').write_bytes(b'corrupt')
    checkpoint(source, 3)
    with pytest.raises(ValueError, match='integrity'):
        module.cycle(source, target, **arguments)
    assert (target / 'step-0000001' / 'model.pt').exists()
    assert (target / 'step-0000003' / 'COMPLETE.json').exists()


def test_extra_files_are_not_removed(tmp_path):
    source, target, arguments = fixture(tmp_path)
    for step in range(1, 3):
        checkpoint(source, step)
        module.cycle(source, target, **arguments)
    (target / 'step-0000001' / 'personal-note').write_text('preserve')
    checkpoint(source, 3)
    with pytest.raises(ValueError, match='unexpected'):
        module.cycle(source, target, **arguments)
    assert (target / 'step-0000001' / 'personal-note').read_text() == 'preserve'


def test_foreign_checkpoint_fails_before_copy(tmp_path):
    source, target, arguments = fixture(tmp_path)
    path = checkpoint(source, 1)
    manifest = json.loads((path / 'COMPLETE.json').read_text())
    manifest['spec_hash'] = 'b' * 64
    atomic_json(path / 'COMPLETE.json', manifest, overwrite=True)
    with pytest.raises(ValueError, match='different run'):
        module.cycle(source, target, **arguments)
    assert not target.exists()


def test_unmanaged_directory_is_preserved(tmp_path):
    source, target, arguments = fixture(tmp_path)
    target.mkdir(parents=True)
    (target / 'step-0000000').mkdir()
    (target / 'step-0000000' / 'keep').write_text('unmanaged')
    for step in range(1, 5):
        checkpoint(source, step)
        module.cycle(source, target, **arguments)
    assert (target / 'step-0000000' / 'keep').read_text() == 'unmanaged'


def test_restart_rechecks_latest_copy_and_fails_if_corrupted(tmp_path):
    source, target, arguments = fixture(tmp_path)
    checkpoint(source, 1)
    module.cycle(source, target, **arguments)
    (target / 'step-0000001' / 'model.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'):
        module.cycle(source, target, **arguments)
    assert (source / 'step-0000001' / 'model.pt').stat().st_size == 1024


def test_late_supervisor_backfills_two_sources_before_terminal_success(tmp_path):
    source, target, arguments = fixture(tmp_path)
    for step in range(1, 5):
        checkpoint(source, step)
    assert module.cycle(source, target, **arguments)
    assert {p.name for p in target.glob('step-*')} == {'step-0000003', 'step-0000004'}


def test_corrupt_survivor_preserves_good_redundant_backup(tmp_path):
    source, target, arguments = fixture(tmp_path)
    verified = set()
    for step in range(1, 3):
        checkpoint(source, step)
        module.cycle(source, target, verified_this_process=verified, **arguments)
    (target / 'step-0000002' / 'model.pt').write_bytes(b'corrupt')
    checkpoint(source, 3)
    with pytest.raises(ValueError, match='integrity'):
        module.cycle(source, target, verified_this_process=verified, **arguments)
    assert (target / 'step-0000001' / 'model.pt').stat().st_size == 1024
    assert (target / 'step-0000003' / 'COMPLETE.json').exists()


@pytest.mark.parametrize('removed', ['directory', 'marker'])
def test_missing_cached_copy_or_marker_is_recovered_before_next_publication(tmp_path, removed):
    source, target, arguments = fixture(tmp_path)
    verified = set()
    for step in (1, 2):
        checkpoint(source, step)
        module.cycle(source, target, verified_this_process=verified, **arguments)
    missing = target/'step-0000002'
    if removed == 'directory':
        module.shutil.rmtree(missing)
    else:
        (missing/'RETENTION_OWNER.json').unlink()
    assert not module.cycle(source, target, verified_this_process=verified, **arguments)
    for step in (1, 2):
        copy = target/f'step-{step:07d}'
        assert (copy/'RETENTION_OWNER.json').is_file()
        assert (copy/'model.pt').read_bytes() == bytes([step])*1024
        assert (copy/'rank-00000.pt').read_bytes() == bytes([step])*1024
    assert len(list(source.glob('step-*'))) == 2


def test_expired_deadline_requests_stop_before_copy_or_terminal_success(tmp_path, monkeypatch):
    source, target, arguments = fixture(tmp_path)
    stop = tmp_path/'run.stop'
    monkeypatch.setattr(sys, 'argv', ['watch_checkpoint_backups',
        '--source', str(source), '--destination', str(target),
        '--spec-hash', arguments['spec_hash'], '--terminal-step', '4',
        '--deadline', '2000-01-01T00:00:00Z', '--stop-file', str(stop),
        '--log', str(arguments['log'])])

    def forbidden_cycle(*args, **kwargs):
        pytest.fail('Expired supervisor started a copy cycle')

    monkeypatch.setattr(module, 'cycle', forbidden_cycle)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    with pytest.raises(TimeoutError, match='deadline'):
        module.main()
    assert stop.is_file()
    events = [json.loads(line) for line in arguments['log'].read_text().splitlines()]
    assert [event['event'] for event in events] == ['backup_supervisor_failed_training_stop_requested']


def test_copy_finishing_after_deadline_requests_stop_instead_of_terminal_success(tmp_path, monkeypatch):
    source, target, arguments = fixture(tmp_path)
    stop = tmp_path/'run.stop'
    deadline = datetime(2026, 10, 16, tzinfo=timezone.utc)
    clock = [deadline-timedelta(seconds=1)]
    monkeypatch.setattr(module, 'datetime', SimpleNamespace(
        fromisoformat=datetime.fromisoformat, now=lambda timezone: clock[0]))
    monkeypatch.setattr(sys, 'argv', ['watch_checkpoint_backups',
        '--source', str(source), '--destination', str(target),
        '--spec-hash', arguments['spec_hash'], '--terminal-step', '4',
        '--deadline', deadline.isoformat(), '--stop-file', str(stop),
        '--log', str(arguments['log'])])

    def finish_copy(*args, **kwargs):
        clock[0] = deadline+timedelta(seconds=1)
        return True

    monkeypatch.setattr(module, 'cycle', finish_copy)
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    with pytest.raises(TimeoutError, match='deadline'):
        module.main()
    assert stop.is_file()
    events = [json.loads(line) for line in arguments['log'].read_text().splitlines()]
    assert [event['event'] for event in events] == ['backup_supervisor_failed_training_stop_requested']
