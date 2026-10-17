"""One bounded host leader, four fixed runs, sequential verified backup cycles.

Launch only with the separately frozen helper and roster hashes. No trainer is
started, source directory created, or source checkpoint deleted. A rejected
duplicate leader does not request training stops. Once elected, failures and the
UTC deadline request every configured STOP and preserve incomplete copies.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time


class LeaderBusy(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc)


def read_roster(path, expected_sha256):
    path = Path(path)
    if path.is_symlink() or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Require an ordinary roster and its frozen SHA256')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Frozen backup roster changed')
    roster = json.loads(raw)
    if (roster['schema'] != 'shared-checkpoint-backup-v1' or len(roster['runs']) != 4
            or not re.fullmatch('[0-9a-f]{64}', roster['spec_hash'])
            or not re.fullmatch('[0-9a-f]{64}', roster['source_sha256'])
            or type(roster['terminal_step']) is not int or roster['terminal_step'] < 2):
        raise ValueError('Require four runs, fixed hashes, and terminal exposure of at least two')
    deadline = datetime.fromisoformat(roster['deadline_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None:
        raise ValueError('Explicit timezone-aware UTC deadline required')
    paths = [roster[key] for key in ('source_root', 'lock_file', 'log_file', 'state_file')]
    paths += [run[key] for run in roster['runs'] for key in ('source', 'destination', 'stop_file')]
    if any(not Path(value).is_absolute() or Path(value).is_symlink() for value in paths):
        raise ValueError('Roster paths must be absolute and not symlinks')
    sources = [Path(run['source']).resolve() for run in roster['runs']]
    destinations = [Path(run['destination']).resolve() for run in roster['runs']]
    controls = [Path(roster[key]).resolve() for key in ('lock_file', 'log_file', 'state_file')]
    controls += [Path(run['stop_file']).resolve() for run in roster['runs']]
    run_ids = [run['run_id'] for run in roster['runs']]
    if (len(set(run_ids)) != 4 or any(name != source.name for name, source in zip(run_ids, sources))
            or len(set(controls)) != len(controls)):
        raise ValueError('Run identities and control paths must be unique')
    storage = sources + destinations
    if any(a == b or a in b.parents or b in a.parents
           for index, a in enumerate(storage) for b in storage[index+1:]):
        raise ValueError('Run and backup directories must not overlap')
    if any(control == directory or directory in control.parents for control in controls for directory in storage):
        raise ValueError('Control files must be outside all source and backup directories')
    if any(path.exists() and not path.is_file() for path in controls):
        raise ValueError('Existing control paths must be regular files')
    roster['roster_sha256'] = expected_sha256
    return roster


@contextmanager
def leader_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('Leader lock must be a regular file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LeaderBusy('Shared backup leader already holds the lock') from error
        yield
    finally:
        os.close(fd)


def request_stops(roster, reason):
    errors = {}
    for run in roster['runs']:
        path = Path(run['stop_file'])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError('STOP must be a regular non-symlink file')
            if not path.exists():
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    stream.write(reason + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
        except Exception as error:
            errors[run['run_id']] = f'{type(error).__name__}: {error}'
    return errors


def supervise(roster, cycle, *, atomic_json, event, clock=utc_now, sleep=time.sleep, install_signals=True):
    deadline = datetime.fromisoformat(roster['deadline_utc'].replace('Z', '+00:00'))
    completed, verified = set(), {run['run_id']: set() for run in roster['runs']}
    log, state = Path(roster['log_file']), Path(roster['state_file'])
    handlers = {}

    def check_deadline():
        if clock() >= deadline:
            raise TimeoutError('Shared backup supervisor UTC deadline reached')

    def publish(status, **fields):
        atomic_json(state, dict(status=status, utc=clock().isoformat(), pid=os.getpid(),
            roster_sha256=roster['roster_sha256'], terminal_step=roster['terminal_step'],
            terminal_verified_runs=sorted(completed), **fields), overwrite=True)

    def backup(run):
        check_deadline()
        event(log, 'shared_run_cycle_start', run_id=run['run_id'])
        complete = cycle(Path(run['source']), Path(run['destination']), spec_hash=roster['spec_hash'],
            terminal_step=roster['terminal_step'], log=log, verified_this_process=verified[run['run_id']])
        check_deadline()
        if complete and len([path for path in Path(run['destination']).glob('step-*')
                             if (path/'RETENTION_OWNER.json').exists()]) != 2:
            raise ValueError('Terminal run does not have two verified managed backups')
        event(log, 'shared_run_cycle_end', run_id=run['run_id'], terminal_verified=complete)
        return complete

    def interrupted(number, _frame):
        raise RuntimeError(f'Shared backup supervisor received signal {number}')

    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        state.parent.mkdir(parents=True, exist_ok=True)
        check_deadline()
        if install_signals:
            handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, interrupted)
        publish('RUNNING')
        while True:
            for run in roster['runs']:
                if run['run_id'] not in completed and backup(run):
                    completed.add(run['run_id'])
                publish('RUNNING')
            if len(completed) == len(roster['runs']):
                # Earlier terminal runs may have waited for the later chains.
                # Recheck every retained pair once before shared success.
                for run in roster['runs']:
                    if not backup(run):
                        raise ValueError('A completed terminal checkpoint disappeared')
                check_deadline()
                publish('COMPLETE')
                event(log, 'shared_all_terminal_backups_verified')
                return
            check_deadline()
            sleep(min(30, (deadline-clock()).total_seconds()))
    except BaseException as error:
        reason = f'{type(error).__name__}: {error}'
        stop_errors = request_stops(roster, reason)
        try:
            publish('FAILED_STOP_REQUESTED', error=reason, stop_errors=stop_errors)
            event(log, 'shared_backup_failed_all_stops_requested', error=reason, stop_errors=stop_errors)
        except Exception as reporting_error:
            print(json.dumps(dict(error=reason, stop_errors=stop_errors,
                                  reporting_error=str(reporting_error))), file=sys.stderr)
        raise
    finally:
        if install_signals:
            for number, handler in handlers.items():
                signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roster', type=Path, required=True)
    parser.add_argument('--roster-sha256', required=True)
    args = parser.parse_args()
    roster = read_roster(args.roster, args.roster_sha256)
    sys.dont_write_bytecode = True
    sys.path.insert(0, roster['source_root'])
    from weekend_thinking.artifacts import atomic_json, source_fingerprint
    if source_fingerprint(roster['source_root'])['sha256'] != roster['source_sha256']:
        raise ValueError('Pinned active backup source changed')
    from scripts.watch_checkpoint_backups import cycle, event
    if Path(cycle.__code__.co_filename).resolve() != Path(roster['source_root'])/'scripts/watch_checkpoint_backups.py':
        raise ValueError('Backup cycle imported from a different source snapshot')
    with leader_lock(roster['lock_file']):
        supervise(roster, cycle, atomic_json=atomic_json, event=event)


if __name__ == '__main__':
    main()
