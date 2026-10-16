"""Bounded per-run backup supervisor; retain two verified EBS copies.

Host orchestration only. Bind the source snapshot, arguments and matching driver
stop-file in the production launch receipt. Never modify a source checkpoint.
Deadline checks bracket complete copy cycles; a copy is not interrupted by the
clock check. The driver's independent deadline controls training during a copy.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weekend_thinking.artifacts import atomic_json, canonical_hash, utc_now
from weekend_thinking.checkpoint_backup import copy_checkpoint, read_manifest, verify_files


def event(log_path, kind, **fields):
    with log_path.open('a') as stream:
        stream.write(json.dumps(dict(event=kind, utc=utc_now(), **fields), sort_keys=True)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(manifest, *, run_id, spec_hash, terminal_step):
    if (manifest.get('run_id') != run_id or manifest.get('spec_hash') != spec_hash
            or not 1 <= manifest['step'] <= terminal_step):
        raise ValueError('Checkpoint belongs to a different run/specification or exposure')


def cycle(source, destination, *, spec_hash, terminal_step, log,
          allow_same_filesystem=False, verified_this_process=None):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    verified_this_process = set() if verified_this_process is None else verified_this_process
    if source.is_symlink() or destination.is_symlink():
        raise ValueError('Run directories must not be symlinks')
    if source.resolve() == destination.resolve() or source.resolve() in destination.resolve().parents or destination.resolve() in source.resolve().parents:
        raise ValueError('Backup destination overlaps the source')
    owner = canonical_hash(dict(source=str(source.resolve()), destination=str(destination.resolve()),
                                spec_hash=spec_hash, retain=2, terminal_step=terminal_step))
    candidates = []
    for path in sorted(source.glob('step-*')):
        if not re.fullmatch(r'step-[0-9]{7}', path.name) or not (path/'COMPLETE.json').exists():
            continue
        if path.is_symlink():
            raise ValueError('Checkpoint directory is a symlink')
        manifest, raw = read_manifest(path)
        identity(manifest, run_id=source.name, spec_hash=spec_hash, terminal_step=terminal_step)
        if path.name != f"step-{manifest['step']:07d}":
            raise ValueError('Checkpoint directory/step mismatch')
        candidates.append((manifest['step'], path, manifest, raw))
    if not candidates:
        return False
    newest_step = max(row[0] for row in candidates)
    survivors = sorted(candidates, reverse=True)[:2]
    verified_now, changed = set(), False
    for step, original, manifest, raw in reversed(survivors):
        target = destination/original.name
        marker = target/'RETENTION_OWNER.json'
        expected = dict(owner=owner, manifest_sha256=hashlib.sha256(raw).hexdigest())
        key = (str(target), expected['manifest_sha256'])
        if marker.is_symlink():
            raise ValueError('Backup retention marker mismatch')
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError('Backup retention marker mismatch')
            if (target/'COMPLETE.json').read_bytes() != raw:
                raise ValueError('Published backup manifest changed')
        else:
            # A cached verification cannot establish that a copy still exists.
            # Reuse copy_checkpoint's verified recovery for a missing copy/marker.
            verified_this_process.discard(key)
        if key not in verified_this_process:
            receipt = copy_checkpoint(original, target,
                require_distinct_filesystems=not allow_same_filesystem)
            if not marker.exists():
                atomic_json(marker, expected)
            verified_this_process.add(key)
            verified_now.add(str(target))
            changed = True
            event(log, 'backup_verified', step=step, receipt=receipt)
    # Backfill the newest two sources on first/restarted supervision. Before
    # pruning or terminal success, independently verify BOTH retained copies;
    # an old publication marker does not establish current file integrity.
    if changed or newest_step == terminal_step:
        for step, original, manifest, raw in survivors:
            target = destination/original.name
            if str(target) not in verified_now:
                verify_files(target, manifest)
            if (target/'COMPLETE.json').read_bytes() != raw:
                raise ValueError('Retained backup manifest changed')
            event(log, 'retained_copy_verified', step=step)
    owned = []
    for path in destination.glob('step-*'):
        marker = path/'RETENTION_OWNER.json'
        if not marker.exists():
            continue  # Never adopt or remove an unmanaged directory.
        if path.is_symlink() or marker.is_symlink():
            raise ValueError('Managed backup or marker is a symlink')
        metadata = json.loads(marker.read_text())
        saved, saved_raw = read_manifest(path)
        identity(saved, run_id=source.name, spec_hash=spec_hash, terminal_step=terminal_step)
        if metadata != dict(owner=owner, manifest_sha256=hashlib.sha256(saved_raw).hexdigest()):
            raise ValueError('Managed backup ownership or manifest changed')
        owned.append((saved['step'], path, saved, saved_raw))
    # Delete a redundant EBS copy only after the new copy is fully verified and
    # the old source is independently intact. The two newest copies remain.
    for old_step, old, old_manifest, old_raw in sorted(owned, reverse=True)[2:]:
        original = source/old.name
        if original.is_symlink() or (original/'COMPLETE.json').read_bytes() != old_raw:
            raise ValueError('Retain old backup: source identity changed')
        verify_files(original, old_manifest)
        expected_files = set(old_manifest['files']) | {'COMPLETE.json', 'BACKUP.json',
            'BACKUP_SOURCE.json', 'RETENTION_OWNER.json'}
        if (set(p.name for p in old.iterdir()) != expected_files
                or any(p.is_symlink() or not p.is_file() for p in old.iterdir())):
            raise ValueError('Retain old backup: unexpected directory contents')
        event(log, 'remove_redundant_backup', step=old_step, path=str(old),
              source_manifest_sha256=hashlib.sha256(old_raw).hexdigest())
        shutil.rmtree(old)
    return newest_step == terminal_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--spec-hash', required=True)
    parser.add_argument('--terminal-step', type=int, required=True)
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00'))
    if deadline.tzinfo is None or not re.fullmatch('[0-9a-f]{64}', args.spec_hash) or args.terminal_step < 1:
        parser.error('Explicit UTC deadline, spec hash and positive terminal exposure required')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    def stop_on_signal(number, _frame):
        raise RuntimeError(f'Backup supervisor received signal {number}')
    signal.signal(signal.SIGTERM, stop_on_signal)
    verified = set()
    try:
        while True:
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError('Backup supervisor deadline reached before terminal checkpoint')
            complete = cycle(args.source, args.destination, spec_hash=args.spec_hash,
                terminal_step=args.terminal_step, log=args.log, verified_this_process=verified)
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError('Backup supervisor deadline reached before terminal checkpoint')
            if complete:
                event(args.log, 'terminal_backup_complete')
                return
            time.sleep(30)
    except BaseException as error:
        args.stop_file.parent.mkdir(parents=True, exist_ok=True)
        args.stop_file.touch(exist_ok=True)
        event(args.log, 'backup_supervisor_failed_training_stop_requested',
              error=f'{type(error).__name__}: {error}', stop_file=str(args.stop_file))
        raise


if __name__ == '__main__':
    main()
