"""Verified, restartable local copies of complete Neuron checkpoints.

This does not load model tensors or contact an external storage service. A copy
on EBS protects against scratch-volume loss; it remains on the same instance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil

from .artifacts import atomic_json, sha256_file, utc_now


def read_manifest(directory: Path) -> tuple[dict, bytes]:
    path = directory / 'COMPLETE.json'
    if path.is_symlink():
        raise ValueError('Checkpoint manifest must be an ordinary file.')
    raw = path.read_bytes()
    manifest = json.loads(raw)
    world, step = manifest.get('world_size'), manifest.get('step')
    if (manifest.get('schema') != 'neuron-v1' or type(world) is not int or
            world < 1 or type(step) is not int or step < 0):
        raise ValueError('Expected a complete Neuron checkpoint.')
    expected = {'model.pt'} | {f'rank-{rank:05d}.pt' for rank in range(world)}
    if set(manifest.get('files', {})) != expected:
        raise ValueError('Incomplete or unsafe checkpoint file set.')
    for name, entry in manifest['files'].items():
        if (type(entry.get('bytes')) is not int or entry['bytes'] < 1 or
                not re.fullmatch('[0-9a-f]{64}', entry.get('sha256', ''))):
            raise ValueError(f'Invalid checkpoint file metadata: {name}')
        target = directory / name
        if target.is_symlink() or not target.is_file():
            raise ValueError(f'Checkpoint file is missing or a symlink: {name}')
    return manifest, raw


def verify_files(directory: Path, manifest: dict) -> None:
    for name, entry in manifest['files'].items():
        path = directory / name
        if (path.is_symlink() or path.stat().st_size != entry['bytes'] or
                sha256_file(path) != entry['sha256']):
            raise ValueError(f'Checkpoint integrity mismatch: {name}')


def _signature(path: Path) -> tuple[int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def copy_checkpoint(source: str | Path, destination: str | Path, *,
                    reserve_bytes: int = 20 * 1024**3,
                    require_distinct_filesystems: bool = True) -> dict:
    """Copy trusted own-run tensors; publish COMPLETE and the directory last.

    A failed copy leaves a clearly incomplete sibling. A retry checks and reuses
    only whole files whose hashes match the original source manifest. Existing
    complete destinations are verified rather than overwritten.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if reserve_bytes < 0 or source == destination:
        raise ValueError('Invalid backup destination or reserve.')
    if source.is_symlink() or destination.is_symlink():
        raise ValueError('Checkpoint directories must not be symlinks.')
    manifest, raw = read_manifest(source)
    digest = hashlib.sha256(raw).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_device, target_device = source.stat().st_dev, destination.parent.stat().st_dev
    if require_distinct_filesystems and source_device == target_device:
        raise ValueError('Backup must be on a different filesystem from scratch.')
    if destination.exists():
        saved, saved_raw = read_manifest(destination)
        if saved_raw != raw:
            raise ValueError('Destination belongs to a different checkpoint.')
        verify_files(destination, saved)
        return {'status': 'already_verified', 'destination': str(destination),
                'manifest_sha256': digest, 'bytes': sum(e['bytes'] for e in saved['files'].values())}

    staging = destination.with_name('.' + destination.name + '.incomplete')
    if staging.is_symlink():
        raise ValueError('Backup staging directory must not be a symlink.')
    identity = dict(schema=1, source=str(source.resolve()), manifest_sha256=digest)
    if staging.exists():
        if json.loads((staging / 'BACKUP_SOURCE.json').read_text()) != identity:
            raise ValueError('Staging directory belongs to a different checkpoint.')
    else:
        staging.mkdir()
        atomic_json(staging / 'BACKUP_SOURCE.json', identity)

    signatures = {name: _signature(source / name) for name in manifest['files']}
    pending = []
    for name, entry in manifest['files'].items():
        path = staging / name
        if path.is_symlink():
            raise ValueError(f'Backup file must not be a symlink: {name}')
        if path.exists():
            if path.stat().st_size == entry['bytes'] and sha256_file(path) == entry['sha256']:
                continue
            raise ValueError(f'Whole staged file is corrupt: {name}')
        pending.append(name)
    required = sum(manifest['files'][name]['bytes'] for name in pending)
    if shutil.disk_usage(staging).free < required + reserve_bytes:
        raise ValueError('Insufficient backup space; preserve existing checkpoints.')

    for name in pending:
        entry = manifest['files'][name]
        temp = staging / ('.' + name + '.part')
        if temp.is_symlink():
            raise ValueError(f'Partial backup file must not be a symlink: {name}')
        hasher, copied = hashlib.sha256(), 0
        with (source / name).open('rb') as reader, temp.open('wb') as writer:
            for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                writer.write(chunk)
                hasher.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if copied != entry['bytes'] or hasher.hexdigest() != entry['sha256']:
            raise ValueError(f'Source integrity mismatch while copying: {name}')
        if sha256_file(temp) != entry['sha256']:
            raise ValueError(f'Backup read verification failed: {name}')
        os.replace(temp, staging / name)

    if (source / 'COMPLETE.json').read_bytes() != raw or any(
            _signature(source / name) != signatures[name] for name in signatures):
        raise ValueError('Source checkpoint changed during backup.')
    # Also validate files reused from an earlier interrupted attempt.
    verify_files(staging, manifest)
    receipt = dict(schema=1, source=str(source.resolve()), destination=str(destination),
                   manifest_sha256=digest, copied_utc=utc_now(),
                   bytes=sum(e['bytes'] for e in manifest['files'].values()),
                   source_device=source_device, destination_device=target_device,
                   distinct_filesystems=source_device != target_device,
                   external_backup=False)
    atomic_json(staging / 'BACKUP.json', receipt, overwrite=True)
    complete = staging / 'COMPLETE.json'
    if complete.is_symlink():
        raise ValueError('Staged complete manifest must not be a symlink.')
    if complete.exists():
        if complete.read_bytes() != raw:
            raise ValueError('Staged complete manifest changed.')
    else:
        with complete.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    staging_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(staging_fd)
    finally:
        os.close(staging_fd)
    os.rename(staging, destination)
    parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return dict(status='copied_and_verified', **receipt)
