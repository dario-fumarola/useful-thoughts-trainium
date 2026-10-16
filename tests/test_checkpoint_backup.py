import hashlib
import json

import pytest

import weekend_thinking.checkpoint_backup as backup
from weekend_thinking.checkpoint_backup import copy_checkpoint


def checkpoint(path):
    path.mkdir()
    files = {}
    for name, data in [('model.pt', b'model payload'), ('rank-00000.pt', b'rank zero'),
                       ('rank-00001.pt', b'rank one')]:
        (path / name).write_bytes(data)
        files[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest = dict(schema='neuron-v1', step=3, world_size=2, run_id='fixture',
                    spec_hash='frozen', files=files)
    (path / 'COMPLETE.json').write_text(json.dumps(manifest))
    return manifest


def copy(source, target):
    return copy_checkpoint(source, target, reserve_bytes=0, require_distinct_filesystems=False)


def test_verified_copy_and_idempotent_existing_destination(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'backup'
    manifest = checkpoint(source)
    assert copy(source, target)['status'] == 'copied_and_verified'
    assert (target / 'COMPLETE.json').read_bytes() == (source / 'COMPLETE.json').read_bytes()
    for name in manifest['files']:
        assert (target / name).read_bytes() == (source / name).read_bytes()
    assert copy(source, target)['status'] == 'already_verified'
    (target / 'model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError, match='integrity mismatch'):
        copy(source, target)


def test_corrupt_source_never_publishes_complete_backup(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'backup'
    checkpoint(source)
    (source / 'rank-00000.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Source integrity mismatch'):
        copy(source, target)
    assert not target.exists()
    assert not (tmp_path / '.backup.incomplete' / 'COMPLETE.json').exists()
    # Fix the source and restart: verified model file can be reused; partial rank cannot.
    (source / 'rank-00000.pt').write_bytes(b'rank zero')
    assert copy(source, target)['status'] == 'copied_and_verified'


def test_rejects_unsafe_manifest_and_incomplete_source(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'backup'
    manifest = checkpoint(source)
    manifest['files']['../escape'] = manifest['files'].pop('model.pt')
    (source / 'COMPLETE.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='unsafe checkpoint file set'):
        copy(source, target)
    assert not target.exists()


def test_rejects_same_filesystem_and_symlinks(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'backup'
    checkpoint(source)
    with pytest.raises(ValueError, match='different filesystem'):
        copy_checkpoint(source, target, reserve_bytes=0)
    (source / 'model.pt').unlink()
    (source / 'model.pt').symlink_to(source / 'rank-00000.pt')
    with pytest.raises(ValueError, match='symlink'):
        copy(source, target)


def test_capacity_failure_preserves_source_and_does_not_publish(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'backup'
    checkpoint(source)
    before = (source / 'model.pt').read_bytes()
    with pytest.raises(ValueError, match='Insufficient backup space'):
        copy_checkpoint(source, target, reserve_bytes=10**30, require_distinct_filesystems=False)
    assert (source / 'model.pt').read_bytes() == before
    assert not target.exists()


def test_source_mutation_during_copy_prevents_publication(tmp_path, monkeypatch):
    source, target = tmp_path / 'source', tmp_path / 'backup'
    checkpoint(source)
    original_hash = backup.sha256_file

    def mutate_after_readback(path):
        digest = original_hash(path)
        if path.name == '.model.pt.part':
            (source / 'model.pt').write_bytes(b'changed model')
        return digest

    monkeypatch.setattr(backup, 'sha256_file', mutate_after_readback)
    with pytest.raises(ValueError, match='Source checkpoint changed'):
        copy(source, target)
    assert not target.exists()
