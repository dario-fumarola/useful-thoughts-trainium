"""Copy physically closed evaluation groups to EBS without reading answer content."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import shutil
import time

from weekend_thinking.artifacts import atomic_json, sha256_file

R = Path('/home/ubuntu/dario')
E = R / 'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
O = R / 'local/nvme-cpt/evaluation/permutation-program-v28-fixed-v1'
B = R / 'local/backups/evaluation/permutation-program-v28-fixed-v1'
read = lambda path: json.loads(Path(path).read_text())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    os.fsync(descriptor)
    os.close(descriptor)


def main():
    registry = read(E / 'registry.json')
    work = E / 'EXECUTION'
    deadline = datetime.fromisoformat(read(work / 'plan.json')['worker_deadline_utc']).timestamp() + 300
    B.mkdir(parents=True, exist_ok=False)
    receipts = B / 'receipts'
    receipts.mkdir()
    done = {}
    while True:
        for job in registry['jobs']:
            key = job['key']
            if key in done:
                continue
            closed = work / (key + '-closed.json')
            if not closed.exists():
                continue
            assert read(closed)['exit_code'] == 0, 'Archive failed/partial groups separately; no model retry'
            closure = read(work / key / 'closure.json')
            assert closure['assigned_physical_device_empty'] and closure['all_original_session_processes_absent']
            assert closure['actual_attempts'] == job['attempts']
            source = Path(job['output'])
            assert source.resolve().is_relative_to(O.resolve()) and source.is_dir()
            assert not any(path.is_symlink() for path in source.rglob('*'))
            paths = sorted(path for path in source.rglob('*') if path.is_file())
            size = sum(path.stat().st_size for path in paths)
            assert shutil.disk_usage(R).free > 35 * 2**30 + size + 2**30
            staging = B / (key + '.staging-v1')
            target = B / key
            assert not staging.exists() and not target.exists()
            shutil.copytree(source, staging)
            files = {}
            for path in paths:
                relative = str(path.relative_to(source))
                copied = staging / relative
                digest = sha256_file(path)
                assert copied.stat().st_size == path.stat().st_size and sha256_file(copied) == digest
                with copied.open('rb') as stream:
                    os.fsync(stream.fileno())
                files[relative] = dict(bytes=path.stat().st_size, sha256=digest)
            assert len(files) == sum(p.is_file() for p in staging.rglob('*'))
            for directory in sorted((p for p in staging.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                sync_directory(directory)
            sync_directory(staging)
            staging.rename(target)
            sync_directory(B)
            receipt = dict(status='PASS_CLOSED_EVALUATION_FULL_FILE_READBACK_FSYNC',
                utc=datetime.now(timezone.utc).isoformat(), key=key, source=str(source), copy=str(target),
                physical_closure_sha256=sha256_file(work / key / 'closure.json'),
                original_protocol_sha256=job['protocol_sha256'], files=files, bytes=size,
                source_sha256=sha256_file(__file__),
                limits='Byte/hash copy only; no answer parsing/scoring/model calls. Same-instance EBS,not off-instance backup.')
            atomic_json(receipts / (key + '.json'), receipt)
            sync_directory(receipts)
            done[key] = dict(receipt=str(receipts / (key + '.json')), bytes=size)
            print(json.dumps(dict(status=receipt['status'], key=key, files=len(files), bytes=size, groups_protected=len(done))), flush=True)
        if len(done) == 23:
            atomic_json(B / 'COMPLETE.json', dict(status='PASS_ALL23_CLOSED_EVALUATIONS_PROTECTED', groups=done,
                        source_sha256=sha256_file(__file__), model_calls=0, score_inspection=False))
            sync_directory(B)
            print('ALL23_CLOSED_EVALUATIONS_PROTECTED_WITHOUT_SCORE_INSPECTION', flush=True)
            return
        if (work / 'completion.json').exists():
            assert read(work / 'completion.json')['status'] == 'COMPLETE_ALL_FIXED23_GROUPS_PHYSICALLY_CLOSED_NO_SCORES_REVIEWED', 'Incomplete run retained; preserve partials separately'
        assert time.time() < deadline, 'Finite archival wait elapsed; no automatic model retry'
        time.sleep(30)


if __name__ == '__main__':
    main()
