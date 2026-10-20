"""Protect latest per-run models and one full recovery state; keep all originals.

Lossless streams are fully read back before a backup-only predecessor retires.
Metrics are snapshotted through the checkpoint cursor, not copied while growing.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tarfile

import zstandard

from train.neuron_driver import verify_checkpoint
from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed',type=int,choices=(2026102029,2026102030),required=True)
parser.add_argument('--step',type=int,choices=(64,640,1280),required=True)
args=parser.parse_args()
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-cpt-scaffold-v29-science-v1'
B=R/'local/backups/science/real-cpt-scaffold-v29';B.mkdir(parents=True,exist_ok=True)
registry=json.loads((C/'protocols-prepared.json').read_text());floor=35*2**30

def syncdir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def restored_digest(path):
    h=hashlib.sha256();count=0
    with path.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        while block:=reader.read(16*2**20):h.update(block);count+=len(block)
    return dict(bytes=count,sha256=h.hexdigest())

with (C/'large-EBS-writer.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    index_path=B/'INDEX.json';index=json.loads(index_path.read_text()) if index_path.exists() else dict(models={},full=None,history=[])
    jobs={name:job for name,job in registry['jobs'].items() if job['seed']==args.seed}
    assert len(jobs)==3
    checkpoints={};new_models={}
    for name,job in jobs.items():
        p=json.loads(Path(job['protocol']).read_text());out=Path(job['output']);cp=out/f'step-{args.step:07d}'
        if args.step==64:
            audit=json.loads((C/'jobs'/name/'stage64/independent-audit.json').read_text())
            assert audit['status'].startswith('PASS_')
        manifest=verify_checkpoint(cp,canonical_hash(p),8);checkpoints[name]=(cp,manifest)
        source=cp/'model.pt';expected=manifest['files']['model.pt']
        assert sha256_file(source)==expected['sha256']
        folder=B/'models'/name;folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'step-{args.step:07d}-model.pt.zst';stage=folder/(target.name+'.staging')
        assert not target.exists() and not stage.exists()
        assert shutil.disk_usage(B).free>floor+expected['bytes']+2**30
        with stage.open('xb') as raw:
            with zstandard.ZstdCompressor(level=3,threads=4).stream_writer(raw,closefd=False) as compressed:
                with source.open('rb') as stream:
                    while block:=stream.read(16*2**20):compressed.write(block)
            raw.flush();os.fsync(raw.fileno())
        assert restored_digest(stage)==expected
        digest=sha256_file(stage);os.rename(stage,target);syncdir(folder)
        item=dict(archive=str(target),archive_sha256=digest,archive_bytes=target.stat().st_size,
            original_model=str(source),original=expected,step=args.step,job=name,COMPLETE_sha256=sha256_file(cp/'COMPLETE.json'))
        prior=index['models'].get(name)
        index['models'][name]=item;index['history'].append(dict(event='model_verified',**item))
        atomic_json(index_path,index,overwrite=index_path.exists())
        if prior:
            original=Path(prior['original_model']);old=Path(prior['archive'])
            assert sha256_file(original)==prior['original']['sha256'] and sha256_file(old)==prior['archive_sha256']
            old.unlink();syncdir(old.parent)
            index['history'].append(dict(event='backup_only_older_model_retired',archive=str(old),original_retained=str(original)))
            atomic_json(index_path,index,overwrite=True)
        new_models[name]=item
        print('MODEL_EBS_READBACK_PASS',name,args.step,flush=True)

    # Operational choice only: most recently completed checkpoint, never its loss.
    name=max(checkpoints,key=lambda k:(checkpoints[k][1]['created_utc'],k))
    cp,manifest=checkpoints[name];job=jobs[name];out=Path(job['output'])
    snapshot=Path('/dev/shm')/f'cpt29-envelope-{args.seed}-{args.step}';snapshot.mkdir(mode=0o700,exist_ok=False)
    lines=(out/'metrics.jsonl').read_bytes().splitlines(keepends=True)[:args.step]
    assert [json.loads(v)['step'] for v in lines]==list(range(1,args.step+1))
    (snapshot/'metrics.jsonl').write_bytes(b''.join(lines));shutil.copy2(out/'run.json',snapshot/'run.json')
    shutil.copy2(job['protocol'],snapshot/'protocol.json')
    files={f'step-{args.step:07d}/'+p.name:p for p in cp.iterdir() if p.is_file()}
    files.update({'run-envelope/'+p.name:p for p in snapshot.iterdir()})
    expected={key:dict(bytes=path.stat().st_size,sha256=sha256_file(path)) for key,path in files.items()}
    raw_bytes=sum(v['bytes'] for v in expected.values())
    assert shutil.disk_usage(B).free>floor+raw_bytes+2**30
    target=B/f'full-{name}-step-{args.step:07d}.tar.zst';stage=B/(target.name+'.staging')
    assert not target.exists() and not stage.exists()
    with stage.open('xb') as raw:
        with zstandard.ZstdCompressor(level=3,threads=4).stream_writer(raw,closefd=False) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as archive:
                for key,path in sorted(files.items()):archive.add(path,arcname=key,recursive=False)
        raw.flush();os.fsync(raw.fileno())
    observed={}
    with stage.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as compressed:
        with tarfile.open(fileobj=compressed,mode='r|') as archive:
            for member in archive:
                assert member.isfile() and member.name in expected and member.name not in observed
                stream=archive.extractfile(member);h=hashlib.sha256();count=0
                while block:=stream.read(16*2**20):h.update(block);count+=len(block)
                observed[member.name]=dict(bytes=count,sha256=h.hexdigest())
                assert observed[member.name]==expected[member.name]
    assert observed==expected
    digest=sha256_file(stage);os.rename(stage,target);syncdir(B)
    prior=index['full'];full=dict(archive=str(target),archive_sha256=digest,archive_bytes=target.stat().st_size,
        original_checkpoint=str(cp),COMPLETE_sha256=sha256_file(cp/'COMPLETE.json'),job=name,step=args.step,files=expected)
    index['full']=full;index['history'].append(dict(event='full_verified',**full));atomic_json(index_path,index,overwrite=True)
    if prior:
        old=Path(prior['archive']);original=Path(prior['original_checkpoint'])
        assert sha256_file(old)==prior['archive_sha256'] and sha256_file(original/'COMPLETE.json')==prior['COMPLETE_sha256']
        old.unlink();syncdir(B)
        index['history'].append(dict(event='backup_only_previous_full_retired',archive=str(old),original_retained=str(original)))
        atomic_json(index_path,index,overwrite=True)
    snapshot_index=C/f'protection-index-seed{args.seed}-step{args.step}.json'
    with snapshot_index.open('xb') as dest:
        dest.write(index_path.read_bytes());dest.flush();os.fsync(dest.fileno())
    syncdir(B)
    atomic_json(C/f'protection-seed{args.seed}-step{args.step}.json',dict(status='PASS_ALL_THREE_LATEST_MODEL_ARCHIVES_AND_ONE_LATEST_FULL_EBS_READBACK',
        model_archives=new_models,full_archive=full,index_sha256=sha256_file(index_path),
        immutable_index_snapshot=str(snapshot_index),
        all_original_full_states_retained_in_RAM=True,scope='Same-instance EBS; every listed latest model and one full optimizer state. Other full originals remain in volatile RAM,not all on EBS.'))
    syncdir(C)
    print('FULL_EBS_MEMBER_READBACK_PASS',name,args.step,flush=True)
