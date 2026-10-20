"""Scoped rolling checkpoints for V28, isolated from all historical runs.

Only this new study's intermediate payloads may retire after a newer full state
passes readback. Every terminal model remains. At most one latest full per arm
is retained across the two declared seeds. This is a prospective storage policy,
not a change to historical checkpoint retention.
"""
from contextlib import contextmanager
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time

from weekend_thinking.artifacts import atomic_json,canonical_hash,sha256_file

RUN_ROOT=Path('/home/ubuntu/dario/local/science-runs/controlled-computation-v28')
MODEL64_ROOT=Path('/home/ubuntu/dario/local/nvme-cpt/models/controlled-computation-v28-development64')
POLICY='v28_latest_full_per_arm_keep_terminal_models'
NAME=re.compile(r'program-(direct|trace|pause)-seed(2026102028|2026102029)-v1')


@contextmanager
def export_lease(deadline):
    """Serialize large exports so three trainers cannot overrun the disk reserve."""
    RUN_ROOT.mkdir(parents=True,exist_ok=True)
    path=RUN_ROOT/'.checkpoint-export.lock'
    if path.is_symlink():raise ValueError('A regular scoped lock file is required')
    with path.open('a+b') as stream:
        while True:
            try:
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if datetime.now(timezone.utc)>=deadline:
                    raise TimeoutError('Checkpoint export lease deadline')
                time.sleep(.1)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def publish_phase(evidence,name,p):
    if re.fullmatch(r'(lease|retained)-[0-9]+',name) is None:
        raise ValueError('Explicit checkpoint host phase required')
    atomic_json(Path(evidence)/(name+'.json'),dict(spec_hash=canonical_hash(p),phase=name))


def await_phase(evidence,name,p,deadline):
    """Wait on the host, outside the installed Neuron device-barrier timeout."""
    path=Path(evidence)/(name+'.json')
    while not path.exists():
        if datetime.now(timezone.utc)>=deadline:raise TimeoutError('Checkpoint host phase deadline')
        time.sleep(.1)
    if path.is_symlink() or read(path)!=dict(spec_hash=canonical_hash(p),phase=name):
        raise ValueError('Checkpoint host phase belongs to another protocol')


def read(path):return json.loads(Path(path).read_text())


def scoped(run):
    run=Path(run)
    if (not run.is_absolute() or run.resolve()!=run or run.parent!=RUN_ROOT.resolve()
            or NAME.fullmatch(run.name) is None):
        raise ValueError('Only newly created V28 study runs may use rolling retention')
    return run


def full_readback(run,directory,p):
    run=scoped(run);directory=Path(directory)
    if (p.get('checkpoint_retention')!=POLICY or directory.parent!=run
            or directory.resolve()!=directory or not directory.name.startswith('step-')):
        raise ValueError('Declared scoped checkpoint and retention policy required')
    m=read(directory/'COMPLETE.json');step=m['step']
    expected={'model.pt',*[f'rank-{i:05d}.pt' for i in range(4)]}
    if (m['run_id']!=run.name or m['world_size']!=4 or set(m['files'])!=expected
            or directory.name!=f'step-{step:07d}' or m['spec_hash']!=canonical_hash(p)
            or m['metadata']['pipeline_schema']!='permutation-program-sft-v28'
            or m['metadata']['arm']!=p['program_arm'] or m['metadata']['seed']!=p['seed']
            or NAME.fullmatch(run.name).groups()!=(p['program_arm'],str(p['seed']))):
        raise ValueError('Full-state identity differs from the released program run')
    for name,value in m['files'].items():
        path=directory/name
        if path.is_symlink() or path.stat().st_size!=value['bytes'] or sha256_file(path)!=value['sha256']:
            raise ValueError('New full-state readback failed; retain every older payload')
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    receipt=dict(status='FULL_STATE_READBACK_FSYNC',utc=datetime.now(timezone.utc).isoformat(),
        run_id=run.name,step=step,COMPLETE_sha256=sha256_file(directory/'COMPLETE.json'),
        files=m['files'],source_sha256=sha256_file(__file__))
    atomic_json(directory/'READBACK.json',receipt)
    return receipt


def retire_payload(directory,replacement,*,keep_model):
    """Keep manifests/metrics; unlink only named payloads under this study root."""
    directory=Path(directory);run=scoped(directory.parent)
    if directory.resolve()!=directory or not re.fullmatch(r'step-[0-9]{7}',directory.name):
        raise ValueError('A real scoped checkpoint directory is required')
    m=read(directory/'COMPLETE.json')
    if m['run_id']!=run.name or set(m['files'])!={'model.pt',*[f'rank-{i:05d}.pt' for i in range(4)]}:
        raise ValueError('Cannot retire an unrelated checkpoint')
    receipt=read(directory/'READBACK.json')
    if receipt['COMPLETE_sha256']!=sha256_file(directory/'COMPLETE.json'):
        raise ValueError('Earlier full-state readback receipt changed')
    # The caller supplies a new verified complete state from the same arm.
    old_identity=NAME.fullmatch(run.name).groups()
    new_identity=NAME.fullmatch(replacement.get('run_id',''))
    if (replacement.get('status')!='FULL_STATE_READBACK_FSYNC' or new_identity is None
            or new_identity.group(1)!=old_identity[0]
            or int(new_identity.group(2))<int(old_identity[1])
            or replacement['run_id']==run.name and replacement['step']<=m['step']):
        raise ValueError('A newer verified replacement from the same arm is required')
    replacement_dir=RUN_ROOT/replacement['run_id']/f'step-{replacement["step"]:07d}'
    if (read(replacement_dir/'READBACK.json')!=replacement
            or sha256_file(replacement_dir/'COMPLETE.json')!=replacement['COMPLETE_sha256']
            or any(not (replacement_dir/name).is_file() for name in replacement['files'])):
        raise ValueError('The full replacement must still exist at its verified location')
    names=[name for name in m['files'] if name!='model.pt' or not keep_model]
    for name in names:
        path=directory/name
        if path.is_symlink():raise ValueError('Never unlink a checkpoint payload symlink')
    intent=dict(status='ROLLING_RETIREMENT_INTENT',policy=POLICY,
        utc=datetime.now(timezone.utc).isoformat(),replacement=replacement,
        model_retained=keep_model,only_payload_names=names,old_COMPLETE_sha256=sha256_file(directory/'COMPLETE.json'))
    atomic_json(directory/'RETIREMENT_INTENT.json',intent,overwrite=True)
    for name in names:
        path=directory/name
        if path.exists():path.unlink()
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    atomic_json(directory/'PAYLOAD_RETIRED.json',dict(intent,status='ROLLING_PAYLOAD_RETIRED'),overwrite=True)


def preserve_development64(run,directory,receipt):
    """Keep the declared model64 milestone separately from rolling optimizer state."""
    source=directory/'model.pt';expected=receipt['files']['model.pt']
    MODEL64_ROOT.mkdir(parents=True,exist_ok=True)
    target=MODEL64_ROOT/(run.name+'-step64.pt')
    if target.exists():
        if target.is_symlink() or sha256_file(target)!=expected['sha256']:
            raise ValueError('Existing development64 model differs')
        return
    if shutil.disk_usage(MODEL64_ROOT).free<35*2**30+expected['bytes']:
        raise RuntimeError('Development64 preservation would cross the disk reserve')
    temporary=target.with_suffix('.partial')
    with source.open('rb') as src, temporary.open('xb') as dst:
        shutil.copyfileobj(src,dst,length=8*2**20);dst.flush();os.fsync(dst.fileno())
    if sha256_file(temporary)!=expected['sha256']:
        raise ValueError('Development64 model readback failed; preserve original')
    temporary.rename(target)
    atomic_json(target.with_suffix('.json'),dict(status='DEVELOPMENT64_MODEL_READBACK_FSYNC',
        model_sha256=expected['sha256'],source=str(source),source_FULL_readback=receipt,
        limits='Developmental weights retained on same-instance NVMe; not a science endpoint or off-instance backup.'))
    fd=os.open(MODEL64_ROOT,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def rotate(run,newest,p):
    """Protect current full; prune only old intermediate payloads from this run."""
    run=scoped(run);receipt=full_readback(run,newest,p)
    if receipt['step']==64:preserve_development64(run,Path(newest),receipt)
    for old in sorted(run.glob('step-*')):
        if old==Path(newest) or (old/'PAYLOAD_RETIRED.json').exists():continue
        if (old/'COMPLETE.json').exists() and read(old/'COMPLETE.json')['step']<receipt['step']:
            retire_payload(old,receipt,keep_model=False)
    # Wave two may retire wave one's optimizer only after its completed terminal
    # full state was independently audited. Its terminal model is never removed.
    if p['seed']==2026102029:
        previous=RUN_ROOT/f'program-{p["program_arm"]}-seed2026102028-v1'
        terminal=previous/'step-0004096'
        if terminal.exists() and not (terminal/'PAYLOAD_RETIRED.json').exists():
            audit=read(previous/'terminal-full-audit.json')
            if (audit.get('status')!='PASS_V28_TERMINAL_ALL_STATE' or audit.get('step')!=4096
                    or audit.get('COMPLETE_sha256')!=sha256_file(terminal/'COMPLETE.json')):
                raise ValueError('Previous seed terminal full audit must precede optimizer retirement')
            retire_payload(terminal,receipt,keep_model=True)
    return receipt
