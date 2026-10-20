"""Check the closed CPU diagnostic once and protect all small original evidence."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,os,shutil,subprocess,sys,time
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/selector-TRAIN-decisions-CPU-v1';W=R/'local/worktrees/document-evidence-eval-v23'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
design=read(C/'design.json');contract=read(C/'check-contract.json');source=R/'local/analysis/check_selector_train_decisions_cpu_20261019.py'
assert sha(source)==contract['checker_sha256']
deadline=datetime.fromisoformat(design['deadline_utc'])+timedelta(minutes=20)
while not (C/'exit.json').exists():
    if datetime.now(timezone.utc)>=deadline:raise TimeoutError('No closed CPU producer; no retry')
    time.sleep(10)
assert read(C/'exit.json')['exit_code']==0 and (C/'result.json').exists()
env=os.environ.copy();env.update(USE_TORCH_XLA='0',PYTHONPATH=str(W)+':'+str(R/'local/test-deps')+':'+str(R/'local/analysis'),OMP_NUM_THREADS='16',MKL_NUM_THREADS='16')
with (C/'independent-check.log').open('xb') as f:
    p=subprocess.run([sys.executable,'-u',str(source)],cwd=W,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
with (C/'independent-check-exit.json').open('x') as f:json.dump(dict(exit_code=p.returncode,source_sha256=sha(source),utc=datetime.now(timezone.utc).isoformat()),f,indent=2)
assert p.returncode==0
B=R/'local/backups/engineering/selector-TRAIN-decisions-CPU-v1/POST_RUN';B.mkdir(exist_ok=False)
files=[*[p for p in C.iterdir() if p.is_file()],R/'local/analysis/selector_train_decisions_cpu_20261019.py',source,Path(__file__).resolve()]
required=sum(p.stat().st_size for p in files);assert shutil.disk_usage(R).free>35*2**30+required+2**29
saved={}
for src in files:
    dst=B/src.name;assert not dst.exists();digest=sha(src);shutil.copyfile(src,dst)
    with dst.open('rb') as f:os.fsync(f.fileno())
    assert sha(src)==sha(dst)==digest;saved[src.name]=dict(bytes=dst.stat().st_size,sha256=digest)
fd=os.open(B,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='PASS_CPU_DIAGNOSTIC_ORIGINAL_OUTPUTS_AND_SOURCES_READBACK_FSYNC',utc=datetime.now(timezone.utc).isoformat(),files=saved,EBS=str(B),same_instance=True,accelerator_calls=0)
with (B/'ARCHIVE.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
receipt['archive_sha256']=sha(B/'ARCHIVE.json')
with (C/'protection.json').open('x') as f:json.dump(receipt,f,indent=2)
print(receipt['status'],flush=True)
