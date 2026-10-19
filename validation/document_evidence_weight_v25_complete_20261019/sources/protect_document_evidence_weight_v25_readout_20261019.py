"""Protect original closed output/source bytes on EBS before public readout."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import os
import shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
A=R/'local/backups/science/document-evidence-weight-v25-ablation-v1/COMPLETE_READOUT'
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
assert read(C/'complete-readout-before-protection.json')['status']=='COMPLETE_FIXED_V25_READOUT_PENDING_ORIGINAL_ARCHIVE'
assert read(C/'terminal-pipeline-completion.json')['status']=='TERMINAL_AUDITED_PROTECTED_BOUND_NO_EVALUATION_LAUNCHED'
assert all(read(C/path)['assigned_physical_device_empty'] for path in ('TRAIN-fit/GEN/closure.json','HELD-native/GEN/closure.json','HELD-replay/GEN/closure.json','ORDINARY/GEN/closure.json'))
outputs=[R/'local/nvme-cpt/evaluation'/name for name in (
 'document-evidence-weight-v25-TRAIN256-v1','document-evidence-weight-v25-held255-native-v1',
 'document-evidence-weight-v25-held255-answer-replay-v1','document-evidence-weight-v25-ordinary256-v1')]
training=R/'local/nvme-cpt/runs/document-evidence-weight-v25-selection1-TRAIN2048-seed2026101923-v1'
files={str(p.relative_to(R)):p for root in [C,*outputs] for p in root.rglob('*') if p.is_file()}
for p in training.iterdir():
 if p.is_file():files[str(p.relative_to(R))]=p
source_names=set(read(C/'release.json')['pipeline_sources'])
source_names.update(str(p.relative_to(R)) for p in (R/'local/analysis').glob('*document_evidence_weight_v25*20261019.py'))
for name in source_names:files[name]=R/name
# Preserve fixed public contracts and source annexes, not the entire historical repository.
for p in (R/'validation').glob('document_evidence_weight_v25*'):
 for q in ([p] if p.is_file() else p.rglob('*')):
  if q.is_file():files[str(q.relative_to(R))]=q
assert all(not p.is_symlink() for p in files.values())
required=sum(p.stat().st_size for p in files.values())
assert required<2*2**30 and shutil.disk_usage(R).free>35*2**30+required+2**29
A.mkdir(parents=True,exist_ok=False);saved={}
for name,source in sorted(files.items()):
 target=A/name;target.parent.mkdir(parents=True,exist_ok=True)
 before=sha(source);shutil.copyfile(source,target)
 with target.open('rb') as f:os.fsync(f.fileno())
 assert sha(source)==sha(target)==before
 saved[name]=dict(bytes=target.stat().st_size,sha256=before)
for d in [*[p for p in A.rglob('*') if p.is_dir()],A]:
 fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='PASS_ORIGINAL_CLOSED_V25_OUTPUTS_SOURCES_AND_READOUT_READBACK_FSYNC',utc=datetime.now(timezone.utc).isoformat(),
 EBS=str(A),files=saved,total_bytes=sum(v['bytes'] for v in saved.values()),files_count=len(saved),
 limits='Same-instance EBS, not off-instance. Original outputs and all original full checkpoints remain; terminal/latest-full weights already separately protected.')
with (A/'ARCHIVE.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
receipt['EBS_archive_sha256']=sha(A/'ARCHIVE.json')
with (C/'readout-protection.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
for d in (A,C):
 fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
