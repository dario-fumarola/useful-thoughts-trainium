"""Relocate one inactive intermediate-model BACKUP duplicate; preserve originals.

The latest full science recovery state and all six EBS terminal models remain
untouched. Copy/readback/fsync on the other existing NVMe precedes unlinking
only the old, single-linked EBS intermediate duplicate. No original is moved.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/home/ubuntu/dario')
C=R/'local/diagnostics/document-evidence-plan-v23-storage-preparation-v1'
C.mkdir(exist_ok=False)
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with Path(p).open('x') as s:json.dump(v,s,indent=2);s.flush();os.fsync(s.fileno())
def syncdir(p):
 fd=os.open(p,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
r=read(R/'local/diagnostics/document-QA-v21-study-v1/first-full128-protection.json')
source=Path(r['original'])/'model.pt'
old=Path(r['EBS'])/'model.pt'
assert old==R/'local/backups/science/document-QA-v21-study-v1/FIRST_FULL128/model.pt'
expected=r['files']['model.pt']
assert old.is_file() and not old.is_symlink() and old.stat().st_nlink==1
assert source.is_file() and source.stat().st_dev!=old.stat().st_dev
assert source.stat().st_size==old.stat().st_size==expected['bytes']
assert sha(source)==sha(old)==expected['sha256']
latest=R/'local/backups/science/document-QA-v21-study-v1/TERMINALS/latest-full256'
lm=read(latest/'COMPLETE.json')
assert all((latest/n).is_file() for n in lm['files'])
terminals=R/'local/backups/science/document-QA-v21-study-v1/TERMINALS/models'
assert len(list(terminals.glob('*/model.pt')))==6
volume=R/'local/nvme-study'
assert volume.is_mount() and shutil.disk_usage(volume).free>35*2**30+expected['bytes']
dest=volume/'backups/document-QA-v21-inactive128-model-duplicate-v1/model.pt'
dest.parent.mkdir(parents=True,exist_ok=False)
shutil.copyfile(old,dest)
with dest.open('rb') as stream:os.fsync(stream.fileno())
syncdir(dest.parent)
assert dest.stat().st_size==expected['bytes'] and sha(dest)==sha(source)==sha(old)==expected['sha256']
receipt=dict(status='THREE_COPIES_HASHED_BEFORE_INACTIVE_EBS_DUPLICATE_RELOCATION',
 utc=datetime.now(timezone.utc).isoformat(),original=str(source),former_backup=str(old),
 relocated_backup=str(dest),bytes=expected['bytes'],sha256=expected['sha256'],
 latest_full_EBS_untouched=str(latest),six_EBS_terminal_models_untouched=True,
 scope='One inactive V21 step128 model backup duplicate only. Original full state remains on NVMe-cpt;second model copy is on NVMe-study. Same-instance storage,not off-instance.',
 root_free_before_bytes=shutil.disk_usage(R).free)
write(C/'intent.json',receipt)
old.unlink();syncdir(old.parent)
assert source.is_file() and dest.is_file()
receipt.update(status='PASS_INACTIVE_INTERMEDIATE_DUPLICATE_RELOCATED_ORIGINAL_AND_TERMINALS_UNCHANGED',
 root_free_after_bytes=shutil.disk_usage(R).free,utc=datetime.now(timezone.utc).isoformat())
write(old.parent/'MODEL_BACKUP_RELOCATED.json',receipt)
write(C/'completion.json',receipt)
print(json.dumps(receipt,indent=2))
