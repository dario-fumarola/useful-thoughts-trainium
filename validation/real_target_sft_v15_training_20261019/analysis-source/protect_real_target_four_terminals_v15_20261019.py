"""Protect all scientific models and one latest full terminal recovery state."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os
from real_target_state_audit_v15_20261019 import read,sha
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-target-sft-v15-study-v1'
audit_file=C/'terminal-independent-audit.json';audit=read(audit_file);plan=read(C/'plan.json')
assert audit['status']=='PASS_ALL_FOUR_TERMINAL320_TRAINING_STATES_NATIVE32_ORDER_AND_COUNTERS'
assert audit['plan_sha256']==sha(C/'plan.json')
A=R/'local/backups/science/real-target-sft-v15-study-v1/TERMINALS';A.mkdir(exist_ok=False)
manifest_by_model={(x['arm'],x['seed']):x['checkpoints']['320'] for x in audit['models']}
latest=max(plan['jobs'],key=lambda j:read(Path(j['output'])/'step-0000320/COMPLETE.json')['created_utc'])
need=sum((Path(j['output'])/'step-0000320/model.pt').stat().st_size for j in plan['jobs'])
need+=sum(v['bytes'] for n,v in read(Path(latest['output'])/'step-0000320/COMPLETE.json')['files'].items() if n.startswith('rank-'))
assert shutil.disk_usage(R).free>35*2**30+need
files={};protected={}
def copy(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 with dest.open('rb') as f:os.fsync(f.fileno())
 assert sha(source)==sha(dest)
 files[str(dest.relative_to(A))]=dict(bytes=dest.stat().st_size,sha256=sha(dest))
for job in plan['jobs']:
 key=(job['arm'],job['seed']);source=Path(job['output'])/'step-0000320';d=A/'models'/f'{key[0]}-seed{key[1]}'
 assert sha(source/'COMPLETE.json')==manifest_by_model[key]['COMPLETE_sha256']
 assert sha(source/'model.pt')==manifest_by_model[key]['model_sha256']
 for name in ('model.pt','COMPLETE.json'):copy(source/name,d/name)
 copy(Path(job['protocol']),d/'protocol.json')
 protected[f'{key[0]}-seed{key[1]}']=dict(original=str(source/'model.pt'),EBS=str(d/'model.pt'),sha256=sha(d/'model.pt'))
 print('Terminal model protected',key,flush=True)
source=Path(latest['output'])/'step-0000320';d=A/'latest-full320';d.mkdir()
source_model=A/'models'/f'{latest["arm"]}-seed{latest["seed"]}'/'model.pt'
os.link(source_model,d/'model.pt');assert sha(d/'model.pt')==sha(source/'model.pt')
files[str((d/'model.pt').relative_to(A))]=dict(bytes=(d/'model.pt').stat().st_size,sha256=sha(d/'model.pt'),hardlink_of=str(source_model.relative_to(A)))
for name,v in read(source/'COMPLETE.json')['files'].items():
 if name.startswith('rank-'):copy(source/name,d/name)
copy(source/'COMPLETE.json',d/'COMPLETE.json')
copy(audit_file,A/'terminal-independent-audit.json')
copy(C/'plan.json',A/'plan.json');copy(C/'release.json',A/'release.json')
copy(Path(__file__),A/'protect-source.py')
receipt=dict(status='PASS_ALL_FOUR_TERMINAL_MODELS_AND_LATEST_FULL320_PROTECTED',utc=datetime.now(timezone.utc).isoformat(),
 models=protected,latest_full=dict(original=str(source),EBS=str(d),COMPLETE_sha256=sha(d/'COMPLETE.json')),
 audit_sha256=sha(audit_file),files=files,
 limits=['Alloriginalfull160/320statesremain. EBSretainsallterminalmodelsandlatestfulloptimizer;modelhardlinkiswithinonepersistentfilesystem.','Same-instancebackup;notanoff-instancearchive.'])
with (A/'ARCHIVE.json').open('x') as f:json.dump(receipt,f,indent=2)
with (A/'ARCHIVE.json').open('rb') as f:os.fsync(f.fileno())
for folder in [*[p for p in A.rglob('*') if p.is_dir()],A]:
 fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt['EBS_archive_sha256']=sha(A/'ARCHIVE.json')
with (C/'terminal-protection.json').open('x') as f:json.dump(receipt,f,indent=2)
with (C/'terminal-protection.json').open('rb') as f:os.fsync(f.fileno())
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2),flush=True)
