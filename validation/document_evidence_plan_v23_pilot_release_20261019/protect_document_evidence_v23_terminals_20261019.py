"""Protect new full state first;only then retire older optimizer backup duplicates."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,os
from real_source_state_audit_v17_20261019 import read,sha
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
audit_file=C/'terminal-independent-audit.json';audit=read(audit_file);plan=read(C/'plan.json')
assert audit['status']=='PASS_BOTH_TERMINAL256_EVIDENCE_TRAINING_STATES_NATIVE16_ORDER_COUNTERS_AND_LABELS'
assert audit['plan_sha256']==sha(C/'plan.json')
A=R/'local/backups/science/document-evidence-plan-v23-pilot-v1/TERMINALS';A.mkdir(exist_ok=False)
manifest_by_model={(x['arm'],x['seed']):x['checkpoints']['256'] for x in audit['models']}
latest=max(plan['jobs'],key=lambda j:read(Path(j['output'])/'step-0000256/COMPLETE.json')['created_utc'])
source=Path(latest['output'])/'step-0000256';full_manifest=read(source/'COMPLETE.json')
full_bytes=sum(v['bytes'] for v in full_manifest['files'].values())
assert shutil.disk_usage(R).free>35*2**30+full_bytes+2**29
files={};protected={}
def syncdir(folder):
 fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def copy(source,dest):
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 with dest.open('rb') as f:os.fsync(f.fileno())
 assert sha(source)==sha(dest)
 files[str(dest.relative_to(A))]=dict(bytes=dest.stat().st_size,sha256=sha(dest))
 assert shutil.disk_usage(R).free>35*2**30

def write(path,v):
 with path.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 syncdir(path.parent)

# The new full recovery state is protected before any old optimizer copy changes.
d=A/'latest-full256';d.mkdir()
for name,value in full_manifest['files'].items():
 assert sha(source/name)==value['sha256'];copy(source/name,d/name)
copy(source/'COMPLETE.json',d/'COMPLETE.json');syncdir(d)
for name,value in full_manifest['files'].items():assert sha(d/name)==sha(source/name)==value['sha256']
write(A/'NEW_FULL_PROTECTED.json',dict(status='NEW_FULL256_ALL_PAYLOADS_READBACK_FSYNC_BEFORE_RETIREMENT',utc=datetime.now(timezone.utc).isoformat(),original=str(source),EBS=str(d),COMPLETE_sha256=sha(d/'COMPLETE.json')))
# Fresh full readback of the older original AND EBS duplicate immediately precedes retirement.
oldreceipt=C/'first-full128-protection.json';old=read(oldreceipt)
assert old['status']=='PASS_FIRST128_INDEPENDENT_FULL_STATE_PROTECTED_PREVIOUS_DUPLICATES_RETIRED'
osource,obackup=map(Path,[old['original'],old['EBS']])
assert obackup==R/'local/backups/science/document-evidence-plan-v23-pilot-v1/FIRST_FULL128'
assert osource.is_dir() and obackup.is_dir() and osource.stat().st_dev!=obackup.stat().st_dev
om=read(osource/'COMPLETE.json')
assert om['step']==128 and om['world_size']==8 and set(om['files'])=={'model.pt',*[f'rank-{i:05d}.pt' for i in range(8)]}
assert sha(osource/'COMPLETE.json')==sha(obackup/'COMPLETE.json')==old['COMPLETE_sha256']
assert datetime.fromisoformat(full_manifest['created_utc'])>datetime.fromisoformat(om['created_utc'])
for name,value in om['files'].items():
 for root in (osource,obackup):
  assert not (root/name).is_symlink() and (root/name).stat().st_size==value['bytes'] and sha(root/name)==value['sha256']
only=[obackup/f'rank-{i:05d}.pt' for i in range(8)]
assert all(path.parent==obackup and path.stat().st_nlink==1 for path in only)
retirement=dict(status='FRESH_BOTH_SIDES_OLD_AND_NEW_FULL_VERIFIED_BEFORE_RETIREMENT',utc=datetime.now(timezone.utc).isoformat(),authorization='Existinglatest-fullEBSretentionpolicy;alloriginals/allmodelweightsretained.',old_original=str(osource),old_backup=str(obackup),old_files=om['files'],new_original=str(source),new_backup=str(d),new_COMPLETE_sha256=sha(d/'COMPLETE.json'),only_delete=[str(path) for path in only])
write(A/'OLD_OPTIMIZER_RETIREMENT_INTENT.json',retirement)
for path in only:path.unlink()
assert (obackup/'model.pt').is_file() and all((osource/name).is_file() for name in om['files'])
retirement.update(status='ONLY_OLDER_EVIDENCE128_EBS_OPTIMIZER_DUPLICATES_RETIRED',utc=datetime.now(timezone.utc).isoformat(),limits='OldhistoricalCOMPLETE/ARCHIVEdescribeformerfullcopy;originalfullNVMeunchanged. NewfullEBSverified. Same-instancebackup.')
write(obackup/'OPTIMIZER_DUPLICATES_RETIRED.json',retirement);write(A/'OLD_OPTIMIZER_RETIREMENT_COMPLETE.json',retirement)
# Copy all terminal models;reuse the new full state's model through an EBS hard link.
remaining=sum((Path(j['output'])/'step-0000256/model.pt').stat().st_size for j in plan['jobs'] if j!=latest)
assert shutil.disk_usage(R).free>35*2**30+remaining+2**29
for job in plan['jobs']:
 key=(job['arm'],job['seed']);origin=Path(job['output'])/'step-0000256';folder=A/'models'/f'{key[0]}-seed{key[1]}';folder.mkdir(parents=True)
 assert sha(origin/'COMPLETE.json')==manifest_by_model[key]['COMPLETE_sha256']
 assert sha(origin/'model.pt')==manifest_by_model[key]['model_sha256']
 if job==latest:
  os.link(d/'model.pt',folder/'model.pt');assert sha(folder/'model.pt')==sha(origin/'model.pt')
  files[str((folder/'model.pt').relative_to(A))]=dict(bytes=(folder/'model.pt').stat().st_size,sha256=sha(folder/'model.pt'),hardlink_of='latest-full256/model.pt')
 else:copy(origin/'model.pt',folder/'model.pt')
 copy(origin/'COMPLETE.json',folder/'COMPLETE.json');copy(Path(job['protocol']),folder/'protocol.json')
 protected[f'{key[0]}-seed{key[1]}']=dict(original=str(origin/'model.pt'),EBS=str(folder/'model.pt'),sha256=sha(folder/'model.pt'))
 print('Terminalmodelprotected',key,flush=True)
for name in ('terminal-independent-audit.json','plan.json','release.json'):copy(C/name,A/name)
copy(Path(__file__),A/'protect-source.py')
receipt=dict(status='PASS_BOTH_TERMINAL_MODELS_AND_LATEST_FULL256_PROTECTED',utc=datetime.now(timezone.utc).isoformat(),models=protected,latest_full=dict(original=str(source),EBS=str(d),COMPLETE_sha256=sha(d/'COMPLETE.json')),audit_sha256=sha(audit_file),files=files,old_optimizer_retirement_sha256=sha(A/'OLD_OPTIMIZER_RETIREMENT_COMPLETE.json'),root_free_bytes=shutil.disk_usage(R).free,limits=['Alloriginalfull128/256statesremainNVMe. EBSretainsallterminalmodels/latestglobalfulloptimizer;the earlier protected128modelweight also remains.','Same-instanceprotection;notanoff-instancearchive.'])
write(A/'ARCHIVE.json',receipt)
for folder in [*[p for p in A.rglob('*') if p.is_dir()],A]:syncdir(folder)
receipt['EBS_archive_sha256']=sha(A/'ARCHIVE.json');write(C/'terminal-protection.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2),flush=True)
