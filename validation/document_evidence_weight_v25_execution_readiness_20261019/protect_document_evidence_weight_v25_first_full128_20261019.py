"""Protect one independently audited complete128 recovery state as soon as available.

Never remove an original or model weight. Older inactive EBS optimizer duplicates
may retire only AFTER the newer full copy is verified and both old sides rehash.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,time
import torch
from train.partial_state_driver import ROOT,order_for
from weekend_thinking.pretrained import load_base
from document_evidence_weight_v25_state_audit_20261019 import read,sha,audit_checkpoint,independent_counts,audit_metrics
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-weight-v25';assert ROOT==W
C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1';plan=read(C/'plan.json');release=read(C/'release.json');assert release['plan_sha256']==sha(C/'plan.json')
P=C/'first-full128-protection';P.mkdir(exist_ok=False);deadline=datetime.fromisoformat(plan['deadline_utc'])
while True:
 candidates=[]
 for job in plan['jobs']:
  m=Path(job['output'])/'step-0000128/COMPLETE.json'
  if m.exists():candidates.append((read(m)['created_utc'],job['key'],job))
 if candidates:break
 if datetime.now(timezone.utc)>=deadline:raise TimeoutError('No checkpoint128 before declared deadline;no automaticretry')
 if all((C/job['key']/'failure.json').exists() for job in plan['jobs']):raise RuntimeError('All science jobsfailed before128;no backup fabricated')
 time.sleep(15)
job=max(candidates,key=lambda v:(v[0],v[1]))[2];p=read(job['protocol']);out=Path(job['output']);source=out/'step-0000128'
model,tok=load_base(p['model_dir'])
data=[json.loads(line) for line in Path(p['data_file']).open()]
counts=independent_counts(data,tok)
p,order,observed=audit_metrics(job,128,counts)
expected=observed[-1]['counters']
check=audit_checkpoint(source,p,job['arm'],out.name,order,job['protocol_sha256'],expected,model)
(P/'independent-state-audit.json').write_text(json.dumps(check,indent=2));del model;import gc;gc.collect()
B=R/'local/backups/science/document-evidence-weight-v25-ablation-v1/FIRST_FULL128';B.mkdir(parents=True,exist_ok=False)
m=read(source/'COMPLETE.json');required=sum(v['bytes'] for v in m['files'].values());assert shutil.disk_usage(R).free>35*2**30+required+2**30
files={}
for name,v in {**m['files'],'COMPLETE.json':dict(bytes=(source/'COMPLETE.json').stat().st_size,sha256=sha(source/'COMPLETE.json'))}.items():
 src,dst=source/name,B/name;assert src.stat().st_size==v['bytes'] and sha(src)==v['sha256'];shutil.copyfile(src,dst)
 with dst.open('rb') as f:os.fsync(f.fileno())
 assert dst.stat().st_size==v['bytes'] and sha(dst)==v['sha256'];files[name]=v
for name,src in [('protocol.json',Path(job['protocol'])),('independent-state-audit.json',P/'independent-state-audit.json'),('protection-source.py',Path(__file__))]:
 dst=B/name;shutil.copyfile(src,dst);assert sha(src)==sha(dst)
 with dst.open('rb') as f:os.fsync(f.fileno())
fd=os.open(B,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='COMPLETE128_FULL_READBACK_FSYNC_PROTECTED_BEFORE_OLD_DUPLICATE_RETIREMENT',utc=datetime.now(timezone.utc).isoformat(),model=job['key'],original=str(source),EBS=str(B),COMPLETE_sha256=sha(source/'COMPLETE.json'),files=files,audit_sha256=sha(P/'independent-state-audit.json'),root_free_bytes=shutil.disk_usage(R).free,limits='Same-instance,not off-instance. All original/modelweights retained.')
(B/'ARCHIVE.json').write_text(json.dumps(receipt,indent=2));(P/'new-full-readback.json').write_text(json.dumps(receipt,indent=2))
with (B/'ARCHIVE.json').open('rb') as f:os.fsync(f.fileno())
# The previous latest-full state remains complete until this point.
old=read(R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/terminal-protection.json');oldsource=Path(old['latest_full']['original']);oldbackup=Path(old['latest_full']['EBS']);oldm=read(oldsource/'COMPLETE.json')
assert sha(oldsource/'COMPLETE.json')==sha(oldbackup/'COMPLETE.json')==old['latest_full']['COMPLETE_sha256']
for name,v in oldm['files'].items():
 assert (oldsource/name).stat().st_size==(oldbackup/name).stat().st_size==v['bytes'] and sha(oldsource/name)==sha(oldbackup/name)==v['sha256']
paths=[oldbackup/f'rank-{rank:05d}.pt' for rank in range(8)];assert all(v.stat().st_nlink==1 for v in paths)
intent=dict(status='NEW_FULL128_PROTECTED_OLD_BOTH_SIDES_FRESHLY_REVERIFIED_BEFORE_EBS_OPTIMIZER_DUPLICATE_RETIREMENT',utc=datetime.now(timezone.utc).isoformat(),new_full_receipt_sha256=sha(B/'ARCHIVE.json'),old_original=str(oldsource),old_backup=str(oldbackup),old_files=oldm['files'],only_delete=[str(v) for v in paths])
(P/'retirement-intent.json').write_text(json.dumps(intent,indent=2))
with (P/'retirement-intent.json').open('rb') as f:os.fsync(f.fileno())
for path in paths:path.unlink()
assert all((oldsource/n).exists() for n in oldm['files']) and (oldbackup/'model.pt').exists() and all((B/n).exists() for n in m['files'])
retirement=dict(intent,status='ONLY_OLD_V23_EBS_OPTIMIZER_DUPLICATES_RETIRED_ALL_ORIGINALS_AND_MODELS_RETAINED',utc=datetime.now(timezone.utc).isoformat(),root_free_bytes=shutil.disk_usage(R).free)
(oldbackup/'OPTIMIZER_DUPLICATES_RETIRED.json').write_text(json.dumps(retirement,indent=2));(P/'old-optimizer-retirement.json').write_text(json.dumps(retirement,indent=2))
for directory in (oldbackup,P):
 fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt.update(status='PASS_FIRST128_INDEPENDENT_FULL_STATE_PROTECTED_PREVIOUS_DUPLICATES_RETIRED',old_optimizer_retirement_sha256=sha(P/'old-optimizer-retirement.json'),root_free_bytes=shutil.disk_usage(R).free)
(C/'first-full128-protection.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2),flush=True)
