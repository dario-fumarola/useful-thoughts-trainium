"""Seal the original pre-held calibration-write failure and physical closure."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,shutil,subprocess
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
helper=R/'local/diagnostics/document-evidence-plan-v23-world8-qualification-v1/qualify.py'
spec=importlib.util.spec_from_file_location('failure_ownership_helpers',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
top=h.topology();assert all(not d['neuron_processes'] for d in top if d['neuron_device'] in [6,7])
B=R/'local/backups/evaluation/document-evidence-plan-v23-likelihood-v1-failure';B.mkdir(parents=True,exist_ok=False)
files={};witnesses=[]
def copy(src,relative):
 dst=B/relative;dst.parent.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(R).free>35*2**30+src.stat().st_size
 shutil.copyfile(src,dst)
 with dst.open('rb') as stream:os.fsync(stream.fileno())
 assert sha(src)==sha(dst);files[str(relative)]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
for g in read(C/'likelihood-registry.json')['groups']:
 O,control=Path(g['output']),Path(g['control'])
 for shard in range(4):
  d=O/'scores'/f'shard-{shard}';assert not (d/'scores.jsonl').exists()
  launch=read(control/f'shard-{shard}/launch.json');assert not h.members(launch['identity']['session'])
  partial=read(d/'own-model-TRAIN-calibration.json');assert set(partial)=={'QA'} and partial['QA']['passed']
  witnesses.append(dict(model=g['model'],shard=shard,QA_passed=True,ordinary_result_not_persisted=True,held_score_file_absent=True,all_owned_session_processes_absent=True))
 for label,folder in [('original-output',O),('original-control',control)]:
  for src in folder.rglob('*'):
   if src.is_file():copy(src,Path(label)/g['model']/src.relative_to(folder))
for name in ['likelihood-execution-plan.json','likelihood-registry.json','likelihood-watcher.log','readout-watcher.log','readout-execution-plan.json']:
 copy(C/name,Path('control')/name)
for src in (C/'likelihood-pipeline').iterdir():
 if src.is_file():copy(src,Path('original-pipeline')/src.name)
copy(R/'local/worktrees/document-evidence-span-v23/train/document_evidence_plan_likelihood.py',Path('original-worker.py'))
receipt=dict(status='PRESERVED_ORIGINAL_CALIBRATION_WRITE_FAILURE_ZERO_HELD_ROWS_PHYSICALLY_CLOSED',utc=datetime.now(timezone.utc).isoformat(),
 cause='FileExistsError when second TRAIN panel republishes cumulative calibration JSON with no-clobber atomic_json. QA passes;ordinary result unrecorded;source stops before anyheld loop.',
 witnesses=witnesses,assigned_physical6_7_empty=True,files=files,
 no_scientific_outcome_inspection=True,models_and_all1530_native_attempts_unchanged=True,
 limits='Original failed outputs/source/logs retained,not an automatic retry. Explicitrootreview/sourceIOrepair required before new attempt. No numerical/scientificgate changes.')
with (B/'ARCHIVE.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.flush();os.fsync(stream.fileno())
for d in [p for p in B.rglob('*') if p.is_dir()]+[B]:
 fd=os.open(d,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
with (C/'likelihood-v1-failure-preserved.json').open('x') as stream:json.dump(dict(status=receipt['status'],archive=str(B),archive_sha256=sha(B/'ARCHIVE.json'),cause=receipt['cause']),stream,indent=2)
print(receipt['status'])
