"""Readback/fsync all complete pilot outputs and sources,without weights/secrets."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
F=R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(P/'independent-readout-check.json')['report_sha256']==sha(F)
B=R/'local/backups/evaluation/document-evidence-plan-v23-pilot-complete-v1';B.mkdir(parents=True,exist_ok=False)
files={}
def copy(src,relative):
 dst=B/relative;dst.parent.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(R).free>35*2**30+src.stat().st_size
 shutil.copyfile(src,dst)
 with dst.open('rb') as stream:os.fsync(stream.fileno())
 assert sha(src)==sha(dst);files[str(relative)]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
copy(F,Path('report.json'))
for path in P.iterdir():
 if path.is_file():copy(path,Path('public')/path.name)
for label in ['generation','likelihood']:
 for group in read(C/(label+'-registry-v2.json' if label=='likelihood' else label+'-registry.json'))['groups']:
  output=Path(group['output'])
  for path in output.rglob('*'):
   if path.is_file():copy(path,Path(label)/group['model']/path.relative_to(output))
for name in ['plan.json','release.json','prospective-design.json','readout-addendum.json','readout-execution-plan.json','readout-execution-plan-v2.json','likelihood-registry-v2.json','likelihood-v1-failure-preserved.json','fixed-example-selection.json','generation-execution-plan.json','likelihood-execution-plan.json','terminal-independent-audit.json','terminal-protection.json']:
 copy(C/name,Path('control')/name)
for name in read(C/'readout-execution-plan-v2.json')['source_files']:
 copy(R/name,Path('sources')/Path(name).name)
for directory in [p for p in B.rglob('*') if p.is_dir()]+[B]:
 fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='PASS_COMPLETE_PILOT_READBACK_FSYNC_ALL_NATIVE_AND_LL_ROWS',utc=datetime.now(timezone.utc).isoformat(),report_sha256=sha(F),archive=str(B),files=files,limits='Same-instance EBS,not off-instance;original full checkpoints remain NVMe,latestfull andterminalmodels protected separately.')
with (B/'ARCHIVE.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.flush();os.fsync(stream.fileno())
with (C/'readout-protection.json').open('x') as stream:json.dump(dict(status=receipt['status'],archive=str(B),receipt_sha256=sha(B/'ARCHIVE.json'),report_sha256=sha(F)),stream,indent=2)
for directory in [B,C]:
 fd=os.open(directory,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(receipt['status'])
