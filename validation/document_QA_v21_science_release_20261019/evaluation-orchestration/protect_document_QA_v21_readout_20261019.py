"""Protect every original execution plus independently checked public readout."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-QA-v21-study-v1';F=R/'validation/document_QA_v21_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(F);check=read(P/'independent-readout-check.json');assert check['report_sha256']==sha(F) and check['status']=='PASS_NATIVE_ACTIONS_FIELDS_CANONICAL_ALIAS_METRICS_VECTORS_BOOTSTRAPS_AND_GATES'
S=P/'analysis-source';S.mkdir(exist_ok=False)
for name in ['analyze_document_QA_v21_20261019.py','check_document_QA_v21_readout_20261019.py','render_document_QA_v21_20261019.py','protect_document_QA_v21_readout_20261019.py','watch_document_QA_v21_readout_20261019.py','document_QA_v21_likelihood_job_20261019.py','supervise_document_QA_v21_likelihood_group_20261019.py','watch_document_QA_v21_likelihood_20261019.py']:
 shutil.copyfile(R/'local/analysis'/name,S/name)
assert sha(S/'analyze_document_QA_v21_20261019.py')==r['analyzer_sha256'] and sha(S/'check_document_QA_v21_readout_20261019.py')==check['checker_sha256']
A=R/'local/backups/evaluation/document-QA-v21-complete';A.mkdir(parents=True,exist_ok=False);files={}
def copy(source,target):
 target.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(R).free>35*2**30+source.stat().st_size
 shutil.copyfile(source,target)
 with target.open('rb') as f:os.fsync(f.fileno())
 assert sha(source)==sha(target);files[str(target.relative_to(A))]=dict(bytes=source.stat().st_size,sha256=sha(source),original=str(source))
roots={}
for g in r['generation_inventory']:roots['generation/'+Path(g['output']).name]=Path(g['output'])
for g in read(C/'likelihood-registry.json')['groups']:roots['likelihood/'+g['model']]=Path(g['output'])
roots['control']=C
for label,root in roots.items():
 for source in sorted(root.rglob('*')):
  if source.is_file():copy(source,A/label/source.relative_to(root))
for source in [F,*[p for p in P.rglob('*') if p.is_file()]]:copy(source,A/'PUBLIC-READOUT'/source.relative_to(R/'validation'))
receipt=dict(status='ALL_ORIGINAL_GENERATION_LIKELIHOOD_CONTROLS_PUBLIC_READOUT_AND_ANALYSIS_READBACK_PROTECTED',utc=datetime.now(timezone.utc).isoformat(),files=files,report_sha256=sha(F),checker_sha256=sha(P/'independent-readout-check.json'),limits='Same-instanceEBS readback/fsync,not off-instance. All originals retained. Fullmodel/optimizer protection is in separate terminal protection receipts.')
(A/'READBACK.json').write_text(json.dumps(receipt,indent=2))
for folder in sorted([A,*[p for p in A.rglob('*') if p.is_dir()]],key=lambda p:-len(p.parts)):
 fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
with (A/'READBACK.json').open('rb') as f:os.fsync(f.fileno())
(P/'readout-protection.json').write_text(json.dumps(dict(status=receipt['status'],archive=str(A),files=len(files),receipt_sha256=sha(A/'READBACK.json'),report_sha256=sha(F),limits=receipt['limits']),indent=2))
print(json.dumps(dict(status=receipt['status'],files=len(files)),indent=2),flush=True)
