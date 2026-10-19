"""Publish the already checked/protected V25 readout, retaining all old results."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-weight-v25-ablation-v1'
F=R/'validation/document_evidence_weight_v25_complete_20261019.json';P=F.with_suffix('')
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
def rows(p):return [json.loads(v) for v in p.read_text().splitlines()]
protection=read(C/'readout-protection.json');assert protection['status']=='PASS_ORIGINAL_CLOSED_V25_OUTPUTS_SOURCES_AND_READOUT_READBACK_FSYNC'
assert sha(Path(protection['EBS'])/'ARCHIVE.json')==protection['EBS_archive_sha256']
for filename,digest in read(C/'postprocessing-sources.json')['files'].items():assert sha(R/filename)==digest
report=read(C/'complete-readout-before-protection.json')
assert report['status']=='COMPLETE_FIXED_V25_READOUT_PENDING_ORIGINAL_ARCHIVE' and report['primary_decision']=='FAIL'
assert report['primary_TRAIN_fitting']['strata']['3']['exact']==64
assert read(C/'operational-control-check.json')['status']=='PASS_CLOSED_FIELD_OMISSION_AND_DECLARED_EMPTY_FALLBACK_CHECK'
assert sha(R/'validation/document_evidence_plan_v23_pilot_complete_20261019.json')=='ca7cc615495d0cdade77ef70cce0f19055e3456b1c79c460a7f84008dec809b3'
assert sha(R/'validation/document_selector_fit_v24_complete_20261019.json')=='b98ca968835f5f93232ed018e336dfb42f338146cef34fc05de450177212bcb7'
assert not F.exists() and not P.exists()
P.mkdir();files={}
def record(path,original=None):
 files[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
 if original is not None:files[path.name]['original_sha256']=sha(original)
def copy(source,name=None):
 target=P/(name or source.name);shutil.copyfile(source,target);assert sha(source)==sha(target);record(target,source)
def compressed(source,name):
 target=P/name;target.write_bytes(gzip.compress(source.read_bytes(),mtime=0));assert gzip.decompress(target.read_bytes())==source.read_bytes();record(target,source)
for path in [C/'TRAIN-fit/result.json',C/'TRAIN-fit/independent-check.json',C/'held-native-and-replay-result.json',C/'held-independent-check.json',C/'ORDINARY/result.json',C/'ORDINARY/independent-check.json',C/'held-selection-descriptive.json',C/'operational-control-check.json',C/'actual-cost.json',C/'fixed-eight-examples.json',C/'terminal-independent-audit.json',C/'terminal-protection.json',C/'postprocessing-sources.json']:
 name=path.name
 if path.parent.name in ('TRAIN-fit','ORDINARY'):name=path.parent.name+'-'+name
 copy(path,name)
compressed(C/'TRAIN-fit/native-and-scored.jsonl','TRAIN-native-and-scored.jsonl.gz')
compressed(C/'held-native-and-replay-scored.json','held-native-and-replay-scored.json.gz')
compressed(R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2/held255-gold-only.jsonl','fixed-held-gold-annotations.jsonl.gz')
copy(C/'HELD-replay/replay-selection-metadata.json')
for name in ('TRAIN256','held255-native','held255-answer-replay','ordinary256'):
 output=R/f'local/nvme-cpt/evaluation/document-evidence-weight-v25-{name}-v1';copy(output/'protocol.json',name+'-protocol.json')
for label,folder in [('TRAIN','TRAIN-fit'),('held-native','HELD-native'),('held-replay','HELD-replay'),('ordinary','ORDINARY')]:copy(C/folder/'GEN/closure.json',label+'-physical-closure.json')
# A compact self-contained ordinary-score table, preserving original values.
old=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
paths={g['arm']:Path(g['output']) for g in read(old/'likelihood-registry-v2.json')['groups']}
paths['B0']=Path(read(old/'readout-execution-plan-v2.json')['B0_ordinary_reference']['output'])
paths['weight1']=R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-ordinary256-v1'
ordinary=[]
for name,path in sorted(paths.items()):
 subset=[dict(model=name,**v) for shard in range(4) for v in rows(path/'scores'/f'shard-{shard}'/'scores.jsonl') if v['panel']=='ordinary'];assert len(subset)==256;ordinary.extend(subset)
raw=''.join(json.dumps(v)+'\n' for v in ordinary).encode();target=P/'ordinary-all-model-token-scores.jsonl.gz';target.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(target.read_bytes())==raw;record(target)
for source in (C/'figures').iterdir():copy(source)
# Original current-model numeric witnesses, without republishing model weights.
for name in ('TRAIN256','held255-native','held255-answer-replay'):
 output=R/f'local/nvme-cpt/evaluation/document-evidence-weight-v25-{name}-v1'
 for shard in range(4):
  for file in ('native.json','cache-prefill.json'):copy(output/'answers'/f'shard-{shard}'/file,f'{name}-shard{shard}-{file}')
for shard in range(4):copy(paths['weight1']/'scores'/f'shard-{shard}'/'own-model-TRAIN-calibration.json',f'ordinary-shard{shard}-calibration.json')
sources=P/'sources';sources.mkdir()
source_files=set(read(C/'release.json')['pipeline_sources'])|set(read(C/'postprocessing-sources.json')['files'])
source_files.update(['local/analysis/document_evidence_weight_v25_ordinary_20261019.py','local/analysis/supervise_document_evidence_weight_v25_ordinary_20261019.py',str(Path(__file__).relative_to(R))])
source_payloads={}
for filename in sorted(source_files):
 src=R/filename;target=sources/src.name;shutil.copyfile(src,target);assert sha(src)==sha(target);source_payloads[filename]=dict(public='sources/'+target.name,sha256=sha(target))
report.update(status='COMPLETE_SINGLE_COEFFICIENT_ABLATION_FIXED_READOUT',publication_utc=datetime.now(timezone.utc).isoformat(),
 interpretation=dict(primary_fitting='FAIL: three-support64/96 versus59/96, below90% and paired interval crosses zero.',
  developmental_recipe='Native143/255 versus original matched-direct127/255: +6.3pp[1.6,11.4], one previously inspected cohort/one data-order seed.',
  incremental_inference='Native143 versus same-model direct138: +2.0pp[-1.6,5.5]. Closed-field own143 versus empty139: +1.6pp[-1.6,4.7]. Reliable emitted-ID benefit is not established.',
  direct_training_transfer='Direct138 versus original matched direct127: +4.3pp[-0.4,9.4], not established by this conditional interval.',
  ordinary_cost='NLL+0.0666 nats/token versus B0; -0.0687 versus original0.1 selector. Smaller regression, not broad pretraining improvement.',
  baseline_limit='Source/final-answer coefficients match, but the historical direct control retains auxiliary weight0.1. A format-only1.0 control has not been tested.',
  decision='Close the failed coefficient-only fitting repair. Preserve the better developmental recipe tradeoff without claiming a reliable inference channel.',
  next='Prepare a bounded format-only control with the same auxiliary1.0 coefficient before capacity/data scaling or fresh confirmation.'),
 operational_control_check=read(C/'operational-control-check.json'),
 protected_originals={k:v for k,v in protection.items() if k!='files'},
 reused_TRAIN_input_files={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [R/'validation/document_selector_fit_v24_complete_20261019/TRAIN256-labels.jsonl.gz',R/'validation/document_selector_fit_v24_complete_20261019/TRAIN256-questions-only.jsonl.gz']},
 payloads=files,source_payloads=source_payloads,
 preservation='All original V23/V24 reports and all earlier scores/gates unchanged. All original full128/256 states retained; terminal model/latest full EBS separately protected. No new training or scale-up release.')
with F.open('x') as f:json.dump(report,f,indent=2)
# Protect exactly the publication after original raw readout protection.
B=R/'local/backups/science/document-evidence-weight-v25-ablation-v1/PUBLICATION';B.mkdir(exist_ok=False)
protected={}
for src in [F,*[p for p in P.rglob('*') if p.is_file()]]:
 relative=src.relative_to(R/'validation');target=B/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
 with target.open('rb') as f:os.fsync(f.fileno())
 assert sha(src)==sha(target);protected[str(relative)]=dict(bytes=target.stat().st_size,sha256=sha(target))
for folder in [*[p for p in B.rglob('*') if p.is_dir()],B]:
 fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='PASS_FINAL_V25_PUBLICATION_READBACK_FSYNC',utc=datetime.now(timezone.utc).isoformat(),files=protected,EBS=str(B),report_sha256=sha(F),limits='Same-instance backup.')
with (C/'publication-protection.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status=report['status'],primary_fitting=report['primary_decision'],report=str(F),report_sha256=sha(F),files=len(files),source_files=len(source_payloads),protected_public_files=len(protected))))
