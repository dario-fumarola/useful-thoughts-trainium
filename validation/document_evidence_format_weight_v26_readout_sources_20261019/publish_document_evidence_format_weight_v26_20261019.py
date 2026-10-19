"""Publish only the checked/protected control, without rewriting historical claims."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1'
stamp=datetime.now(timezone.utc).strftime('%Y%m%d')
F=R/f'validation/document_evidence_format_weight_v26_complete_{stamp}.json';P=F.with_suffix('')
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(v) for v in p.read_text().splitlines()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
protection=read(C/'readout-protection.json')
assert protection['status']=='PASS_ORIGINAL_CLOSED_V26_OUTPUTS_SOURCES_AND_READOUT_READBACK_FSYNC'
assert sha(Path(protection['EBS'])/'ARCHIVE.json')==protection['EBS_archive_sha256']
for filename,digest in read(C/'postprocessing-sources.json')['files'].items():assert sha(R/filename)==digest
report=read(C/'complete-readout-before-protection.json')
assert report['status']=='COMPLETE_FIXED_V26_READOUT_PENDING_ORIGINAL_ARCHIVE'
assert not F.exists() and not P.exists();P.mkdir();files={}
def record(p,original=None):
    files[p.name]=dict(bytes=p.stat().st_size,sha256=sha(p))
    if original is not None:files[p.name]['original_sha256']=sha(original)
def copy(src,name=None):
    dst=P/(name or src.name);shutil.copyfile(src,dst);assert sha(src)==sha(dst);record(dst,src)
def compressed(src,name):
    dst=P/name;dst.write_bytes(gzip.compress(src.read_bytes(),mtime=0));assert gzip.decompress(dst.read_bytes())==src.read_bytes();record(dst,src)
for path in [C/'held-result.json',C/'held-independent-check.json',C/'ORDINARY/result.json',C/'ORDINARY/independent-check.json',C/'actual-cost.json',C/'fixed-eight-examples.json',C/'terminal-independent-audit.json',C/'terminal-protection.json',C/'postprocessing-sources.json']:
    copy(path,'ORDINARY-'+path.name if path.parent.name=='ORDINARY' else path.name)
compressed(C/'held-scored.json','held-all-comparison-native-and-scored.json.gz')
compressed(R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2/held255-gold-only.jsonl','fixed-held-gold-annotations.jsonl.gz')
for label,path in [('held','HELD'),('ordinary','ORDINARY')]:
    output=R/f'local/nvme-cpt/evaluation/document-evidence-format-weight-v26-{"held255" if label=="held" else "ordinary256"}-v1'
    copy(output/'protocol.json',label+'-protocol.json');copy(C/path/'GEN/closure.json',label+'-physical-closure.json')
    if label=='held':
        for shard in range(4):
            for name in ('native.json','cache-prefill.json'):copy(output/'answers'/f'shard-{shard}'/name,f'held-shard{shard}-{name}')
    else:
        for shard in range(4):copy(output/'scores'/f'shard-{shard}'/'own-model-TRAIN-calibration.json',f'ordinary-shard{shard}-calibration.json')
old=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
paths={v['arm']:Path(v['output']) for v in read(old/'likelihood-registry-v2.json')['groups']}
paths['B0']=Path(read(old/'readout-execution-plan-v2.json')['B0_ordinary_reference']['output'])
paths['selector1']=R/'local/nvme-cpt/evaluation/document-evidence-weight-v25-ordinary256-v1'
paths['format1']=R/'local/nvme-cpt/evaluation/document-evidence-format-weight-v26-ordinary256-v1'
ordinary=[]
for name,path in sorted(paths.items()):
    values=[dict(model=name,**v) for shard in range(4) for v in rows(path/'scores'/f'shard-{shard}'/'scores.jsonl') if v['panel']=='ordinary'];assert len(values)==256;ordinary.extend(values)
raw=''.join(json.dumps(v)+'\n' for v in ordinary).encode();target=P/'ordinary-all-model-token-scores.jsonl.gz';target.write_bytes(gzip.compress(raw,mtime=0));assert gzip.decompress(target.read_bytes())==raw;record(target)
for src in (C/'figures').iterdir():copy(src)
sources=P/'sources';sources.mkdir();source_payloads={}
source_files=set(read(C/'release.json')['pipeline_sources'])|set(read(C/'postprocessing-sources.json')['files'])
for filename in sorted(source_files):
    src=R/filename;dst=sources/src.name;shutil.copyfile(src,dst);assert sha(src)==sha(dst);source_payloads[filename]=dict(public='sources/'+dst.name,sha256=sha(dst))
report.update(status='COMPLETE_COEFFICIENT_MATCHED_FORMAT_CONTROL_FIXED_READOUT',publication_utc=datetime.now(timezone.utc).isoformat(),
    interpretation=dict(primary_developmental_criterion=report['primary_developmental_decision'],
        primary='Selector1 direct minus format1 direct on the same already-inspected255 questions. Full point/interval and corrected/regressed counts are retained.',
        inference='Existing same-model emitted-ID comparisons stay separate and unchanged. The current primary concerns training-side direct-answer benefit.',
        confirmation='One data-order seed and adaptively reused developmental panel. Any positive needs separately frozen fresh, powered evaluation; a missed criterion is not a zero-effect finding.',
        cost='Generated actions and ordinary NLL are separate costs. Reserved-chip accounting is not a matched serving-latency or FLOP benchmark.',
        next='Close this missing-control question. Use its estimates to decide whether fresh confirmation is justified; no automatic coefficient sweep, scaling or RL.'),
    protected_originals={k:v for k,v in protection.items() if k!='files'},payloads=files,source_payloads=source_payloads,
    preservation='All original outputs/full128/256 states remain; terminal/latest-full EBS separately protected. All old scientific scores, gates and primary reports remain unchanged.')
with F.open('x') as f:json.dump(report,f,indent=2)
B=R/'local/backups/science/document-evidence-format-weight-v26-control-v1/PUBLICATION';B.mkdir(exist_ok=False);protected={}
for src in [F,*[p for p in P.rglob('*') if p.is_file()]]:
    relative=src.relative_to(R/'validation');dst=B/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    with dst.open('rb') as f:os.fsync(f.fileno())
    assert sha(src)==sha(dst);protected[str(relative)]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
for folder in [*[p for p in B.rglob('*') if p.is_dir()],B]:
    fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='PASS_FINAL_V26_PUBLICATION_READBACK_FSYNC',utc=datetime.now(timezone.utc).isoformat(),files=protected,EBS=str(B),report_file=str(F),report_sha256=sha(F),limits='Same-instance backup.')
with (C/'publication-protection.json').open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status=report['status'],primary=report['primary_developmental_decision'],report=str(F),report_sha256=sha(F),protected_public_files=len(protected))))
