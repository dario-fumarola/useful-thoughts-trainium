"""Protect metadata evidence and publish one bounded feasibility result."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-v27-confirmation-feasibility-v1';D=R/'local/nvme-study/data/musique-DEV-mirror-metadata-v27';A=R/'local/backups/data/document-evidence-v27-confirmation-feasibility-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
check=read(C/'independent-check.json');assert check['status'].startswith('PASS_INDEPENDENT_')
train=read(C/'result.json');dev=read(C/'DEV-mirror-metadata-result.json');assert dev['independent_count_upper_bound']<1024 and train['disjoint_fresh_eligible_rows']==0
assert check['DEV_result_sha256']==sha(C/'DEV-mirror-metadata-result.json') and check['TRAIN_result_sha256']==sha(C/'result.json')
F=R/'validation/document_evidence_v27_confirmation_feasibility_20261020.json';P=F.with_suffix('');P.mkdir(exist_ok=False);A.mkdir(exist_ok=False)
sources=[R/'local/analysis'/n for n in ['audit_document_evidence_v27_confirmation_feasibility_20261020.py','inspect_document_evidence_v27_DEV_metadata_20261020.py','inspect_document_evidence_v27_mirror_DEV_metadata_20261020.py','inspect_document_evidence_v27_mirror_DEV_metadata_v2_20261020.py','check_document_evidence_v27_feasibility_20261020.py','publish_document_evidence_v27_feasibility_20261020.py']]
files=[*[p for p in C.iterdir() if p.is_file()],*[p for p in D.iterdir() if p.is_file()],*[p for p in (R/'local/nvme-study/data/musique-DEV-metadata-v27').iterdir() if p.is_file()],*sources]
saved={}
for src in files:
 name=str(src.relative_to(R));dst=A/name;dst.parent.mkdir(parents=True,exist_ok=True);assert shutil.disk_usage(R).free>35*2**30+src.stat().st_size
 before=sha(src);shutil.copyfile(src,dst)
 with dst.open('rb') as f:os.fsync(f.fileno())
 assert sha(dst)==sha(src)==before;saved[name]=dict(bytes=src.stat().st_size,sha256=before)
for d in [*[p for p in A.rglob('*') if p.is_dir()],A]:
 fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
receipt=dict(status='PASS_METADATA_RAW_PROVENANCE_SOURCES_AND_RESULTS_READBACK_FSYNC',utc=datetime.now(timezone.utc).isoformat(),files=saved,limits='Same-instance backup. No model observations or TEST data.')
(A/'ARCHIVE.json').write_text(json.dumps(receipt,indent=2))
payloads={}
for name in ['plan.json','result.json','DEV-acquisition-attempt1-failure.json','DEV-mirror-metadata-attempt1-failure.json','DEV-metadata-schema-repair.json','DEV-mirror-metadata-result.json','independent-check.json']:
 src=C/name;dst=P/name;shutil.copyfile(src,dst);payloads[name]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
for src in [D/'acquisition.json',D/'plan.json',*sources]:
 name=('mirror-'+src.name if src.parent==D else src.name);dst=P/name;shutil.copyfile(src,dst);payloads[name]=dict(bytes=dst.stat().st_size,sha256=sha(dst))
report=dict(status='CLOSED_FRESH_SOURCE_FEASIBILITY_INSUFFICIENT_FOR_PLANNED_CONFIRMATION_NO_MODELS',utc=datetime.now(timezone.utc).isoformat(),question='Can unchanged source exclusions supply the planned powered fresh confirmation of V26?',
 result=dict(TRAIN_remaining_eligible=0,DEV_total=2417,DEV_clean_eligible=875,support_overlap_components=333,largest_component=37,fixed_greedy_independent=343,independent_upper_bound=614,desired_independent=1024),
 decision='Do not release a smaller underpowered confirmation, change old gates, or weaken exclusions. V26 stays developmental. A future confirmation needs a new source/protocol; proceed independently with the controlled-computation mechanism step of the north star.',
 unchanged_V26_report_sha256=sha(R/'validation/document_evidence_format_weight_v26_complete_20261020.json'),independent_check=check,
 source_provenance='Author raw DEV endpoint returned quota HTML before any question rows. Two pinned community DEV mirrors agree byte-for-byte; all19938 cached TRAIN records rechecked against author raw. Original namespace omission handled with conservative bare-ID exclusions.',
 operational_repair='A metadata guard incorrectly required20 passages;16 rows have17-19. Original partial inventory/source/failure preserved; v2 retains every supplied paragraph without data/score changes or another download.',
 statistical_limits=['Question counts are not independent evidence counts. No outcomes/semantic review/manual examples/model calls occurred.',
 'Normal-approximation power is planning only. At true3pp, point>=3pp gate has asymptotic marginal pass probability.5, not.8.',
 'For true5pp/discordance.15, n1024 gives approximate95% marginal passage and>=80% four-contrast union-bound lower bound; smaller effects or higher discordance need more.',
 'Greedy is not a maximum; the614 bound follows from1229 distinct original IDs and>=2 consumed per independent question.',
 'Author annotations can be imperfect; no novelty relative to base pretraining or general capability claim. TEST unrequested.'],
 archive_sha256=sha(A/'ARCHIVE.json'),archive_files=len(saved),public_payloads=payloads)
F.write_text(json.dumps(report,indent=2))
B=A/'PUBLICATION';B.mkdir()
for src in [F,*[p for p in P.iterdir() if p.is_file()]]:
 dst=B/src.relative_to(R/'validation');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
 with dst.open('rb') as f:os.fsync(f.fileno())
 assert sha(dst)==sha(src)
(C/'publication.json').write_text(json.dumps(dict(report=str(F),sha256=sha(F),archive_sha256=sha(A/'ARCHIVE.json'),public_files=len(payloads)+1),indent=2))
print(json.dumps(dict(status=report['status'],report=str(F),sha256=sha(F),raw_source_files=len(saved),public_files=len(payloads)+1)))
