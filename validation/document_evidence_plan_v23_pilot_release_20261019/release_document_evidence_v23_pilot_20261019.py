"""Concrete serial release after root review;no scientific result is read here."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
p=read(C/'plan.json');q=read(p['qualification_file'])
assert p['status']=='PREPARED_TWO_FRESH_EVIDENCE_MODELS_PENDING_SERIAL_RELEASE'
assert q['status']=='PASS_ACTUAL_DOCUMENT_EVIDENCE_PLAN_FULL_RESTORE'
assert sha(p['qualification_file'])==p['qualification_sha256']
DQ=R/'local/diagnostics/document-evidence-plan-v23-decode-qualification-v1/independent-audit.json'
assert read(DQ)['status']=='PASS_ACTUAL_TRAIN_ONLY_EVIDENCE_PLAN_FP32_DECODING' and sha(DQ)==p['decoding_qualification_sha256']
assert sha(C/'prospective-design.json')==p['prospective_design_sha256'] and sha(C/'readout-addendum.json')==p['readout_addendum_sha256']
assert len(p['jobs'])==2 and {j['arm'] for j in p['jobs']}=={'evidence_direct','evidence_select'}
assert all(j['seed']==2026101923 and not Path(j['output']).exists() for j in p['jobs'])
assert shutil.disk_usage(R).free>35*2**30+p['budget']['measured_qualification_full_bytes']+2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+4*(p['budget']['measured_qualification_full_bytes']+512*2**20)
# Source-bound decoder functions only;no model load orheld forward.
import sys
sys.path.insert(0,str(R/'local/worktrees/document-evidence-eval-v23'))
from train.document_evidence_plan_evaluate import sources
names=['document_evidence_v23_state_audit_20261019.py','protect_document_evidence_v23_first_full128_20261019.py','audit_document_evidence_v23_terminals_20261019.py','protect_document_evidence_v23_terminals_20261019.py','bind_document_evidence_v23_terminals_20261019.py','watch_document_evidence_v23_terminals_20261019.py','real_source_state_audit_v17_20261019.py']
release=dict(status='RELEASED_DOCUMENT_EVIDENCE_V23_PILOT_AND_FIXED_EVALUATION',utc=datetime.now(timezone.utc).isoformat(),
 plan_sha256=sha(C/'plan.json'),supervisor_sha256=sha(C/'supervise-job.py'),
 helper_sha256=sha(R/'local/diagnostics/document-evidence-plan-v23-world8-qualification-v1/qualify.py'),
 qualification_file=p['qualification_file'],qualification_sha256=p['qualification_sha256'],
 decoding_qualification_file=str(DQ),decoding_qualification_sha256=sha(DQ),
 data_manifest_sha256=p['held_primary_manifest_sha256'],ordinary_manifest_sha256=p['ordinary_manifest_sha256'],
 evaluation_source_files=sources(),model_specs={j['key']:dict(arm=j['arm'],seed=j['seed'],step=256,run_id=Path(j['output']).name,training_protocol_sha256=j['protocol_sha256']) for j in p['jobs']},
 model_modes={'evidence_direct':['direct','selection'],'evidence_select':['direct','selection','empty','oracle']},
 pipeline_sources={str((R/'local/analysis'/name).relative_to(R)):sha(R/'local/analysis'/name) for name in names},
 training_snapshot=p['source_commit'],evaluation_snapshot=p['evaluation_source_commit'],span_snapshot='e0607fd',
 primary='Bselected-Adirect complete canonicalEM>=3pp andpaired95%lower>0 onall255. Nonemptyvalidselection>=95%. No second seedbeforethese declaredpilotgates.',
 inference_claim='Requires positive paired95%lower bounds versus SAME B direct AND B empty,as frozen readoutaddendum. Net recipe gain is distinct.',
 reviewer_decision='PASS_BOUNDED_PILOT_RELEASE. Actual11updates,all6fullstates andfreshprocessrestore pass;warmupdates about14s,post-save groupedPSS83-97GiB. Same source/finalcoefficients independently reconstructed. Existing4chips,expected6chip-hours,hard24chip-hours until00:00UTC. Original128/256states retained;firstfullEBSprotection watcher andterminal audit/protection/binding prepared. No heldout model outcome inspected.',
 protection='Protect new128full before retiring older EBSoptimizerduplicate;old original andnewbackup reread. New256full protects before older128EBSoptimizerduplicate retires. Originalstates/allterminalmodels retained;35GiBfilesystem and512GiBhostavailable floors. One historical intermediate model BACKUP duplicate relocated with3-copyhash/fsync tosecondexistingNVMe;latestfull/allV21terminalmodels untouched untilnewfullprotected. Same-instance,not off-instance.',
 limits=['New supervision/loss/schedule recipe,not a controlledV21rootcause ablation.','Fresh255 mostlytwohop;not officialS5,nopretrainingnovelty,semanticgold oradequatepowerclaim.','SupportIDs are imperfect author annotations,not entailment certificates.','No RL,8B,splitrescue,broadpretraining,postanswerinternalization orordinaryCoTsuperiorityclaim.','Terminal256 only;no result-driven extensions,retries,prompt/parser changes orcheckpointselection.'])
with (C/'release.json').open('x') as stream:json.dump(release,stream,indent=2)
print(json.dumps({k:release[k] for k in ['status','plan_sha256','training_snapshot','evaluation_snapshot','span_snapshot']},indent=2))
