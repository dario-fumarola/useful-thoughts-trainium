"""Build the two-arm pilot only after actual new-objective/decoder audits pass.

No release or model launch is performed here. Root must review the concrete
protocols, storage protection and measured qualification before release.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess
from train.partial_state_driver import ROOT, source_files, validate, order_for

R=Path('/home/ubuntu/dario')
W=R/'local/worktrees/document-evidence-v23'
E=R/'local/worktrees/document-evidence-eval-v23'
Q=R/'local/diagnostics/document-evidence-plan-v23-world8-qualification-v1'
DQ=R/'local/diagnostics/document-evidence-plan-v23-decode-qualification-v1'
C=R/'local/diagnostics/document-evidence-plan-v23-pilot-v1'
D=R/'local/nvme-cpt/data/document-evidence-plan-v23-TRAIN2048-held255-v2'
assert ROOT==W
read=lambda path:json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
 return h.hexdigest()
def write(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2)

a=read(Q/'qualification-independent-audit.json')
decoder=read(DQ/'independent-audit.json')
base=read(Q/'protocol.json')
assert a['status']=='PASS_ACTUAL_DOCUMENT_EVIDENCE_PLAN_FULL_RESTORE'
assert a['actual_updates']==dict(evidence_select=8,evidence_direct=3)
assert a['source_files']==base['source_files']==source_files()
assert a['qualified_training_shape']=={k:base[k] for k in ['world_size','micro_batch','global_batch','width','target_slots','head_chunk','accumulation']}
assert decoder['status']=='PASS_ACTUAL_TRAIN_ONLY_EVIDENCE_PLAN_FP32_DECODING' and decoder['execution_errors']==0
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()=='73281416f9f5c8ab953e279f8940479ab6bbe384'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=E,text=True).strip()=='625fb0f6d6c28c11a02d60a9c578d6a5f1cad603'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=E,text=True).strip()
manifest=read(D/'manifest.json')
assert sha(D/'manifest.json')==base['data_manifest_sha256']
assert read(D/'independent-source-check.json')['status']=='PASS_INDEPENDENT_SOURCE_SPLIT_AND_LABEL_IDENTITY_CPU_ONLY'
for record in manifest['files'].values():assert sha(record['path'])==record['sha256']
ordinary=R/'local/nvme-cpt/data/document-QA-v21-ordinary256-v2/manifest.json'
om=read(ordinary)
assert om['items']==om['unique_pages']==256 and sha(om['data_file'])==om['data_sha256']
C.mkdir(exist_ok=False)
for name,source in [('prospective-design.json',R/'configs/document_evidence_v23_pilot.json'),
                    ('readout-addendum.json',R/'configs/document_evidence_v23_readout.json')]:
 shutil.copyfile(source,C/name)
design=read(C/'prospective-design.json')
assert design['pilot_seed']==2026101923 and design['training']['steps']==256

jobs=[]
for i,arm in enumerate(('evidence_direct','evidence_select')):
 p=dict(base,purpose='document_evidence_plan_pilot',items=2048,steps=256,save_every=128,
    visible_cores=f'{i*8}-{i*8+7}',seed=2026101923,
    hardware_qualification_file=str(Q/'qualification-independent-audit.json'),
    hardware_qualification_sha256=sha(Q/'qualification-independent-audit.json'),
    held_primary_manifest=str(D/'manifest.json'),held_primary_manifest_sha256=sha(D/'manifest.json'),
    prospective_design_sha256=sha(C/'prospective-design.json'),
    readout_addendum_sha256=sha(C/'readout-addendum.json'),ordinary_manifest_sha256=sha(ordinary))
 validate(p)
 assert Counter(order_for(p))==Counter({j:4 for j in range(2048)})
 protocol=C/f'protocol-{arm}-seed2026101923.json'
 write(protocol,p)
 out=R/'local/nvme-cpt/runs'/f'document-evidence-plan-v23-{arm}-TRAIN2048-seed2026101923-v1'
 assert not out.exists()
 jobs.append(dict(arm=arm,seed=2026101923,key=f'{arm}-seed2026101923',
    protocol=str(protocol),protocol_sha256=sha(protocol),output=str(out),physical_devices=[2*i,2*i+1]))

supervisor=(R/'local/diagnostics/document-QA-v21-study-v1/supervise-job.py').read_text()
for old,new in [
 ('document-QA-v21-world8-width3328-qualification-v1',Q.name),
 ('document-QA-v21-study-v1',C.name),
 ('local/worktrees/document-qa-v21','local/worktrees/document-evidence-v23'),
 ('RELEASED_DOCUMENT_QA_V21_SCIENCE_AND_FIXED_EVALUATION','RELEASED_DOCUMENT_EVIDENCE_V23_PILOT_AND_FIXED_EVALUATION'),
 ('PASS_ACTUAL_DOCUMENT_QA_FULL_RESTORE','PASS_ACTUAL_DOCUMENT_EVIDENCE_PLAN_FULL_RESTORE'),
 ("('qa_direct','qa_prior','qa_post')","('evidence_direct','evidence_select')"),
 ('port=53000+10*','port=53100+10*'),("f'dq21{port}'","f'dq23{port}'")]:
 assert old in supervisor,old
 supervisor=supervisor.replace(old,new)
ast.parse(supervisor)
assert 'document-QA-v21' not in supervisor and 'qa_prior' not in supervisor
(C/'supervise-job.py').write_text(supervisor)
full=read(R/'local/nvme-cpt/diagnostics/document-evidence-plan-v23-world8-qualification-v1/step-0000008/COMPLETE.json')
full_bytes=sum(v['bytes'] for v in full['files'].values())
assert shutil.disk_usage(R/'local/nvme-cpt').free>4*(full_bytes+512*2**20)+35*2**30
assert shutil.disk_usage(R).free>full_bytes+35*2**30+512*2**20
storage=R/'local/diagnostics/document-evidence-plan-v23-storage-preparation-v1/completion.json'
assert read(storage)['status']=='PASS_INACTIVE_INTERMEDIATE_DUPLICATE_RELOCATED_ORIGINAL_AND_TERMINALS_UNCHANGED'
plan=dict(status='PREPARED_TWO_FRESH_EVIDENCE_MODELS_PENDING_SERIAL_RELEASE',
 utc=datetime.now(timezone.utc).isoformat(),source_root=str(W),source_commit='73281416f9f5c8ab953e279f8940479ab6bbe384',
 evaluation_source_commit='625fb0f6d6c28c11a02d60a9c578d6a5f1cad603',jobs=jobs,
 deadline_utc='2026-10-20T00:00:00+00:00',qualification_file=str(Q/'qualification-independent-audit.json'),
 qualification_sha256=sha(Q/'qualification-independent-audit.json'),decoding_qualification_sha256=sha(DQ/'independent-audit.json'),
 prospective_design_sha256=sha(C/'prospective-design.json'),readout_addendum_sha256=sha(C/'readout-addendum.json'),
 held_primary_manifest_sha256=sha(D/'manifest.json'),ordinary_manifest_sha256=sha(ordinary),
 execution='Two disjoint world8 partitions may run concurrently. FreshB0,identical question order,terminal256 primary. No automatic retry,early outcome review or checkpoint selection. No second seed released.',
 checkpoint_policy='Four original full128/256 states remain NVMe. Protect first complete128 full onEBS and readback before retiring ANY oldEBSduplicate. Reverify oldoriginal and its separateEBSmodel first. Retain all original scientific states andall terminalmodels. Newfull256 protects FIRST before retiring superseded128backup;35GiBfloor always.',
 storage_preparation_file=str(storage),storage_preparation_sha256=sha(storage),
 budget=dict(root_free_bytes=shutil.disk_usage(R).free,NVMe_free_bytes=shutil.disk_usage(R/'local/nvme-cpt').free,
   measured_qualification_full_bytes=full_bytes,extra_per_state_reserve_bytes=512*2**20,
   max_reserved_chips=4,expected_training_chip_hours=6,hard_reserved_chip_hour_upper_bound=24),
 scope='New passage-selection recipe pilot,not a V21causalablation. Fresh255 mostlytwohop;no broadpretraining,RL,newsize,postanswerinternalization orCoTsuperiorityclaim.')
write(C/'plan.json',plan)
print(json.dumps(dict(status=plan['status'],jobs=2,output=str(C)),indent=2))
