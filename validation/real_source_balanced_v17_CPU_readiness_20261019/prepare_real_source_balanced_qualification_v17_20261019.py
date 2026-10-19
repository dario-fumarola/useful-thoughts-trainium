"""Prepare one bounded both-mask/full-recovery check;no scientific-model release."""
from pathlib import Path
from datetime import datetime,timezone
import ast
import hashlib
import json
from train.partial_state_driver import ROOT,source_files,validate

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/real-source-balanced-v17';assert ROOT==W
C=R/'local/diagnostics/real-source-balanced-v17-world8-target640-qualification-v1'
C.mkdir(exist_ok=False)
old=R/'local/diagnostics/real-target-sft-v15-world8-target128-qualification-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit_path=R/'local/analysis/real-source-balanced-v17-native-pack-audit-20261019.json'
audit=json.loads(audit_path.read_text())
assert audit['status']=='ALL_10240_NATIVE_TRAIN_VIEWS_CAUSAL_MASKS_AND_LABEL_COUNTS_PASS'
assert audit['source_files']==source_files()
p=json.loads((old/'protocol.json').read_text())
p.update(schema='real-source-balanced-v17',purpose='real_source_balanced_qualification',target_slots=640,
    view_schedule='same_two_component_view_every_occurrence',
    loss='half_mean_source_prefix_plus_half_mean_task_per_example',
    source_loss_weight=.5,task_loss_weight=.5,source_files=source_files())
validate(p);(C/'protocol.json').write_text(json.dumps(p,indent=2))
s=(old/'qualify.py').read_text()
s=s.replace('real-target-sft-v15-world8-target128-qualification-v1','real-source-balanced-v17-world8-target640-qualification-v1')
s=s.replace('local/worktrees/real-target-sft-v15','local/worktrees/real-source-balanced-v17')
s=s.replace('2026-10-19T10:30:00+00:00','2026-10-19T13:30:00+00:00')
s=s.replace("'target_cot'","'source_cot'").replace("'target_direct'","'source_direct'")
s=s.replace('/dev/shm/rt15q','/dev/shm/rs17q').replace('65421','65481').replace('65422','65482')
ast.parse(s);assert 'real-target-sft-v15' not in s
(C/'qualify.py').write_text(s)
plan=dict(status='PREPARED_BOUNDED_SOURCE_BALANCED_MECHANICS_ONLY',utc=datetime.now(timezone.utc).isoformat(),
    source_root=str(W),source_commit='37028a1',source_files=source_files(),
    protocol_sha256=sha(C/'protocol.json'),supervisor_sha256=sha(C/'qualify.py'),
    native_pack_audit_sha256=sha(audit_path),CPU_tests=1656,
    design='Disposable first128 TRAIN. source_cot fresh1–6;physicalclosure;fullrestore6/updates7–8;separate fresh source_direct1–3. World8×micro4,width640,target640,LR1e-5,BF16 active/F32 masters,one ZeRO reducer.',
    endpoints=['24 initial native checks repeat four canonical contexts;actual BF16/dtype and bidirectional KL gates unchanged.',
               '11 finite updates,9 explicit changed-master/cast/replica witness updates;all6 full states read back independently.',
               'Both source/task losses and counts must match independently computed masks;global weighted loss equals0.5source+0.5task.',
               'Fresh process verifies every saved6 optimizer digest,RNG/cursor/state before updates7–8;no uninterrupted-trajectory bitwise claim.',
               'Post-save grouped PSS growth save4→6<2GiB;hostavailable>=512GiB andfilesystemfree>=35GiB.',
               'All3 sessions physically close;physical12/13 empty after every stage.'],
    deadline_utc='2026-10-19T13:30:00+00:00',
    resources='Logical48–55 onphysical12/13;one stage at a time,3600s maximum each,no automatic retry.',
    retention='All original full qualification states onNVMe;small audit/source/closure evidence readback-protected onEBS. No science model enters this check.',
    limits='This is mechanics qualification only. No held-model forward,scientific learning release,new serving stack or stochastic RL qualification.')
(C/'plan.json').write_text(json.dumps(plan,indent=2))
print(json.dumps(dict(control=str(C),protocol_sha256=plan['protocol_sha256'],supervisor_sha256=plan['supervisor_sha256']),indent=2),flush=True)
