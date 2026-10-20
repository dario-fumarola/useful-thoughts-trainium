"""Prepare disposable all-view qualification. Does not launch or release learning."""
from pathlib import Path
import json,hashlib,subprocess,shutil,py_compile
from train.partial_state_driver import ROOT,source_files,validate
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28'
C=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1'
D=R/'local/nvme-study/data/permutation-program-v28-cpu-v1'
assert ROOT==W
read=lambda f:json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(f,v):
 with Path(f).open('x') as g:json.dump(v,g,indent=2)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
C.mkdir(exist_ok=False)
old=read(R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1/protocol.json')
p={k:old[k] for k in ['model_id','model_dir','base_protocol','base_protocol_sha256','base_signature','precision_contract','runtime_policy','parameter_dtype','optimizer_dtype','head_chunk','schedule','checkpoint_export_cleanup','checkpoint_memory_diagnostic']}
p.update(schema='permutation-program-sft-v28',purpose='permutation_program_qualification',experiment_id='controlled-computation-v28',world_size=4,micro_batch=8,global_batch=32,accumulation=1,width=256,target_slots=64,items=128,steps=8,save_every=4,seed=2026102028,lr=1e-5,view_schedule='qualification_all_four_views_each_update',loss='separate_mean_answer_work_per_example',answer_loss_weight=1.,work_loss_weight=1.,data_file=str(D/'TRAIN-labels.jsonl'),data_sha256=sha(D/'TRAIN-labels.jsonl'),data_manifest_file=str(D/'manifest.json'),data_manifest_sha256=sha(D/'manifest.json'),source_files=source_files(),visible_cores='48-51')
validate(p);write(C/'protocol.json',p)
oldC=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1'
s=(oldC/'qualify.py').read_text()
# New identity/topology/paths only. Existing owned-process/resource supervisor is reused.
replacements={
'Coefficient-matched format-only world8 qualification':'Permutation-program all-view world4 qualification',
"local/worktrees/document-evidence-format-weight-v26":"local/worktrees/controlled-computation-v28",
"RELEASED_BOUNDED_FORMAT_WEIGHT_MECHANICS_ONLY":"RELEASED_BOUNDED_PROGRAM_ALL_VIEW_MECHANICS_ONLY",
"assert os.path.ismount(R/'local/nvme')":"assert os.path.ismount(R/'local/nvme-study')",
'Disposable first128 format-only TRAIN mechanics':'Disposable first128 program TRAIN all-view mechanics',
'local/nvme/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1':'local/nvme-study/diagnostics/permutation-program-v28-world4-qualification-v1',
"'evidence_direct'":"'program_qualification'",
"(12,13)":"(12,)",
"len(devices)==2":"len(devices)==1",
"df26q":"pp28q",
"NEURON_RT_VISIBLE_CORES='48-55',NEURONCORE_NUM_DEVICES='8'":"NEURON_RT_VISIBLE_CORES='48-51',NEURONCORE_NUM_DEVICES='4'",
"63251":"63351","63252":"63352",
"len(allocated)==8":"len(allocated)==4",
"len({v['pid'] for v in allocated})==8":"len({v['pid'] for v in allocated})==4",
"actual8_on_physical12_13":"actual4_on_physical12",
"R/'local/nvme',R/'local/nvme-cpt'":"R/'local/nvme-study',R/'local/nvme-cpt'",
"assigned_physical12_13_empty":"assigned_physical12_empty",
"range(2,step+1,2)":"range(4,step+1,4)",
"manifest['world_size']==8":"manifest['world_size']==4",
"range(8)":"range(4)",
"format_control_steps=8":"program_mixed_view_steps=8",
}
for a,b in replacements.items():
 assert a in s,a
 s=s.replace(a,b)
# One total reserved physical chip, not four chips. Full4/full8 retained unchanged.
(C/'qualify.py').write_text(s)
py_compile.compile(str(C/'qualify.py'),doraise=True)
write(C/'preparation.json',dict(status='PREPARED_BOUNDED_PROGRAM_QUALIFICATION_NOT_YET_RELEASED',source_root=str(W),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),protocol_sha256=sha(C/'protocol.json'),supervisor_sha256=sha(C/'qualify.py'),scope='Only8 mixed-view updates,full4/new-process restore/full8. No learned-capability evaluation.',full_states=[4,8],no_automatic_retry=True))
print(json.dumps(dict(status='PREPARED_BOUNDED_PROGRAM_QUALIFICATION_NOT_YET_RELEASED',control=str(C),protocol_sha256=sha(C/'protocol.json'))))
