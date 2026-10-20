"""Prepare the fixed six-model reference; no launch or execution release."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,shutil
from train.partial_state_driver import ROOT,source_files,validate,order_for
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';C=R/'local/diagnostics/permutation-program-v28-study-v1'
assert ROOT==W
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip();assert commit.startswith('ce04236')
assert '1855 passed' in (W/'local/checks/tests.PaV2GN/tests.txt').read_text()
C.mkdir(exist_ok=False)
basep=read(R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/protocol.json')
D=R/'local/nvme-study/data/permutation-program-v28-study-v1';design=R/'validation/permutation_program_v28_design_20261020.json'
hardware=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualification-independent-audit.json'
decoder=R/'local/diagnostics/permutation-program-v28-TRAIN4-decode-qualification-v1/independent-audit.json'
assert read(hardware)['status']=='PASS_ACTUAL_PROGRAM_ALL_VIEW_FULL_RESTORE'
assert read(decoder)['status']=='PASS_ACTUAL_TRAIN_PROGRAM_DECODER_NATIVE_CACHE_RAW_TOKENS_AND_CPU_GREEDY'
assert read(R/'local/diagnostics/permutation-program-v28-storage-v1/retirement-complete.json')['all_originals_retained']
assert read(D/'manifest.json')['files']['TRAIN-labels.jsonl']==basep['data_sha256']
assert sha(D/'TRAIN-labels.jsonl')==basep['data_sha256']
assert sha(design)=='d01498ec6eec98fedad28bfa5b5426059ccceff196ef7522bb44ea27d98ebba6'
assert shutil.disk_usage(R).free>35*2**30+4*16*basep['base_signature']['parameters']+12*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+6*2*basep['base_signature']['parameters']+4*2**30
jobs={}
for wave,seed in enumerate((2026102028,2026102029)):
 orders=[]
 for index,arm in enumerate(('direct','trace','pause')):
  p=dict(basep,purpose='permutation_program_reference',items=32768,steps=4096,save_every=1024,seed=seed,program_arm=arm,
    checkpoint_retention='v28_latest_full_per_arm_keep_terminal_models',view_schedule='interleaved_fixed_item_plus_epoch_parity',
    data_file=str(D/'TRAIN-labels.jsonl'),data_sha256=sha(D/'TRAIN-labels.jsonl'),data_manifest_file=str(D/'manifest.json'),data_manifest_sha256=sha(D/'manifest.json'),
    hardware_qualification_file=str(hardware),hardware_qualification_sha256=sha(hardware),decoder_qualification_file=str(decoder),decoder_qualification_sha256=sha(decoder),
    prospective_design_file=str(design),prospective_design_sha256=sha(design),source_files=source_files(),visible_cores=f'{index*4}-{index*4+3}')
  validate(p);orders.append(order_for(p));name=f'program-{arm}-seed{seed}-v1';protocol=C/(name+'.json');write(protocol,p)
  output=R/'local/science-runs/controlled-computation-v28'/name;assert not output.exists()
  jobs[name]=dict(key=name,arm=arm,seed=seed,wave=wave,protocol=str(protocol),protocol_sha256=sha(protocol),output=str(output),physical_devices=[index],logical_cores=p['visible_cores'],master_port=63501+index*4+wave*20,root_comm_port=63502+index*4+wave*20)
 assert orders[0]==orders[1]==orders[2]
write(C/'plan.json',dict(status='PREPARED_SIX_MODEL_REFERENCE_NOT_EXECUTION_RELEASE',utc=datetime.now(timezone.utc).isoformat(),source_root=str(W),source_commit=commit,source_files=source_files(),prospective_design_file=str(design),prospective_design_sha256=sha(design),jobs=jobs,scope='Six fresh B0 models,first64/audit/planned restore/terminal4096. No held outputs before all six terminal models.',budget=read(design)['budget'],retention=read(design)['retention']))
print(json.dumps(dict(status='PREPARED_SIX_MODEL_REFERENCE_NOT_EXECUTION_RELEASE',jobs=list(jobs),source_commit=commit)))
