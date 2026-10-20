"""Run once after a V28 phase physically closes; no model outcome evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,torch
from train.partial_state_driver import ROOT,source_files
from weekend_thinking.pretrained import load_base
from permutation_program_state_audit_v28_20261020 import read,sha,metrics,full_state,native
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';assert ROOT==W
C=R/'local/diagnostics/permutation-program-v28-study-v1'
a=argparse.ArgumentParser(description=__doc__);a.add_argument('--key',required=True);a.add_argument('--phase',choices=['first64','terminal'],required=True);args=a.parse_args()
job=read(C/'plan.json')['jobs'][args.key];p=read(job['protocol']);assert p['source_files']==source_files()
stage=C/args.key/args.phase;assert read(stage/'exit.json')['exit_code']==0
closure=read(stage/'closure.json');assert closure['all_original_session_processes_absent'] and closure['assigned_physical_chip_empty']
step=64 if args.phase=='first64' else 4096
p,order,observed=metrics(job,step);torch.set_num_threads(8);base,_=load_base(p['model_dir'])
witnesses=native(job,p);print('Native4 and exact initial master witnesses PASS.',flush=True)
state=full_state(job,p,order,observed,base);print('All actual full-state bytes/shapes/counters PASS.',flush=True)
# Check the model64 copy before its original full state can be pruned.
copy=R/'local/nvme-cpt/models/controlled-computation-v28-development64'/(args.key+'-step64.pt')
receipt=read(copy.with_suffix('.json'));assert sha(copy)==receipt['model_sha256']
report=dict(status='PASS_V28_FIRST64_ALL_STATE' if step==64 else 'PASS_V28_TERMINAL_ALL_STATE',utc=datetime.now(timezone.utc).isoformat(),**state,
 model=job['key'],protocol_sha256=sha(job['protocol']),source_files=source_files(),actual_native_checks=witnesses,observed_updates=step,ordered_exposure_and_component_counts_verified=True,
 development64_sha256=sha(copy),audit_source_sha256=sha(__file__),state_checker_sha256=sha(Path(__file__).with_name('permutation_program_state_audit_v28_20261020.py')),
 limits=['Actual training/recovery mechanics,not free-answer accuracy or generalization.','No uninterrupted-versus-resumed bitwise trajectory claim.','Latest full/terminal model/development64 retention follows the prospectively declared rolling policy.'])
path=stage/'independent-all-state-audit.json'
with path.open('x') as f:json.dump(report,f,indent=2)
if step==4096:
 with (Path(job['output'])/'terminal-full-audit.json').open('x') as f:json.dump(report,f,indent=2)
print(json.dumps({k:report[k] for k in ['status','model','step','COMPLETE_sha256']}),flush=True)
