"""Release the first-seed continuation only after all three actual64 audits pass."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import json,hashlib,subprocess,shutil,importlib.util
from train.partial_state_driver import ROOT,source_files,validate_program_execution
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';C=R/'local/diagnostics/permutation-program-v28-study-v1';assert ROOT==W
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(C/'plan.json');first=read(C/'stage64-release-seed2026102028.json');assert plan['source_files']==source_files()==first['source_files']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
assert sha(C/'supervise-job.py')==first['supervisor_sha256']
assert sha(R/'local/analysis/audit_permutation_program_v28_job_20261020.py')==first['audit_sha256']
assert sha(R/'local/analysis/permutation_program_state_audit_v28_20261020.py')==first['state_checker_sha256']
helper=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py';assert sha(helper)==first['helper_sha256']
spec=importlib.util.spec_from_file_location('v28_continuation_system_checks',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
top=h.topology();assert all(not v['neuron_processes'] for v in top if v['neuron_device'] in (0,1,2)) and h.available()>768*2**30
now=datetime.now(timezone.utc);deadline=datetime.fromisoformat(first['wave_terminal_deadline_utc']);assert (deadline-now).total_seconds()>3*3600
jobs={};audits={}
for key in first['jobs']:
 job=plan['jobs'][key];stage=C/key/'first64';audit=read(stage/'independent-all-state-audit.json')
 assert audit['status']=='PASS_V28_FIRST64_ALL_STATE' and audit['step']==64 and audit['observed_updates']==64 and audit['source_files']==source_files()
 assert read(stage/'exit.json')['exit_code']==0 and read(stage/'closure.json')['assigned_physical_chip_empty']
 checkpoint=Path(job['output'])/'step-0000064';assert sha(checkpoint/'COMPLETE.json')==audit['COMPLETE_sha256']
 assert all((checkpoint/name).stat().st_size==v['bytes'] for name,v in read(checkpoint/'COMPLETE.json')['files'].items())
 jobs[key]=dict(protocol_sha256=job['protocol_sha256'],resume_file=str(checkpoint),resume_COMPLETE_sha256=audit['COMPLETE_sha256'],attempt_id='planned-restart-64')
 audits[key]=dict(path=str(stage/'independent-all-state-audit.json'),sha256=sha(stage/'independent-all-state-audit.json'))
assert len(jobs)==3
# Existing three full states plus one serialized new export and future seed terminal models.
assert shutil.disk_usage(R).free>35*2**30+16*1720574976+12*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+3*2*1720574976+4*2**30
path=C/'terminal-release-seed2026102028.json'
release=dict(status='RELEASED_V28_SCIENCE_TERMINAL4096',utc=now.isoformat(),deadline_utc=deadline.isoformat(),source_files=source_files(),source_commit=plan['source_commit'],jobs=jobs,prospective_design_sha256=plan['prospective_design_sha256'],supervisor_sha256=first['supervisor_sha256'],helper_sha256=first['helper_sha256'],first64_independent_audits=audits,initial_release_sha256=sha(C/'stage64-release-seed2026102028.json'),physical_devices=[0,1,2],whole_reference_training_bound_chip_hours=30,decision='Resume each actual64 state and complete fixed4096. No held outcomes until all six terminals; no performance stopping or checkpoint selection.',no_automatic_retry=True)
with path.open('x') as f:json.dump(release,f,indent=2)
for key,item in jobs.items():
 job=plan['jobs'][key]
 validate_program_execution(read(job['protocol']),SimpleNamespace(execution_release=path,arm=job['arm'],output=Path(job['output']),protocol=Path(job['protocol']),resume=Path(item['resume_file']),halt_after=None))
print(json.dumps({k:release[k] for k in ['status','utc','deadline_utc','physical_devices']}))
