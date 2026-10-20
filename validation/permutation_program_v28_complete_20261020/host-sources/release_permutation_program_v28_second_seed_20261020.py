"""Release fresh seed2029 first64 only after the first wave closes/audits."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import json,hashlib,subprocess,shutil,importlib.util
from train.partial_state_driver import ROOT,source_files,validate_program_execution
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';C=R/'local/diagnostics/permutation-program-v28-study-v1';assert ROOT==W
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(8*2**20),b''):h.update(part)
 return h.hexdigest()
plan=read(C/'plan.json');first=read(C/'stage64-release-seed2026102028.json')
assert plan['source_files']==source_files()==first['source_files']
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['source_commit']
assert sha(C/'supervise-job.py')==first['supervisor_sha256']
for name,key in [('audit_permutation_program_v28_job_20261020.py','audit_sha256'),('permutation_program_state_audit_v28_20261020.py','state_checker_sha256')]:
 assert sha(R/'local/analysis'/name)==first[key]
helper=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py';assert sha(helper)==first['helper_sha256']
spec=importlib.util.spec_from_file_location('v28_second_seed_system_checks',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
prior={}
for key,job in plan['jobs'].items():
 if job['seed']!=2026102028:continue
 phase=C/key/'terminal';out=Path(job['output']);audit_file=out/'terminal-full-audit.json';a=read(audit_file)
 assert a['status']=='PASS_V28_TERMINAL_ALL_STATE' and a['step']==4096 and a['model']==key and a['source_files']==source_files()
 assert read(phase/'exit.json')['exit_code']==0
 closed=read(phase/'closure.json');assert closed['all_original_session_processes_absent'] and closed['assigned_physical_chip_empty']
 assert sha(out/'step-0004096/model.pt')==a['model_sha256']
 assert sha(out/'step-0004096/COMPLETE.json')==a['COMPLETE_sha256']
 prior[key]=dict(path=str(audit_file),sha256=sha(audit_file))
assert len(prior)==3
top=h.topology();assert all(not v['neuron_processes'] and v['logical_neuroncore_config']==2 for v in top if v['neuron_device'] in (0,1,2))
assert h.available()>768*2**30
# The first wave already occupies the three retained-full slots. The global
# lease and prior terminal audit permit one staged full, then same-arm rotation.
assert shutil.disk_usage(R).free>35*2**30+16*1720574976+12*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+6*1720574976+4*2**30
assert shutil.disk_usage(R/'local/nvme-study').free>35*2**30
jobs={k:dict(protocol_sha256=j['protocol_sha256']) for k,j in plan['jobs'].items() if j['seed']==2026102029};assert len(jobs)==3
assert all(not Path(plan['jobs'][key]['output']).exists() for key in jobs)
now=datetime.now(timezone.utc);path=C/'stage64-release-seed2026102029.json'
release=dict(status='RELEASED_V28_SCIENCE_STAGE64_ONLY',utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=35)).isoformat(),wave_terminal_deadline_utc=(now+timedelta(hours=4.75)).isoformat(),
 jobs=jobs,source_files=source_files(),source_commit=plan['source_commit'],prospective_design_sha256=plan['prospective_design_sha256'],
 supervisor_sha256=first['supervisor_sha256'],helper_sha256=first['helper_sha256'],audit_sha256=first['audit_sha256'],state_checker_sha256=first['state_checker_sha256'],
 prior_seed_terminal_audits=prior,maximum_stage_reserved_chip_hours_including_grace=1.85,whole_reference_training_bound_chip_hours=30,physical_devices=[0,1,2],
 scope='Fresh B0 first64 for second data-order seed; same fixed recipe. Actual full64 audits precede restore/4096. No held model outcomes until all six terminals.',checkpoint_policy=plan['retention'],no_automatic_retry=True,
 storage_free_bytes={str(d):shutil.disk_usage(d).free for d in [R,R/'local/nvme-cpt',R/'local/nvme-study']})
with path.open('x') as f:json.dump(release,f,indent=2)
for key in jobs:
 job=plan['jobs'][key]
 validate_program_execution(read(job['protocol']),SimpleNamespace(execution_release=path,arm=job['arm'],output=Path(job['output']),protocol=Path(job['protocol']),resume=None,halt_after=64))
print(json.dumps({k:release[k] for k in ['status','utc','deadline_utc','wave_terminal_deadline_utc']}))
