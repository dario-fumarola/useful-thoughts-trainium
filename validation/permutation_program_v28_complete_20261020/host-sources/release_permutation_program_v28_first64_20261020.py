"""Release only the first seed's actual64-update milestone, not continuation."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import json,hashlib,shutil,subprocess,importlib.util
from train.partial_state_driver import ROOT,source_files,validate_program_execution
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';C=R/'local/diagnostics/permutation-program-v28-study-v1';assert ROOT==W
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(C/'plan.json');assert plan['source_files']==source_files()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==plan['source_commit']
assert '1855 passed' in (W/'local/checks/tests.PaV2GN/tests.txt').read_text()
assert read(R/'local/diagnostics/permutation-program-v28-storage-v1/retirement-complete.json')['all_originals_retained']
helper=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'
spec=importlib.util.spec_from_file_location('v28_release_system_checks',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
top=h.topology();assert all(not v['neuron_processes'] and v['logical_neuroncore_config']==2 for v in top if v['neuron_device'] in (0,1,2))
assert h.available()>768*2**30
jobs={k:dict(protocol_sha256=j['protocol_sha256']) for k,j in plan['jobs'].items() if j['seed']==2026102028};assert len(jobs)==3
params=1720574976
assert shutil.disk_usage(R).free>35*2**30+4*16*params+12*2**30
assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+6*2*params+4*2**30
assert shutil.disk_usage(R/'local/nvme-study').free>35*2**30
for f in ['permutation_program_state_audit_v28_20261020.py','audit_permutation_program_v28_job_20261020.py']:assert (R/'local/analysis'/f).is_file()
now=datetime.now(timezone.utc);path=C/'stage64-release-seed2026102028.json'
release=dict(status='RELEASED_V28_SCIENCE_STAGE64_ONLY',utc=now.isoformat(),deadline_utc=(now+timedelta(minutes=35)).isoformat(),wave_terminal_deadline_utc=(now+timedelta(hours=4.75)).isoformat(),jobs=jobs,source_files=source_files(),source_commit=plan['source_commit'],prospective_design_sha256=plan['prospective_design_sha256'],supervisor_sha256=sha(C/'supervise-job.py'),helper_sha256=sha(helper),audit_sha256=sha(R/'local/analysis/audit_permutation_program_v28_job_20261020.py'),state_checker_sha256=sha(R/'local/analysis/permutation_program_state_audit_v28_20261020.py'),maximum_stage_reserved_chip_hours_including_grace=1.85,whole_reference_training_bound_chip_hours=30,physical_devices=[0,1,2],scope='Only fresh first64 updates for first-seed direct/trace/pause. Independent pure-arm full64 state/order/mask checks precede any4096 continuation release. No held outputs.',checkpoint_policy=plan['retention'],no_automatic_retry=True,storage_free_bytes={str(d):shutil.disk_usage(d).free for d in [R,R/'local/nvme-cpt',R/'local/nvme-study']})
with path.open('x') as f:json.dump(release,f,indent=2)
for key in jobs:
 job=plan['jobs'][key]
 validate_program_execution(read(job['protocol']),SimpleNamespace(execution_release=path,arm=job['arm'],output=Path(job['output']),protocol=Path(job['protocol']),resume=None,halt_after=64))
print(json.dumps({k:release[k] for k in ['status','utc','deadline_utc','wave_terminal_deadline_utc','physical_devices']}))
