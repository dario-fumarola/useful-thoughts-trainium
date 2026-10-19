"""One serial release of four predeclared fresh models;no automatic retry."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,shutil,socket,subprocess,sys
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-source-balanced-v17-study-v1';Q=R/'local/diagnostics/real-source-balanced-v17-world8-target640-qualification-v2'
helper=Q/'qualify.py';spec=importlib.util.spec_from_file_location('release_helper',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
read,sha=h.read,h.sha;plan=read(C/'plan.json');audit=Q/'qualification-independent-audit.json';a=read(audit)
assert a['status']=='PASS_ACTUAL_REAL_SOURCE_BALANCED_FULL_RESTORE' and sha(audit)==plan['qualification_sha256']
assert len(plan['jobs'])==4 and {j['key'] for j in plan['jobs']}=={f'{arm}-seed{seed}' for arm in ('source_direct','source_cot') for seed in (2026101913,2026101914)}
assert sha(R/'validation/real_source_balanced_v17_design_20261019.json')==plan['prospective_design_sha256']
W=Path(plan['source_root']);assert subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True)==''
assert subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=W,text=True).strip()==plan['source_commit']
for job in plan['jobs']:
 p=read(job['protocol']);assert sha(job['protocol'])==job['protocol_sha256']
 assert all(sha(W/k)==v for k,v in p['source_files'].items()) and p['source_files']==a['source_files']
 assert p['schema']=='real-source-balanced-v17' and p['purpose']=='real_source_balanced_study'
 assert sha(p['data_file'])==p['data_sha256'] and sha(p['held_primary_manifest'])==plan['held_primary_manifest_sha256']
 assert not Path(job['output']).exists() and not (C/job['key']).exists()
 for port in (52600+10*plan['jobs'].index(job),52601+10*plan['jobs'].index(job)):
  with socket.socket() as s:s.bind(('127.0.0.1',port))
top=h.topology();assigned=[z for z in top if z['neuron_device'] in range(8)]
assert len(assigned)==8 and all(not z['neuron_processes'] and z['logical_neuroncore_config']==2 for z in assigned)
assert h.available()>900*2**30 and shutil.disk_usage(R).free>58*2**30 and shutil.disk_usage(R/'local/nvme-cpt').free>300*2**30
qfull=read(R/'local/nvme-cpt/diagnostics/real-source-balanced-v17-world8-target640-qualification-v2/step-0000008/COMPLETE.json')
required=sum(v['bytes'] for v in qfull['files'].values());assert shutil.disk_usage(R).free-required-2**30>35*2**30
release=dict(status='RELEASED_FOUR_DECLARED_FRESH_REAL_SOURCE_BALANCED_MODELS',utc=h.utc(),plan_sha256=sha(C/'plan.json'),supervisor_sha256=sha(C/'supervise-job.py'),helper_sha256=sha(helper),qualification_file=str(audit),qualification_sha256=sha(audit),
 scope='Exactlysource_direct/source_cot,twofixedseeds,320updates/twopasses,terminalprimary;noextensions/retries/checkpointselection.',hardware='Four8rankjobsphysical0..7,allfreshB0.',
 review='Rootreviewed1656CPUchecks/all10240actualnativepacks;11actualqualifiedupdates/fullrestore/all6fullstates/memoryaudit;frozendesign/new512+256;sourceclosure/idledevices/ports/storage.',
 resources=dict(root_free_bytes=shutil.disk_usage(R).free,new_full_payload_bytes=required,root_reserve_bytes=35*2**30,backup_order=plan['checkpoint_policy']),
 limits=['Nouninterruptedtrajectoryparity/deviceRNGreadback/stochasticRLqualification.','NoV17capabilityoutcomesreviewed.','Tasklabelsandinferencebudgetsunequal;notFLOPmatching.'])
A=R/'local/backups/science'/C.name/'PRE_LAUNCH';A.mkdir(parents=True,exist_ok=False)
h.write(C/'release.json',release)
files={}
for source in [*C.iterdir(),Path(__file__),R/'local/analysis/prepare_real_source_study_v17_20261019.py',audit,helper,R/'validation/real_source_balanced_v17_design_20261019.json']:
 if not source.is_file():continue
 dest=A/source.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 with dest.open('rb') as f:os.fsync(f.fileno())
 assert sha(source)==sha(dest);files[str(source.relative_to(R))]=sha(source)
h.write(A/'COMPLETE.json',dict(status='PASS_PRELAUNCH_SOURCE_CONTROL_DESIGN_AUDIT_READBACK_FSYNC',files=files))
launched=[]
for job in plan['jobs']:
 shell=C/f'launch-{job["key"]}.sh'
 shell.write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /home/ubuntu/dario/scripts/env.sh\ncd '+str(W)+'\nexport PYTHONPATH="$PWD:/home/ubuntu/dario/local/analysis:/home/ubuntu/dario/local/test-deps"\nexec python -u '+str(C/'supervise-job.py')+' '+job['key']+' > '+str(C/f'{job["key"]}-supervisor.log')+' 2>&1\n')
 session=f'source-train-v17-{job["key"]}'
 subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(shell)],check=True)
 launched.append(dict(job=job['key'],session=session,launch_sha256=sha(shell)))
h.write(C/'controller-release.json',dict(utc=h.utc(),controllers=launched,prelaunch_topology=top))
print(json.dumps(dict(status=release['status'],controllers=launched),indent=2),flush=True)
