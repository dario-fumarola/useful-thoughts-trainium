"""One declared fresh model on an isolated eight-rank partition; no automatic retry."""
from pathlib import Path
from datetime import datetime
import importlib.util,json,os,signal,socket,subprocess,sys,time,shutil
R=Path('/home/ubuntu/dario')
helper=R/'local/diagnostics/document-evidence-format-weight-v26-world8-qualification-v1/qualify.py'
spec=importlib.util.spec_from_file_location('frozen_real_mechanics_supervisor',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
read,write,sha,utc=h.read,h.write,h.sha,h.utc
identity,same,members=h.identity,h.same,h.members
available,topology,memory=h.available,h.topology,h.memory
C=R/'local/diagnostics/document-evidence-format-weight-v26-control-v1';W=R/'local/worktrees/document-evidence-format-weight-v26'
job_key=sys.argv[1];assert len(sys.argv)==2
plan=read(C/'plan.json');release=read(C/'release.json')
assert release['status']=='RELEASED_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_V26_TREATMENT_ONLY'
assert release['plan_sha256']==sha(C/'plan.json')
assert release['supervisor_sha256']==sha(__file__) and release['helper_sha256']==sha(helper)
assert read(release['qualification_file'])['status']=='PASS_ACTUAL_DOCUMENT_EVIDENCE_FORMAT_WEIGHT_FULL_RESTORE'
assert release['qualification_sha256']==sha(release['qualification_file'])
deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
jobs=[v for v in plan['jobs'] if v['key']==job_key];assert len(jobs)==1
arm=jobs[0]['arm'];assert arm == 'evidence_direct'
envspec=read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
for index,job in enumerate(jobs):
 d=C/job['key'];d.mkdir(exist_ok=False)
 p=read(job['protocol']);assert sha(job['protocol'])==job['protocol_sha256']
 assert all(sha(W/k)==v for k,v in p['source_files'].items())
 assert sha(p['held_primary_manifest'])==p['held_primary_manifest_sha256']
 out=Path(job['output']);assert not out.exists()
 assert (R/'local/nvme-cpt').is_mount() and available()>768*2**30
 assert shutil.disk_usage(R/'local/nvme-cpt').free>35*2**30+2*(plan['budget']['measured_full_state_bytes']+2**30) and shutil.disk_usage(R).free>35*2**30
 top=topology();devices=[v for v in top if v['neuron_device'] in job['physical_devices']]
 assert len(devices)==2 and all(not v['neuron_processes'] and v['logical_neuroncore_config']==2 for v in devices)
 assert time.time()<deadline-3600
 port=53300+10*next(i for i,v in enumerate(plan['jobs']) if v['key']==job_key)
 tmp=Path('/dev/shm')/f'df26{port}';tmp.mkdir(mode=0o700,exist_ok=False)
 path=str(tmp/'listener-xxxxxxxxxxxxxxxxxxxxxxxx')
 with socket.socket(socket.AF_UNIX) as s:s.bind(path)
 Path(path).unlink()
 for value in (port,port+1):
  with socket.socket() as s:s.bind(('127.0.0.1',value))
 env=os.environ.copy()
 env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
 assert shutil.which('libneuronpjrt-path',path=env['PATH'])==str(Path(sys.executable).parent/'libneuronpjrt-path')
 for key in envspec['unset']+['MALLOC_ARENA_MAX']:env.pop(key,None)
 env.update(envspec['environment']);env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),
   NEURON_RT_VISIBLE_CORES=p['visible_cores'],NEURONCORE_NUM_DEVICES='8',
   OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',
   MASTER_PORT=str(port),NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{port+1}',TMPDIR=str(tmp))
 assert {k:env.get(k) for k in p['runtime_policy']['environment']}==p['runtime_policy']['environment']
 cmd=['timeout','--signal=TERM','--kill-after=300s',str(min(86400,int(deadline-time.time())))+'s',
   sys.executable,'-u','-m','train.partial_state_driver','--protocol',job['protocol'],
   '--output',str(out),'--arm',arm,'--stop-file',str(d/'STOP'),'--deadline-utc',plan['deadline_utc']]
 with (d/'run.log').open('xb') as log,(d/'host-samples.jsonl').open('x') as samples:
  proc=subprocess.Popen(cmd,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  owned=identity(proc.pid);assert owned and owned['session']==owned['pgid']==proc.pid
  write(d/'launch.json',dict(utc=utc(),identity=owned,command=cmd,prelaunch_topology=top,AF_UNIX_actual_bind_pass=True))
  captured=False;stop=None
  while proc.poll() is None:
   procs=members(owned['session']);sample=dict(utc=utc(),MemAvailable_bytes=available(),
      filesystem_free_bytes={str(root):shutil.disk_usage(root).free for root in (R,R/'local/nvme-cpt')},
      processes=[dict(**v,memory=memory(v['pid'])) for v in procs])
   samples.write(json.dumps(sample)+'\n');samples.flush()
   if not captured:
    actual=topology();allocated=[v for z in actual if z['neuron_device'] in job['physical_devices'] for v in z['neuron_processes']]
    if len(allocated)==8:
     assert len({v['pid'] for v in allocated})==8 and {v['pid'] for v in allocated}<={v['pid'] for v in procs}
     write(d/'first-live.json',dict(utc=utc(),actual8=allocated,topology=actual,processes=procs));captured=True
   if stop is None and (sample['MemAvailable_bytes']<512*2**30 or min(sample['filesystem_free_bytes'].values())<35*2**30 or time.time()>=deadline):
    stop=dict(utc=utc(),sample=sample);write(d/'MEMORY_DISK_OR_DEADLINE_STOP.json',stop)
    # Request the driver's own complete-update/full-save stop first.
    write(d/'STOP',dict(reason='Declared resource or deadline boundary',utc=utc()))
   if stop is not None and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>360:
    h.kill_remaining(owned)
   time.sleep(10)
  code=proc.wait();write(d/'exit.json',dict(utc=utc(),exit_code=code,external_stop=stop))
 if code or stop:
  groups=h.kill_remaining(owned);write(d/'failure.json',dict(status='FAILED_OR_STOPPED_NO_RETRY',exit_code=code,exact_owned_cleanup_groups=groups))
  raise SystemExit('Preserve all original evidence and last complete state;no automatic retry')
 for _ in range(20):
  if not members(owned['session']):break
  time.sleep(1)
 assert captured and not members(owned['session'])
 assert all(not z['neuron_processes'] for z in topology() if z['neuron_device'] in job['physical_devices'])
 completion=read(out/'completion.json');assert completion['status']=='complete' and completion['actual_steps']==256
 verified={}
 for step in (128,256):
  checkpoint=out/f'step-{step:07d}';m=read(checkpoint/'COMPLETE.json')
  assert m['step']==step and m['world_size']==8 and set(m['files'])=={'model.pt',*[f'rank-{rank:05d}.pt' for rank in range(8)]}
  for name,value in m['files'].items():assert (checkpoint/name).stat().st_size==value['bytes'] and sha(checkpoint/name)==value['sha256']
  verified[str(step)]=sha(checkpoint/'COMPLETE.json')
 write(d/'closure.json',dict(status='COMPLETE_TRAINING_PHYSICALLY_CLOSED_FULL_STATES_VERIFIED',utc=utc(),
   all_original_session_processes_absent=True,assigned_devices_empty=True,completion_sha256=sha(out/'completion.json'),checkpoint_manifests=verified))
 print(json.dumps(dict(utc=utc(),arm=arm,seed=job['seed'],status='COMPLETE_PENDING_INDEPENDENT_SCIENCE_AUDIT')),flush=True)
write(C/f'{job_key}-completion.json',dict(status='SINGLE_FROZEN_JOB_CLOSED_PENDING_INDEPENDENT_AUDIT',utc=utc()))
