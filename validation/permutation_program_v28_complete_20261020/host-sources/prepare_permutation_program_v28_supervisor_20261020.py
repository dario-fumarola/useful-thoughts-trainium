"""Write the bounded two-phase per-run supervisor; does not launch it."""
from pathlib import Path
import py_compile
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/permutation-program-v28-study-v1'
s='''"""One released V28 arm/phase, with owned-process and actual-topology guards."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import argparse,json,importlib.util,sys,os,socket,subprocess,time,signal,shutil
from train.partial_state_driver import ROOT,source_files,validate,validate_program_execution
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-study';assert ROOT==W
C=Path(__file__).resolve().parent
helper=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'
spec=importlib.util.spec_from_file_location('v28_owned_process_helpers',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
a=argparse.ArgumentParser(description=__doc__);a.add_argument('--key',required=True);a.add_argument('--phase',choices=['first64','terminal'],required=True);a.add_argument('--release',type=Path,required=True);args=a.parse_args()
plan=h.read(C/'plan.json');release=h.read(args.release);job=plan['jobs'][args.key];p=h.read(job['protocol'])
assert plan['source_files']==source_files()==release['source_files'] and release['supervisor_sha256']==h.sha(__file__) and release['helper_sha256']==h.sha(helper)
assert h.sha(job['protocol'])==job['protocol_sha256']==release['jobs'][args.key]['protocol_sha256'];validate(p)
resume=None if args.phase=='first64' else Path(release['jobs'][args.key]['resume_file'])
check=SimpleNamespace(execution_release=args.release,arm=job['arm'],output=Path(job['output']),protocol=Path(job['protocol']),resume=resume,halt_after=64 if args.phase=='first64' else None)
validate_program_execution(p,check)
deadline=datetime.fromisoformat(release['deadline_utc']).timestamp();assert time.time()<deadline-120
out=Path(job['output']);assert (not out.exists()) if args.phase=='first64' else out.is_dir()
assigned=set(job['physical_devices']);top=h.topology();devices=[v for v in top if v['neuron_device'] in assigned]
assert len(devices)==1 and all(not v['neuron_processes'] and v['logical_neuroncore_config']==2 for v in devices)
assert h.available()>768*2**30
for disk in (R,R/'local/nvme-cpt',R/'local/nvme-study'):assert shutil.disk_usage(disk).free>35*2**30
stage=C/args.key/args.phase;stage.mkdir(parents=True,exist_ok=False)
env_spec=h.read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
assert shutil.which('libneuronpjrt-path',path=env['PATH'])==str(Path(sys.executable).parent/'libneuronpjrt-path')
for key in env_spec['unset']+['MALLOC_ARENA_MAX']:env.pop(key,None)
offset=0 if args.phase=='first64' else 100
master=job['master_port']+offset;comm=job['root_comm_port']+offset
for port in (master,comm):
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
tmp=Path(f'/dev/shm/v28s{job["wave"]}{job["physical_devices"][0]}{offset}');tmp.mkdir(mode=0o700,exist_ok=False)
sock=socket.socket(socket.AF_UNIX);sock.bind(str(tmp/'pymp-probe-listener-xxxxxxxxxxxxxxxxxxxxxxxx'));sock.close();(tmp/'pymp-probe-listener-xxxxxxxxxxxxxxxxxxxxxxxx').unlink()
env.update(env_spec['environment']);env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES=job['logical_cores'],NEURONCORE_NUM_DEVICES='4',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',MASTER_PORT=str(master),NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{comm}',TMPDIR=str(tmp))
command=['timeout','--signal=TERM','--kill-after=120s',str(max(1,int(deadline-time.time())))+'s',sys.executable,'-u','-m','train.partial_state_driver','--protocol',job['protocol'],'--output',job['output'],'--arm',job['arm'],'--execution-release',str(args.release),'--stop-file',str(stage/'STOP'),'--deadline-utc',release['deadline_utc']]
if args.phase=='first64':command+=['--halt-after','64']
else:command+=['--resume',str(resume),'--attempt-id',release['jobs'][args.key]['attempt_id']]
with (stage/'run.log').open('xb') as log,(stage/'host-samples.jsonl').open('x') as samples:
 proc=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=h.identity(proc.pid);assert owned and owned['pgid']==owned['session']==proc.pid
 h.write(stage/'launch.json',dict(utc=h.utc(),identity=owned,command=command,release_sha256=h.sha(args.release),AF_UNIX_actual_bind_pass=True,environment={k:env[k] for k in list(env_spec['environment'])+['NEURON_RT_VISIBLE_CORES','NEURONCORE_NUM_DEVICES','MASTER_PORT','NEURON_RT_ROOT_COMM_ID','TMPDIR']},prelaunch_topology=top))
 captured=False;stop=None
 while proc.poll() is None:
  members=h.members(owned['session']);avail=h.available();sample=dict(utc=h.utc(),MemAvailable_bytes=avail,processes=[dict(**v,memory=h.memory(v['pid'])) for v in members]);samples.write(json.dumps(sample)+'\\n');samples.flush()
  if not captured:
   actual=h.topology();allocated=[v for z in actual if z['neuron_device'] in assigned for v in z['neuron_processes']]
   if len(allocated)==4:
    assert len({v['pid'] for v in allocated})==4 and {v['pid'] for v in allocated}<={v['pid'] for v in members}
    h.write(stage/'first-live.json',dict(utc=h.utc(),topology=actual,actual4_on_assigned_chip=allocated));captured=True
  if stop is None and (avail<512*2**30 or time.time()>=deadline or min(shutil.disk_usage(v).free for v in (R,R/'local/nvme-cpt',R/'local/nvme-study'))<35*2**30):
   stop=dict(utc=h.utc(),sample=sample);h.write(stage/'MEMORY_OR_DEADLINE_STOP.json',stop)
   if h.same(owned):os.kill(owned['pid'],signal.SIGTERM)
  if stop and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>120:h.kill_remaining(owned)
  time.sleep(2)
 code=proc.wait();h.write(stage/'exit.json',dict(utc=h.utc(),exit_code=code,external_stop=stop))
if code or stop:
 h.write(stage/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',exit_code=code,exact_owned_cleanup_groups=h.kill_remaining(owned)));raise SystemExit('V28 phase failed; original outputs retained, no automatic retry')
for _ in range(30):
 if not h.members(owned['session']):break
 time.sleep(1)
assert captured and not h.members(owned['session']) and all(not v['neuron_processes'] for z in h.topology() if z['neuron_device'] in assigned for v in [z])
# Read back the actual source's complete receipt; tensor-byte audit is separate.
expected=64 if args.phase=='first64' else 4096
completion=h.read(out/'completion.json');assert completion['actual_steps']==expected and completion['status']==('stopped_at_boundary' if args.phase=='first64' else 'complete')
checkpoint=out/f'step-{expected:07d}';m=h.read(checkpoint/'COMPLETE.json');receipt=h.read(checkpoint/'READBACK.json')
assert receipt['status']=='FULL_STATE_READBACK_FSYNC' and receipt['step']==expected and receipt['COMPLETE_sha256']==h.sha(checkpoint/'COMPLETE.json')
assert all((checkpoint/name).stat().st_size==v['bytes'] for name,v in m['files'].items())
h.write(stage/'closure.json',dict(status='COMPLETE_PHYSICALLY_CLOSED_PENDING_INDEPENDENT_ALL_STATE_AUDIT',utc=h.utc(),actual_steps=expected,all_original_session_processes_absent=True,assigned_physical_chip_empty=True,completion_sha256=h.sha(out/'completion.json'),COMPLETE_sha256=h.sha(checkpoint/'COMPLETE.json'),READBACK_sha256=h.sha(checkpoint/'READBACK.json')))
'''
(C/'supervise-job.py').write_text(s);py_compile.compile(str(C/'supervise-job.py'),doraise=True)
print('Prepared bounded first64/terminal supervisor; no job launched.')
