"""Coefficient-only world8 qualification: actual full-state save/restart. No retry."""
import hashlib,json,os,signal,socket,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path
C=Path(__file__).resolve().parent;R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-weight-v25'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def identity(pid):
 try:
  v=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,start_ticks=int(v[19]),ppid=int(v[1]),pgid=int(v[2]),session=int(v[3]))
 except (FileNotFoundError,ProcessLookupError):return None
def same(v):
 now=identity(v['pid']);return now and all(now[k]==v[k] for k in ('pid','start_ticks','pgid','session'))
def members(sid):return [v for p in Path('/proc').glob('[0-9]*') if (v:=identity(int(p.name))) and v['session']==sid]
def memory(pid):
 try:
  v={l.split(':',1)[0]:l.split(':',1)[1] for l in (Path('/proc')/str(pid)/'smaps_rollup').read_text().splitlines() if ':' in l}
  return {k:int(v[k].split()[0])*1024 for k in ('Pss','Rss','Private_Clean','Private_Dirty')}
 except (FileNotFoundError,ProcessLookupError):return None
def available():return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
def topology():return json.loads(subprocess.check_output(['/opt/aws/neuron/bin/neuron-ls','--json-output'],text=True))
def kill_remaining(owned):
 groups=set()
 for v in members(owned['session']):
  if same(v) and v['pgid']>1 and v['pgid']!=os.getpgrp():groups.add(v['pgid'])
 for group in groups:
  try:os.killpg(group,signal.SIGKILL)
  except ProcessLookupError:pass
 return sorted(groups)
if __name__=='__main__':
 release=read(C/'qualification-release.json');deadline=datetime.fromisoformat('2026-10-19T22:00:00+00:00').timestamp()
 assert release['status']=='RELEASED_BOUNDED_SELECTION_WEIGHT_MECHANICS_ONLY' and release['supervisor_sha256']==sha(__file__)
 protocol=C/'protocol.json';p=read(protocol)
 assert release['protocol_sha256']==sha(protocol) and all(sha(W/k)==v for k,v in p['source_files'].items())
 assert os.path.ismount(R/'local/nvme-study') and available()>768*2**30
 write(C/'qualification-ready.json',dict(utc=utc(),supervisor=identity(os.getpid()),scope='Disposable first128 pointer TRAIN model;no held/model outcome or science model'))
 env_spec=read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
 out=R/'local/nvme-study/diagnostics/document-evidence-weight-v25-world8-qualification-v1'
 for index,(stage,step,arm) in enumerate((('select-fresh4',4,'evidence_select'),('select-resume8',8,'evidence_select'))):
  run_out = out
  assert index==1 or not run_out.exists()
  top=topology();devices=[z for z in top if z['neuron_device'] in (12,13)]
  assert len(devices)==2 and all(not z['neuron_processes'] and z['logical_neuroncore_config']==2 for z in devices)
  assert time.time()<deadline-300 and available()>768*2**30
  d=C/stage;d.mkdir(exist_ok=False);tmp=Path(f'/dev/shm/dw25q{index}');tmp.mkdir(mode=0o700,exist_ok=False)
  sock=socket.socket(socket.AF_UNIX);sock.bind(str(tmp/'pymp-probe-listener-xxxxxxxxxxxxxxxxxxxxxxxx'));sock.close();(tmp/'pymp-probe-listener-xxxxxxxxxxxxxxxxxxxxxxxx').unlink()
  env=os.environ.copy()
  env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
  assert shutil.which('libneuronpjrt-path',path=env['PATH'])==str(Path(sys.executable).parent/'libneuronpjrt-path')
  for k in env_spec['unset']+['MALLOC_ARENA_MAX']:env.pop(k,None)
  env.update(env_spec['environment']);env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES='48-55',NEURONCORE_NUM_DEVICES='8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',MASTER_PORT=str(64151+2*index),NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{64152+2*index}',TMPDIR=str(tmp))
  for port in (64151+2*index,64152+2*index):
   with socket.socket() as s:s.bind(('127.0.0.1',port))
  command=['timeout','--signal=TERM','--kill-after=30s',str(max(1,int(deadline-time.time())))+'s',sys.executable,'-u','-m','train.partial_state_driver','--protocol',str(protocol),'--output',str(run_out),'--arm',arm,'--stop-file',str(d/'STOP'),'--deadline-utc','2026-10-19T22:00:00+00:00']
  if index==0:command.extend(['--halt-after','4'])
  elif index==1:command.extend(['--resume',str(out/'step-0000004'),'--attempt-id','actual-restart-001'])
  with (d/'run.log').open('xb') as log,(d/'host-samples.jsonl').open('x') as samples:
   proc=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=identity(proc.pid)
   assert owned and owned['pgid']==owned['session']==proc.pid
   write(d/'launch.json',dict(utc=utc(),identity=owned,command=command,environment={k:env[k] for k in list(env_spec['environment'])+['NEURON_RT_VISIBLE_CORES','NEURONCORE_NUM_DEVICES','MASTER_PORT','NEURON_RT_ROOT_COMM_ID','TMPDIR']},AF_UNIX_actual_bind_pass=True,prelaunch_topology=top))
   captured=False;stop=None
   while proc.poll() is None:
    procs=members(owned['session']);avail=available();sample=dict(utc=utc(),MemAvailable_bytes=avail,processes=[dict(**v,memory=memory(v['pid'])) for v in procs]);samples.write(json.dumps(sample)+'\n');samples.flush()
    if not captured:
     actual=topology();allocated=[v for z in actual if z['neuron_device'] in (12,13) for v in z['neuron_processes']]
     if len(allocated)==8:
      assert len({v['pid'] for v in allocated})==8 and {v['pid'] for v in allocated}<={v['pid'] for v in procs}
      write(d/'first-live.json',dict(utc=utc(),topology=actual,processes=procs,actual8_on_physical12_13=allocated));captured=True
    if stop is None and (avail<512*2**30 or min(shutil.disk_usage(z).free for z in (R,R/'local/nvme-study'))<35*2**30 or time.time()>=deadline):
     stop=dict(utc=utc(),sample=sample);write(d/'MEMORY_OR_DEADLINE_STOP.json',stop)
     if same(owned):os.kill(owned['pid'],signal.SIGTERM)
    if stop is not None and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>35:kill_remaining(owned)
    time.sleep(2)
   code=proc.wait();write(d/'exit.json',dict(utc=utc(),exit_code=code,external_stop=stop))
  if code or stop:
   cleanup=kill_remaining(owned);write(d/'failure.json',dict(status='FAILED_NO_RETRY',exit_code=code,exact_owned_cleanup_groups=cleanup));raise SystemExit('Qualification failed; original outputs retained, no retry')
  for _ in range(20):
   if not members(owned['session']):break
   time.sleep(1)
  assert captured and not members(owned['session']) and all(not z['neuron_processes'] for z in topology() if z['neuron_device'] in (12,13))
  completion=read(run_out/'completion.json');assert completion['actual_steps']==step and completion['status']==('complete' if index==1 else 'stopped_at_boundary')
  # Verify every original periodic checkpoint, not just the last snapshot.
  verified={}
  for target in sorted(set(range(2,step+1,2))|{step}):
   checkpoint=run_out/f'step-{target:07d}';manifest=read(checkpoint/'COMPLETE.json')
   assert manifest['step']==target and manifest['world_size']==8 and set(manifest['files'])=={'model.pt',*[f'rank-{r:05d}.pt' for r in range(8)]}
   for name,value in manifest['files'].items():assert (checkpoint/name).stat().st_size==value['bytes'] and sha(checkpoint/name)==value['sha256']
   verified[str(target)]=sha(checkpoint/'COMPLETE.json')
  write(d/'closure.json',dict(status='COMPLETE_PHYSICALLY_CLOSED_FULL_STATES_VERIFIED',utc=utc(),all_original_session_processes_absent=True,assigned_physical12_13_empty=True,actual_steps=step,checkpoint_manifests=verified,completion_sha256=sha(run_out/'completion.json')))
 write(C/'qualification-completion.json',dict(status='ALL_TWO_STAGES_CLOSED_PENDING_INDEPENDENT_AUDIT',utc=utc(),selection_steps=8))
