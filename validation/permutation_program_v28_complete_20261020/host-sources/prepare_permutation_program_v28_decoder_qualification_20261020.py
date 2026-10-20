"""Prepare a single bounded unchanged-B0 TRAIN decoder job on one logical core."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib,subprocess,shutil,importlib.util,py_compile
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-decode';C=R/'local/diagnostics/permutation-program-v28-TRAIN4-decode-qualification-v1';O=R/'local/nvme-cpt/evaluation/permutation-program-v28-B0-TRAIN4-decode-qualification-v1'
C.mkdir(exist_ok=False)
helper=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'
text='''"""One actual B0 TRAIN decoder job; finite resources and no automatic retry."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,os,time,socket,signal,shutil,importlib.util
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-decode'
C=Path(__file__).resolve().parent
helper=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'
spec=importlib.util.spec_from_file_location('program_owned_process_helpers',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
r=h.read(C/'release.json');assert r['supervisor_sha256']==h.sha(__file__) and r['helper_sha256']==h.sha(helper)
O=Path(r['output']);p=h.read(O/'protocol.json');assert h.sha(O/'protocol.json')==r['protocol_sha256']
assert all(h.sha(W/k)==v for k,v in p['source_files'].items())
assert all(not z['neuron_processes'] for z in h.topology() if z['neuron_device']==13) and h.available()>768*2**30
deadline=datetime.fromisoformat(r['deadline_utc']).timestamp();assert time.time()<deadline-120
s=h.read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
assert shutil.which('libneuronpjrt-path',path=env['PATH'])==str(Path(sys.executable).parent/'libneuronpjrt-path')
for k in s['unset']+['MALLOC_ARENA_MAX']:env.pop(k,None)
tmp=Path('/dev/shm/pp28decode1');tmp.mkdir(mode=0o700,exist_ok=False)
env.update(s['environment']);env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES='52',NEURONCORE_NUM_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',MASTER_PORT='63451',NEURON_RT_ROOT_COMM_ID='127.0.0.1:63452',TMPDIR=str(tmp))
for port in (63451,63452):
 with socket.socket() as sock:sock.bind(('127.0.0.1',port))
command=['timeout','--signal=TERM','--kill-after=30s',str(max(1,int(deadline-time.time())))+'s',sys.executable,'-u','-m','train.permutation_program_answer_driver','generate','--output',str(O),'--until',r['deadline_utc']]
with (C/'run.log').open('xb') as log,(C/'host-samples.jsonl').open('x') as samples:
 proc=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=h.identity(proc.pid);assert owned and owned['pgid']==owned['session']==proc.pid
 h.write(C/'launch.json',dict(utc=h.utc(),identity=owned,command=command,environment={k:env[k] for k in list(s['environment'])+['NEURON_RT_VISIBLE_CORES','NEURONCORE_NUM_DEVICES','MASTER_PORT','NEURON_RT_ROOT_COMM_ID','TMPDIR']}))
 captured=False;stop=None
 while proc.poll() is None:
  procs=h.members(owned['session']);avail=h.available();v=dict(utc=h.utc(),MemAvailable_bytes=avail,processes=[dict(**x,memory=h.memory(x['pid'])) for x in procs]);samples.write(json.dumps(v)+'\\n');samples.flush()
  if not captured:
   top=h.topology();allocated=[z for d in top if d['neuron_device']==13 for z in d['neuron_processes']]
   if allocated:
    assert len(allocated)==1 and allocated[0]['pid'] in {x['pid'] for x in procs}
    h.write(C/'first-live.json',dict(utc=h.utc(),topology=top,allocated=allocated));captured=True
  if stop is None and (avail<512*2**30 or time.time()>=deadline or min(shutil.disk_usage(d).free for d in (R,R/'local/nvme-cpt',R/'local/nvme-study'))<35*2**30):
   stop=dict(utc=h.utc(),sample=v);h.write(C/'STOP.json',stop)
   if h.same(owned):os.kill(owned['pid'],signal.SIGTERM)
  if stop and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>35:h.kill_remaining(owned)
  time.sleep(2)
 code=proc.wait();h.write(C/'exit.json',dict(utc=h.utc(),exit_code=code,external_stop=stop))
if code or stop:
 h.write(C/'failure.json',dict(status='FAILED_NO_RETRY',exit_code=code,exact_owned_cleanup_groups=h.kill_remaining(owned)));raise SystemExit('Decode qualification failed; retained, no retry')
for _ in range(20):
 if not h.members(owned['session']):break
 time.sleep(1)
assert captured and not h.members(owned['session']) and all(not z['neuron_processes'] for z in h.topology() if z['neuron_device']==13)
report=h.read(O/'answers/run.json');assert report['status']=='complete_all_attempts' and report['rows']==28
h.write(C/'closure.json',dict(status='COMPLETE_PHYSICALLY_CLOSED_PENDING_INDEPENDENT_CHECK',utc=h.utc(),all_original_session_processes_absent=True,assigned_physical13_empty=True,attempts=28,run_sha256=h.sha(O/'answers/run.json')))
'''
(C/'supervise.py').write_text(text);py_compile.compile(str(C/'supervise.py'),doraise=True)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
p=json.loads((O/'protocol.json').read_text());assert p['purpose']=='TRAIN_PROGRAM_DECODER_MECHANICS_ONLY' and p['items']==4
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
now=datetime.now(timezone.utc)
r=dict(status='RELEASED_BOUNDED_B0_TRAIN_DECODER_MECHANICS_ONLY',utc=now.isoformat(),source_root=str(W),source_commit='24ef45f',protocol_sha256=sha(O/'protocol.json'),supervisor_sha256=sha(C/'supervise.py'),helper_sha256=sha(helper),output=str(O),physical_devices=[13],logical_cores='52',deadline_utc=(now+timedelta(minutes=30)).isoformat(),maximum_reserved_chip_hours=.55,scope='Only28unchanged-B0 attempts on4fixed TRAIN questions. Seven cache/ordinary-prefill checks including supplied pauses. No learned checkpoint, held outcome or science result.',no_automatic_retry=True)
with (C/'release.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
