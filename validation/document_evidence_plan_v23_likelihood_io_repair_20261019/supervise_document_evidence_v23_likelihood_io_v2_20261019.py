"""Four frozen full-span likelihood shards on one empty physical chip;no readout/retry."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,shutil,socket,subprocess,sys,time
R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-evidence-span-v23-io-v2'
helper=R/'local/diagnostics/document-evidence-plan-v23-world8-qualification-v1/qualify.py';spec=importlib.util.spec_from_file_location('qa_owned_process_helpers',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
O=Path(sys.argv[1]).resolve();physical=int(sys.argv[2]);C=Path(sys.argv[3]).resolve();assert len(sys.argv)==4
p=h.read(O/'protocol.json');release=h.read(p['release_file']);assert release['status']=='RELEASED_DOCUMENT_EVIDENCE_V23_PILOT_AND_FIXED_EVALUATION'
repair=h.read(R/'local/diagnostics/document-evidence-plan-v23-pilot-v1/likelihood-io-repair-v2/release.json');assert repair['status']=='RELEASED_EXPLICIT_IO_ONLY_LIKELIHOOD_REPAIR_V2' and repair['controller_sha256']==h.sha(__file__) and repair['protocols'][p['model']['id']]==h.sha(O/'protocol.json')
assert p['release_sha256']==h.sha(p['release_file']) and p['schema']=='document-evidence-plan-v23-pointer-replay-likelihood' and p['shards']==4
worker=W/'train/document_evidence_plan_likelihood.py';assert h.sha(worker)==p['source_files']['train/document_evidence_plan_likelihood.py']
assert all(h.sha(W/n)==v for n,v in p['source_files'].items()) and all(h.sha(O/n)==v for n,v in p['files'].items())
assert p['model']['id'] in release['model_specs'] and sum(v['rows'] for v in p['inventories'].values())==1531
assert not next(z for z in h.topology() if z['neuron_device']==physical)['neuron_processes'] and h.available()>768*2**30
C.mkdir(parents=True,exist_ok=False);deadline=datetime.fromisoformat('2026-10-20T06:00:00+00:00').timestamp();envspec=h.read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
h.write(C/'plan.json',dict(scope='FROZEN_LIKELIHOOD_ONLY_NO_CAPABILITY_READOUT',utc=h.utc(),protocol_sha256=h.sha(O/'protocol.json'),release_sha256=p['release_sha256'],physical_device=physical,logical_cores=list(range(physical*4,physical*4+4)),source_sha256=h.sha(__file__),helper_sha256=h.sha(helper)))
children={};logs=[]
for shard in range(4):
 d=C/f'shard-{shard}';d.mkdir();tmp=Path(f'/dev/shm/de23l{physical}_{shard}_{h.sha(O/"protocol.json")[:10]}');tmp.mkdir(mode=0o700,exist_ok=False)
 port=61000+physical*100+shard*10
 for value in (port,port+1):
  with socket.socket() as s:s.bind(('127.0.0.1',value))
 env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
 for k in envspec['unset']+['MALLOC_ARENA_MAX']:env.pop(k,None)
 env.update(envspec['environment']);env.update(PYTHONPATH=str(W)+':'+str(R/'local/analysis')+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES=str(physical*4+shard),NEURONCORE_NUM_DEVICES='1',TMPDIR=str(tmp),MASTER_PORT=str(port),NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{port+1}',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256')
 assert shutil.which('libneuronpjrt-path',path=env['PATH'])==str(Path(sys.executable).parent/'libneuronpjrt-path')
 command=['timeout','--signal=TERM','--kill-after=30s',str(int(deadline-time.time()))+'s',sys.executable,'-u','-m','train.document_evidence_plan_likelihood','run','--output',str(O),'--shard',str(shard),'--device','xla']
 log=(d/'run.log').open('xb');logs.append(log);proc=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=h.identity(proc.pid);assert owned['session']==owned['pgid']==proc.pid
 h.write(d/'launch.json',dict(utc=h.utc(),identity=owned,command=command,logical_core=physical*4+shard));children[shard]=dict(process=proc,owned=owned,directory=d)
captured=False;stop=None
with (C/'host-samples.jsonl').open('x') as samples:
 while any(v['process'].poll() is None for v in children.values()):
  members=[v for child in children.values() for v in h.members(child['owned']['session'])];avail=h.available()
  sample=dict(utc=h.utc(),MemAvailable_bytes=avail,processes=[dict(**v,memory=h.memory(v['pid'])) for v in members]);samples.write(json.dumps(sample)+'\n');samples.flush()
  if not captured:
   actual=h.topology();assigned=next(z for z in actual if z['neuron_device']==physical)['neuron_processes']
   if len(assigned)==4:
    assert {v['pid'] for v in assigned}<={v['pid'] for v in members};h.write(C/'first-live.json',dict(utc=h.utc(),actual4=assigned,topology=actual));captured=True
  failed=any(v['process'].poll() not in (None,0) for v in children.values())
  if stop is None and (avail<512*2**30 or min(shutil.disk_usage(root).free for root in [R,R/'local/nvme-cpt'])<35*2**30 or time.time()>=deadline or failed):
   stop=dict(utc=h.utc(),sample=sample,failed_peer=failed);h.write(C/'STOP_REASON.json',stop)
   for child in children.values():
    if h.same(child['owned']):os.kill(child['owned']['pid'],15)
  if stop is not None and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>35:
   for child in children.values():h.kill_remaining(child['owned'])
  time.sleep(10)
for log in logs:log.close()
for shard,child in children.items():h.write(child['directory']/'exit.json',dict(utc=h.utc(),exit_code=child['process'].wait()))
if stop or any(v['process'].returncode for v in children.values()):raise SystemExit('Likelihood group failed;retain partial attempts,no automatic retry')
for _ in range(20):
 if not any(h.members(v['owned']['session']) for v in children.values()):break
 time.sleep(1)
assert captured and not any(h.members(v['owned']['session']) for v in children.values())
assert not next(v for v in h.topology() if v['neuron_device']==physical)['neuron_processes']
files={}
for shard in range(4):
 d=O/'scores'/f'shard-{shard}';run=h.read(d/'completion.json')
 assert run['status']=='COMPLETE_ALL_POINTER_REPLAY_LIKELIHOOD_ATTEMPTS_NOT_READOUT' and run['rows']==sum(len(range(shard,v['rows'],4)) for v in p['inventories'].values()) and run['synthetic_execution_rows']==(5 if shard==3 else 0) and run['execution_errors']==0 and run['source_protocol_sha256']==h.sha(O/'protocol.json') and run['scores_sha256']==h.sha(d/'scores.jsonl')
 calibration=h.read(d/'own-model-TRAIN-calibration.json');assert set(calibration)=={'QA','ordinary'} and all(v['passed'] and v['max_token_error']<=.001 and v['absolute_mean_span_error']<=.001 for v in calibration.values())
 files[str(d/'scores.jsonl')]=run['scores_sha256']
h.write(C/'closure.json',dict(status='ALL_FROZEN_LIKELIHOOD_SHARDS_PHYSICALLY_CLOSED_NO_READOUT',utc=h.utc(),actual_rows=1531,execution_errors=0,scores=files,assigned_physical_device_empty=True))
