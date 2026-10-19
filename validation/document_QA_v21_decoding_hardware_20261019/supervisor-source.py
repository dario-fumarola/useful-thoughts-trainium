"""Real-context world8 mechanics: actual full-state save/restart. No retry."""
import hashlib,json,os,signal,socket,subprocess,sys,time,shutil
from datetime import datetime,timezone
from pathlib import Path
C=Path(__file__).resolve().parent;R=Path('/home/ubuntu/dario');W=R/'local/worktrees/document-qa-eval-v21'
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
 release=read(C/'release.json');output=C/'execution';p=read(output/'protocol.json');deadline=datetime.fromisoformat('2026-10-19T17:00:00+00:00').timestamp()
 assert release['status']=='RELEASED_BOUNDED_TRAIN_DECODING_MECHANICS_ONLY' and release['supervisor_sha256']==sha(__file__)
 assert release['protocol_sha256']==sha(output/'protocol.json') and all(sha(W/n)==v for n,v in p['source_files'].items())
 top=topology();device=next(v for v in top if v['neuron_device']==14);assert not device['neuron_processes'] and device['logical_neuroncore_config']==2
 assert available()>768*2**30
 spec=read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
 env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
 for k in spec['unset']+['MALLOC_ARENA_MAX']:env.pop(k,None)
 env.update(spec['environment']);tmp=Path('/dev/shm/dq21evalq');tmp.mkdir(mode=0o700,exist_ok=False)
 env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES='56',NEURONCORE_NUM_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',TMPDIR=str(tmp),MASTER_PORT='65460',NEURON_RT_ROOT_COMM_ID='127.0.0.1:65461')
 assert shutil.which('libneuronpjrt-path',path=env['PATH'])==str(Path(sys.executable).parent/'libneuronpjrt-path')
 for port in (65460,65461):
  with socket.socket() as sock:sock.bind(('127.0.0.1',port))
 command=['timeout','--signal=TERM','--kill-after=30s','5400s',sys.executable,'-u','-m','train.document_qa_evaluate','generate','--output',str(output),'--shard','0','--until','2026-10-19T17:00:00+00:00']
 with (C/'run.log').open('xb') as log,(C/'host-samples.jsonl').open('x') as samples:
  proc=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);owned=identity(proc.pid);assert owned['session']==owned['pgid']==proc.pid
  write(C/'launch.json',dict(utc=utc(),identity=owned,command=command,logical_core=56,physical_device=14,prelaunch_topology=top))
  captured=False;stop=None
  while proc.poll() is None:
   procs=members(owned['session']);avail=available();sample=dict(utc=utc(),MemAvailable_bytes=avail,processes=[dict(**v,memory=memory(v['pid'])) for v in procs]);samples.write(json.dumps(sample)+'\n');samples.flush()
   if not captured:
    actual=topology();assigned=next(v for v in actual if v['neuron_device']==14)['neuron_processes']
    if len(assigned)==1:
     assert assigned[0]['pid'] in {v['pid'] for v in procs};write(C/'first-live.json',dict(utc=utc(),topology=actual,actual1_on_physical14=assigned));captured=True
   if stop is None and (avail<512*2**30 or time.time()>=deadline):
    stop=dict(utc=utc(),sample=sample);write(C/'MEMORY_OR_DEADLINE_STOP.json',stop)
    if same(owned):os.kill(owned['pid'],signal.SIGTERM)
   if stop is not None and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>35:kill_remaining(owned)
   time.sleep(2)
  code=proc.wait();write(C/'exit.json',dict(utc=utc(),exit_code=code,external_stop=stop))
 if code or stop:
  cleanup=kill_remaining(owned);write(C/'failure.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',exit_code=code,owned_cleanup_groups=cleanup));raise SystemExit(1)
 for _ in range(20):
  if not members(owned['session']):break
  time.sleep(1)
 assert captured and not members(owned['session']) and not next(v for v in topology() if v['neuron_device']==14)['neuron_processes']
 d=output/'answers/shard-0';report=read(d/'run.json');assert report['status']=='complete_all_attempts' and report['rows']==6 and report['attempts_sha256']==sha(d/'attempts.jsonl')
 attempts=[json.loads(v) for v in (d/'attempts.jsonl').read_text().splitlines()];assert len(attempts)==6 and all(v['status']=='ok' for v in attempts)
 write(C/'closure.json',dict(status='SIX_TRAIN_ATTEMPTS_PHYSICALLY_CLOSED_PENDING_INDEPENDENT_AUDIT',utc=utc(),actual_attempts=6,all_original_session_processes_absent=True,physical14_empty=True,attempts_sha256=sha(d/'attempts.jsonl')))
