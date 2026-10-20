"""One declared eight-update mechanics run, halt4 then fresh-process resume8.

No automatic retry or scientific run. Reuse only the old supervisor's pure
process/topology/readback utilities, not its historical main/launcher.
"""
from pathlib import Path
from datetime import datetime
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time

from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from train import partial_state_driver as driver
from train.neuron_driver import verify_checkpoint

R=Path('/home/ubuntu/dario')
W=R/'local/worktrees/real-cpt-scaffold-v29'
C=R/'local/diagnostics/real-cpt-scaffold-v29-qualification-v1'
helper=R/'local/diagnostics/real-source-balanced-v17-world8-target640-qualification-v2/qualify.py'
spec=importlib.util.spec_from_file_location('frozen_process_utilities_only',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
release=json.loads((C/'release.json').read_text())
p=json.loads((C/'protocol.json').read_text());driver.validate(p)
assert release['status']=='RELEASED_BOUNDED_TRAIN_MECHANICS_ONLY'
assert release['protocol_sha256']==sha256_file(C/'protocol.json')
assert release['supervisor_sha256']==sha256_file(__file__)
assert release['helper_sha256']==sha256_file(helper)
assert all(sha256_file(W/k)==v for k,v in p['source_files'].items())
assert sha256_file(p['initial_loss_reference_file'])==p['initial_loss_reference_sha256']
reference=json.loads(Path(p['initial_loss_reference_file']).read_text())
assert reference['data_sha256']==p['data_sha256'] and reference['source_files']==p['source_files']
deadline=datetime.fromisoformat(release['deadline_utc']).timestamp()
out=Path(release['output'])
assert out.parent.resolve().is_relative_to(Path('/dev/shm').resolve())
assert not out.exists() and h.available()>768*2**30
envspec=json.loads((R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json').read_text())
stage_costs=[]
for index,(stage,terminal) in enumerate((('fresh4',4),('resume8',8))):
    d=C/stage;d.mkdir(exist_ok=False)
    top=h.topology();assigned=[z for z in top if z['neuron_device'] in (0,1)]
    assert len(assigned)==2 and all(not z['neuron_processes'] and z['logical_neuroncore_config']==2 for z in assigned)
    assert time.time()<deadline-180 and h.available()>768*2**30
    assert shutil.disk_usage(out.parent).free>100*2**30
    assert min(shutil.disk_usage(q).free for q in (R,R/'local/nvme-cpt'))>35*2**30
    tmp=Path(f'/dev/shm/rc29q{index}');tmp.mkdir(mode=0o700,exist_ok=False)
    socket_path=tmp/'probe-xxxxxxxxxxxxxxxxxxxxxxxx'
    with socket.socket(socket.AF_UNIX) as sock:sock.bind(str(socket_path))
    socket_path.unlink()
    port=53290+index*2
    for value in (port,port+1):
        with socket.socket() as sock:sock.bind(('127.0.0.1',value))
    env=os.environ.copy()
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    for key in envspec['unset']+['MALLOC_ARENA_MAX']:env.pop(key,None)
    env.update(envspec['environment'])
    env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES='0-7',
        NEURONCORE_NUM_DEVICES='8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',
        MASTER_PORT=str(port),NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{port+1}',TMPDIR=str(tmp))
    assert {k:env.get(k) for k in p['runtime_policy']['environment']}==p['runtime_policy']['environment']
    cmd=['timeout','--signal=TERM','--kill-after=180s',str(int(deadline-time.time()))+'s',
        sys.executable,'-u','-m','train.partial_state_driver','--protocol',str(C/'protocol.json'),
        '--output',str(out),'--arm',driver.SCAFFOLD_QUALIFICATION_ARM,
        '--stop-file',str(d/'STOP'),'--deadline-utc',release['deadline_utc']]
    if index==0:cmd+=['--halt-after','4']
    else:cmd+=['--resume',str(out/'step-0000004'),'--attempt-id','resume8']
    started=time.monotonic()
    with (d/'run.log').open('xb') as log,(d/'host-samples.jsonl').open('x') as samples:
        proc=subprocess.Popen(cmd,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned=h.identity(proc.pid);assert owned and owned['session']==owned['pgid']==proc.pid
        atomic_json(d/'launch.json',dict(utc=utc_now(),identity=owned,command=cmd,AF_UNIX_bind_pass=True))
        captured=False;stop=None
        while proc.poll() is None:
            members=h.members(owned['session'])
            sample=dict(utc=utc_now(),MemAvailable_bytes=h.available(),
                filesystem_free_bytes={str(q):shutil.disk_usage(q).free for q in (R,R/'local/nvme-cpt',out.parent)},
                processes=[dict(**v,memory=h.memory(v['pid'])) for v in members])
            samples.write(json.dumps(sample)+'\n');samples.flush()
            if not captured:
                active=[v for z in h.topology() if z['neuron_device'] in (0,1) for v in z['neuron_processes']]
                if len(active)==8:
                    assert {v['pid'] for v in active}<={v['pid'] for v in members}
                    atomic_json(d/'first-live.json',dict(utc=utc_now(),actual8=active));captured=True
            if stop is None and (sample['MemAvailable_bytes']<512*2**30
                    or min(sample['filesystem_free_bytes'].values())<35*2**30 or time.time()>=deadline):
                stop=dict(utc=utc_now(),reason='Declared memory/disk/deadline boundary',sample=sample)
                atomic_json(d/'STOP',stop)
            if stop is not None and time.time()-datetime.fromisoformat(stop['utc']).timestamp()>180:
                h.kill_remaining(owned)
            time.sleep(10)
        code=proc.wait()
    atomic_json(d/'exit.json',dict(utc=utc_now(),exit_code=code,external_stop=stop))
    if code or stop:
        groups=h.kill_remaining(owned)
        atomic_json(d/'failure.json',dict(status='FAILED_NO_RETRY',exit_code=code,cleanup_groups=groups))
        raise SystemExit('Preserve original evidence and full states; no automatic retry')
    for _ in range(30):
        if not h.members(owned['session']):break
        time.sleep(1)
    assert captured and not h.members(owned['session'])
    assert all(not z['neuron_processes'] for z in h.topology() if z['neuron_device'] in (0,1))
    seconds=time.monotonic()-started
    stage_costs.append(dict(stage=stage,wall_to_physical_closure_seconds=seconds,reserved_chip_hours=2*seconds/3600))
    checkpoint=out/f'step-{terminal:07d}'
    manifest=verify_checkpoint(checkpoint,canonical_hash(p),8)
    metrics=[json.loads(line) for line in (out/'metrics.jsonl').open()]
    assert [m['step'] for m in metrics]==list(range(1,terminal+1))
    assert all(m['gradient_norm']>0 for m in metrics)
    if index==0:
        actual=metrics[0];components=actual['loss_components']
        errors={name:abs(components[name+'_mean_loss']-value) for name,value in reference['global_means'].items()}
        loss_error=abs(actual['loss']-reference['global_total_loss'])
        bounds=reference['comparisons_fixed_before_device_calls']
        passed=max(errors.values())<=bounds['max_component_mean_abs_error'] and loss_error<=bounds['max_total_abs_error']
        assert all(components[name+'_labels']==value for name,value in reference['global_counts'].items())
        atomic_json(d/'initial-loss-check.json',dict(passed=passed,errors=errors,total_error=loss_error,
            bounds=bounds,CPU_reference_sha256=p['initial_loss_reference_sha256'],actual_metric=actual))
        assert passed,'Actual native first-step component loss differs from frozen CPU BF16 reference'
    else:
        restored=[json.loads((out/'recoveries/resume8'/f'restored-rank-{rank:05d}.json').read_text()) for rank in range(8)]
        assert all(v['full_active_model_bytes_equal'] and v['full_optimizer_bytes_equal']
                   and v['tied_parameter_objects_preserved'] for v in restored)
    atomic_json(d/'closure.json',dict(status='PHYSICALLY_CLOSED_FULL_STATE_READBACK',utc=utc_now(),
        terminal=terminal,manifest_sha256=sha256_file(checkpoint/'COMPLETE.json'),cost=stage_costs[-1]))
atomic_json(C/'hardware-complete.json',dict(status='EIGHT_ACTUAL_UPDATES_AND_FULL_FRESH_PROCESS_RESTORE_COMPLETE',
    utc=utc_now(),output=str(out),costs=stage_costs,total_reserved_chip_hours=sum(v['reserved_chip_hours'] for v in stage_costs),
    remaining='Independent audit and full terminal EBS protection before publishing qualification pass; no science release'))
