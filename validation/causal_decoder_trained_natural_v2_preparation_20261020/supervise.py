"""One spare-core engineering probe, finite bounds and owned-process cleanup."""
from datetime import datetime, timezone
from pathlib import Path
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time

R = Path('/home/ubuntu/dario')
W = R / 'local/worktrees/causal-decoder-batching-v1'
C = R / 'local/diagnostics/causal-decoder-trained-natural-v2'
HELPER = R / 'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'
spec = importlib.util.spec_from_file_location('owned_process_helpers', HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def main():
    plan = h.read(C/'plan.json')
    release = h.read(C/'release.json')
    assert release['status'] == 'RELEASED_ONE_SPARE_CORE_TRAIN_ENGINEERING_ONLY'
    assert release['plan_sha256'] == h.sha(C/'plan.json')
    assert release['supervisor_sha256'] == h.sha(__file__)
    assert plan['physical_device'] == 8 and plan['logical_core'] == 32
    assert not next(v for v in h.topology() if v['neuron_device'] == 8)['neuron_processes']
    assert h.available() > 768*2**30
    assert min(shutil.disk_usage(p).free for p in (R,R/'local/nvme-cpt',R/'local/nvme-study')) > 35*2**30
    tmp = Path('/dev/shm/bdcnat2_32')
    tmp.mkdir(mode=0o700, exist_ok=False)
    probe = tmp/'pymp-probe-listener-xxxxxxxxxxxxxxxxxxxxxxxx'
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(probe))
    probe.unlink()
    for port in (65250,65251):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',port))
    env = os.environ.copy()
    template = h.read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
    for key in template['unset']+['MALLOC_ARENA_MAX']:
        env.pop(key,None)
    env.update(template['environment'])
    env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),
        NEURON_RT_VISIBLE_CORES='32',NEURONCORE_NUM_DEVICES='1',TMPDIR=str(tmp),
        MASTER_PORT='65250',NEURON_RT_ROOT_COMM_ID='127.0.0.1:65251',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256')
    started = time.monotonic()
    command = ['timeout','--signal=TERM','--kill-after=30s','1620s',sys.executable,'-u',
               str(R/'local/analysis/qualify_causal_decoder_trained_natural_v2_20261020.py'),'run']
    with (C/'worker.log').open('xb') as log:
        process = subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned = h.identity(process.pid)
        h.write(C/'launch.json',dict(started_utc=h.utc(),identity=owned,command=command,
            supervisor_sha256=h.sha(__file__),plan_sha256=h.sha(C/'plan.json'),
            physical_device=8,logical_core=32,maximum_reserved_chip_hours=.5))
        observed = False
        stopped = None
        try:
            with (C/'host-samples.jsonl').open('x') as samples:
                while process.poll() is None:
                    members = h.members(owned['session'])
                    sample = dict(utc=h.utc(),MemAvailable_bytes=h.available(),
                        root_free_bytes=shutil.disk_usage(R).free,processes=members)
                    samples.write(json.dumps(sample)+'\n');samples.flush()
                    if not observed:
                        actual = next(v for v in h.topology() if v['neuron_device']==8)['neuron_processes']
                        if actual:
                            assert {v['pid'] for v in actual}<={v['pid'] for v in members}
                            h.write(C/'first-live.json',dict(utc=h.utc(),actual=actual))
                            observed = True
                    if stopped is None and (sample['MemAvailable_bytes']<512*2**30 or sample['root_free_bytes']<35*2**30 or time.monotonic()-started>1650):
                        stopped = time.monotonic()
                        h.write(C/'STOP.json',sample)
                        if h.same(owned):os.kill(owned['pid'],15)
                    if stopped is not None and time.monotonic()-stopped>30:
                        h.kill_remaining(owned)
                    time.sleep(5)
        except BaseException:
            h.kill_remaining(owned)
            raise
    code = process.wait()
    for _ in range(20):
        if not h.members(owned['session']):break
        time.sleep(1)
    empty = not next(v for v in h.topology() if v['neuron_device']==8)['neuron_processes']
    absent = not h.members(owned['session'])
    hours = (time.monotonic()-started)/3600
    closure = dict(utc=h.utc(),exit_code=code,actual_owned_device_observed=observed,
        assigned_physical_device_empty=empty,all_original_session_processes_absent=absent,
        reserved_chip_hours=hours,host_bounds_stop=stopped is not None)
    h.write(C/'closure.json',closure)
    assert code==0 and observed and empty and absent and hours<=.5 and stopped is None
    assert h.read(C/'worker-result.json')['status']=='PASS_TRAINED_NATURAL_STOP_EXACT_BATCH_PARITY'
    print(json.dumps(dict(status='PASS_BOUNDED_TRAIN_ONLY_ENGINEERING_PROBE_PHYSICALLY_CLOSED',**closure)))


if __name__ == '__main__':
    main()
