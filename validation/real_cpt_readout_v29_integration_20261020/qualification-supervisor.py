"""Finite TRAIN-only checks of the four V29 readout worker entrypoints.

One spare physical chip, four serial processes, no learning or held rows.
Only process/topology utility functions are imported from an old qualification;
its launcher is never executed. Original evidence and cache remain untouched.
"""
from datetime import datetime,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from weekend_thinking.artifacts import atomic_json,sha256_file

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/real-cpt-scaffold-v29-eval'
C=R/'local/diagnostics/real-cpt-readout-v29-qualification-v1'
S=R/'local/diagnostics/real-cpt-scaffold-v29-science-v1'
read=lambda p:json.loads(Path(p).read_text())
reference=read(C/'reference/reference.json')
assert reference['purpose']=='engineering-train-only'
release=read(S/'release-terminal1280-seed2026102029.json')
helper=Path(release['process_helper']['path']);assert sha256_file(helper)==release['process_helper']['sha256']
spec=importlib.util.spec_from_file_location('owned_process_utilities_only',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
assert all(sha256_file(W/k)==v for k,v in reference['source_files'].items())
assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
plan=dict(status='RELEASED_ONE_SPARE_CHIP_TRAIN_ONLY_INTEGRATION',utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha256_file(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),
    reference_sha256=sha256_file(C/'reference/reference.json'),helper=release['process_helper'],
    panels=['writer','math','ordinary','scalar'],logical_core=32,physical_chip=8,
    max_wall_seconds_per_panel=1800,max_reserved_chip_hours=2,
    CPU_checks=1928,max_native_logp_error=.001,max_bidirectional_KL=.01,
    scope='Actual frozen V17 terminal weights and fixed TRAIN only; no held rows or parameter updates.',
    limits='Numerical integration qualification, not scientific validation or proof of learned useful computation.')
atomic_json(C/'plan.json',plan)
template=read(release['environment_template']);env=os.environ.copy()
for key in template['unset']:env.pop(key,None)
env.update(template['environment']);env.update(reference['runtime_policy']['environment'])
for key,value in list(env.items()):
    if value is None:env.pop(key)
env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),NEURON_RT_VISIBLE_CORES='32',
    NEURONCORE_NUM_DEVICES='1',NEURON_LOGICAL_NC_CONFIG='2',NEURON_RT_CONTIGUOUS_SCRATCHPAD='0',
    NEURON_COMPILE_CACHE_URL=str(R/'local/neuron-cache-driver'),USE_TORCH_XLA='0')
root_temp=Path('/dev/shm/cpt29-readout-q1');root_temp.mkdir(mode=0o700,exist_ok=False)
total=0.;outcomes={}
# Avoid overlapping new compilations and a large EBS checkpoint archive.
with (S/'large-EBS-writer.lock').open('a') as writer_lock:
    fcntl.flock(writer_lock,fcntl.LOCK_EX)
    for panel in plan['panels']:
        assert all(not r['neuron_processes'] and r['logical_neuroncore_config']==2 for r in h.topology() if r['neuron_device']==8)
        assert h.available()>512*2**30
        assert min(shutil.disk_usage(p).free for p in (R,R/'local/nvme-cpt'))>35*2**30
        d=C/panel;temporary=root_temp/panel;temporary.mkdir(mode=0o700)
        current=env.copy();current['TMPDIR']=str(temporary)
        deadline=datetime.fromtimestamp(time.time()+1800,tz=timezone.utc).isoformat()
        command=['timeout','--signal=TERM','--kill-after=60s','1800s',sys.executable,'-u',
            '-m','train.real_cpt_readout_driver','run','--reference',str(C/'reference'),
            '--output',str(d),'--panel',panel,'--qualify','--deadline-utc',deadline]
        started=time.monotonic();failure=None;observed=False
        with (C/f'{panel}.log').open('xb') as log:
            process=subprocess.Popen(command,cwd=W,env=current,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owned=h.identity(process.pid);assert owned and owned['session']==owned['pgid']==process.pid
            atomic_json(C/f'{panel}-launch.json',dict(command=command,owned=owned,utc=datetime.now(timezone.utc).isoformat()))
            try:
                while process.poll() is None:
                    members=h.members(owned['session'])
                    actual=[v for r in h.topology() if r['neuron_device']==8 for v in r['neuron_processes']]
                    if actual:
                        assert len(actual)==1 and {r['pid'] for r in actual}<={r['pid'] for r in members}
                        if not observed:atomic_json(C/f'{panel}-first-live.json',dict(actual=actual,owned=owned));observed=True
                    assert h.available()>512*2**30
                    assert min(shutil.disk_usage(p).free for p in (R,R/'local/nvme-cpt'))>35*2**30
                    assert total+(time.monotonic()-started)/3600<2
                    time.sleep(10)
            except BaseException as error:
                failure=repr(error);h.kill_remaining(owned)
        code=process.wait()
        for _ in range(30):
            if not h.members(owned['session']):break
            time.sleep(1)
        if h.members(owned['session']):h.kill_remaining(owned)
        absent=not h.members(owned['session'])
        idle=all(not r['neuron_processes'] for r in h.topology() if r['neuron_device']==8)
        hours=(time.monotonic()-started)/3600;total+=hours
        closed=dict(status='SESSION_AND_PHYSICAL_DEVICE_CLOSED' if absent and idle else 'CLOSURE_FAILED',
            exit_code=code,observed_actual_single_process=observed,owned_session_absent=absent,
            device_idle=idle,reserved_chip_hours=hours,failure=failure)
        atomic_json(C/f'{panel}-closure.json',closed);outcomes[panel]=closed
        assert code==0 and observed and absent and idle and failure is None
        assert read(d/'run.json')['status']=='PASS_FIXED_TRAIN_NATIVE_AND_PANEL_MECHANICS'
        print('CLOSED_PASS',panel,hours,flush=True)
atomic_json(C/'complete.json',dict(status='PASS_FOUR_ACTUAL_TRAIN_ONLY_INTEGRATED_READOUT_WORKERS',
    plan_sha256=sha256_file(C/'plan.json'),outcomes=outcomes,reserved_chip_hours=total,
    limits=plan['limits']))
print('PASS_ALL_FOUR_TRAIN_ONLY_PANELS',total,flush=True)
