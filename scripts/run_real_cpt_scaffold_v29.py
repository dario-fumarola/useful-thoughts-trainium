"""Supervise one explicitly released V29 stage of the existing causal engine.

No automatic retry, checkpoint selection, scientific scoring or training math.
The release binds the job, resources, helper source, protocol and this supervisor.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from train import partial_state_driver as driver
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--phase',choices=('stage64','terminal1280'),required=True)
    args=parser.parse_args();C=args.control;release=json.loads(args.release.read_text())
    assert release['supervisor_sha256']==sha256_file(__file__)
    assert release['source_files']==driver.source_files()
    job=release['jobs'][args.job];protocol=Path(job['protocol']);p=json.loads(protocol.read_text())
    assert sha256_file(protocol)==job['protocol_sha256']
    assert job['seed'] in release['released_seeds']
    driver.validate(p);driver.verify_scaffold_science_release(p)
    output=Path(job['output']);core_ids=tuple(range(job['slot']*8,job['slot']*8+8))
    devices=tuple(sorted({v//4 for v in core_ids}))
    assert p['visible_cores']==f'{core_ids[0]}-{core_ids[-1]}' and len(devices)==2
    helper=Path(release['process_helper']['path']);assert sha256_file(helper)==release['process_helper']['sha256']
    spec=importlib.util.spec_from_file_location('owned_process_utilities_only',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    stage=C/'jobs'/args.job/args.phase;stage.mkdir(parents=True,exist_ok=False)
    topology=h.topology()
    assert all(not row['neuron_processes'] and row['logical_neuroncore_config']==2
               for row in topology if row['neuron_device'] in devices)
    assert h.available()>768*2**30
    root=Path(release['repository_root']);source=Path(release['source_root'])
    cache=Path(release['cache_filesystem']);assert min(shutil.disk_usage(q).free for q in (root,cache))>35*2**30
    assert output.parent.resolve().is_relative_to(Path('/dev/shm'))
    assert shutil.disk_usage(output.parent).free>100*2**30
    phase_seconds=release['phase_wall_seconds'][args.phase]
    deadline=datetime.fromtimestamp(time.time()+phase_seconds,tz=timezone.utc).isoformat()
    temporary=Path('/dev/shm')/f'cpt29_{job["seed"]%100}_{job["slot"]}_{args.phase}'
    temporary.mkdir(mode=0o700,exist_ok=False)
    probe=temporary/'pymp-probe-xxxxxxxxxxxxxxxx'
    with socket.socket(socket.AF_UNIX) as s:s.bind(str(probe))
    probe.unlink()
    port=53700+(job['seed']-2026102029)*100+job['slot']*10+(4 if args.phase=='terminal1280' else 0)
    for value in (port,port+1):
        with socket.socket() as s:s.bind(('127.0.0.1',value))
    template=json.loads(Path(release['environment_template']).read_text());env=os.environ.copy()
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    for key in template['unset']+['MALLOC_ARENA_MAX']:env.pop(key,None)
    env.update(template['environment'])
    env.update(PYTHONPATH=str(source)+':'+str(root/'local/test-deps'),
        NEURON_RT_VISIBLE_CORES=p['visible_cores'],NEURONCORE_NUM_DEVICES='8',
        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XLA_COMPILATION_CACHE_SIZE='256',
        MASTER_PORT=str(port),NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{port+1}',TMPDIR=str(temporary))
    assert {k:env.get(k) for k in p['runtime_policy']['environment']}==p['runtime_policy']['environment']
    command=['timeout','--signal=TERM','--kill-after=180s',str(phase_seconds)+'s',sys.executable,'-u',
        '-m','train.partial_state_driver','--protocol',str(protocol),'--output',str(output),
        '--arm',job['arm'],'--stop-file',str(stage/'STOP'),'--deadline-utc',deadline,
        '--execution-release',str(args.release)]
    if args.phase=='stage64':command+=['--halt-after','64']
    else:command+=['--resume',job['resume_checkpoint'],'--attempt-id',f'terminal{job["seed"]%100}']
    identity_args=argparse.Namespace(execution_release=args.release,arm=job['arm'],output=output,
        protocol=protocol,resume=Path(job['resume_checkpoint']) if args.phase!='stage64' else None,
        halt_after=64 if args.phase=='stage64' else None)
    driver.validate_scaffold_execution(p,identity_args)
    key=args.job+'/'+args.phase;ledger_path=C/'reservations.json'

    def ledger_update(update=None):
        with (C/'reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            if update=='start':
                assert key not in ledger
                ledger[key]=dict(status='active',started_monotonic=time.monotonic(),physical_chips=2)
            elif isinstance(update,dict):ledger[key].update(update)
            if update is not None:atomic_json(ledger_path,ledger,overwrite=ledger_path.exists())
            return sum(v.get('reserved_chip_hours',2*(time.monotonic()-v['started_monotonic'])/3600)
                       for v in ledger.values())

    assert ledger_update()<release['science_reserved_chip_hours']-1
    ledger_update('start');started=time.monotonic();observed=False;first_checked=False;stop=None;failure=None
    with (stage/'run.log').open('xb') as log,(stage/'host-samples.jsonl').open('x') as samples:
        try:
            process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException as error:
            ledger_update(dict(status='closed_before_launch',reserved_chip_hours=0.,error=repr(error)))
            atomic_json(stage/'launch-failure.json',dict(error=repr(error)));raise
        owned=h.identity(process.pid);assert owned and owned['session']==owned['pgid']==process.pid
        atomic_json(stage/'launch.json',dict(utc=datetime.now(timezone.utc).isoformat(),identity=owned,
            command=command,protocol_sha256=job['protocol_sha256'],release_sha256=sha256_file(args.release),devices=devices))
        try:
            while process.poll() is None:
                members=h.members(owned['session'])
                sample=dict(utc=datetime.now(timezone.utc).isoformat(),MemAvailable_bytes=h.available(),
                    free_bytes={str(q):shutil.disk_usage(q).free for q in (root,cache,output.parent)},
                    processes=[dict(**v,memory=h.memory(v['pid'])) for v in members],
                    global_science_reserved_chip_hours=ledger_update())
                samples.write(json.dumps(sample)+'\n');samples.flush()
                if not observed:
                    actual=[v for row in h.topology() if row['neuron_device'] in devices for v in row['neuron_processes']]
                    if len(actual)==8:
                        assert {v['pid'] for v in actual}<={v['pid'] for v in members}
                        atomic_json(stage/'first-live.json',dict(actual8=actual,utc=sample['utc']));observed=True
                if args.phase=='stage64' and not first_checked and (output/'metrics.jsonl').exists():
                    lines=(output/'metrics.jsonl').read_text().splitlines()
                    if lines:
                        first=json.loads(lines[0]);ref=json.loads(Path(p['initial_loss_reference_file']).read_text())
                        assert sha256_file(p['initial_loss_reference_file'])==p['initial_loss_reference_sha256']
                        error={name:abs(first['loss_components'][name+'_mean_loss']-value)
                               for name,value in ref['global_means'].items()}
                        total=abs(first['loss']-ref['global_total_loss']);bounds=ref['comparisons_fixed_before_device_calls']
                        counts=all(first['loss_components'][name+'_labels']==value for name,value in ref['global_counts'].items())
                        passed=counts and max(error.values())<=bounds['max_component_mean_abs_error'] and total<=bounds['max_total_abs_error']
                        atomic_json(stage/'first-step-reference.json',dict(passed=passed,component_abs_error=error,total_abs_error=total,counts_exact=counts))
                        if not passed:raise RuntimeError('Preserve failed actual first-step reference; do not change tolerances')
                        first_checked=True
                if stop is None and (sample['MemAvailable_bytes']<512*2**30
                        or min(sample['free_bytes'].values())<35*2**30
                        or sample['global_science_reserved_chip_hours']>=release['science_reserved_chip_hours']-1
                        or time.monotonic()-started>=phase_seconds-180):
                    stop=time.monotonic();atomic_json(stage/'STOP',dict(reason='Declared resource boundary',sample=sample))
                if stop is not None and time.monotonic()-stop>180:h.kill_remaining(owned)
                time.sleep(10)
        except BaseException as error:
            failure=f'{type(error).__name__}: {error}'
            h.kill_remaining(owned)
    code=process.wait()
    for _ in range(30):
        if not h.members(owned['session']):break
        time.sleep(1)
    if h.members(owned['session']):h.kill_remaining(owned)
    absent=not h.members(owned['session'])
    idle=all(not row['neuron_processes'] for row in h.topology() if row['neuron_device'] in devices)
    hours=2*(time.monotonic()-started)/3600
    ledger_update(dict(status='closed',reserved_chip_hours=hours,exit_code=code,owned_session_absent=absent,devices_idle=idle))
    closure=dict(exit_code=code,actual_owned8_observed=observed,owned_session_absent=absent,
        assigned_devices_idle=idle,reserved_chip_hours=hours,external_stop=stop is not None,
        first_step_checked=first_checked if args.phase=='stage64' else 'original first-step preserved',failure=failure)
    atomic_json(stage/'closure.json',closure)
    assert code==0 and observed and absent and idle and stop is None and failure is None
    assert args.phase!='stage64' or first_checked
    from train.neuron_driver import verify_checkpoint
    terminal=64 if args.phase=='stage64' else 1280
    manifest=verify_checkpoint(output/f'step-{terminal:07d}',canonical_hash(p),8)
    metrics=[json.loads(line) for line in (output/'metrics.jsonl').open()]
    assert [v['step'] for v in metrics]==list(range(1,terminal+1))
    atomic_json(stage/'completed.json',dict(status='PHYSICALLY_CLOSED_ORDERED_STAGE_AND_FULL_SHA_READBACK',
        terminal=terminal,COMPLETE_sha256=sha256_file(output/f'step-{terminal:07d}'/'COMPLETE.json'),
        protocol_sha256=job['protocol_sha256'],spec_hash=manifest['spec_hash'],closure=closure))
    print(json.dumps(closure),flush=True)


if __name__=='__main__':main()
