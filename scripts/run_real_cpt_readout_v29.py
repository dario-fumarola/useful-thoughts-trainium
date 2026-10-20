"""One finite V29 readout phase on an exclusively reserved physical chip.

This schedules the qualified workers; it contains no model math or parsing.
Four scalar shards share one physical-chip reservation, not four reservations.
No automatic retry or partial-panel result selection is implemented.
"""
import argparse
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

from train.real_cpt_readout_driver import collect,load_reference,sources
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now


def read(path):return json.loads(Path(path).read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--panel',choices=('writer','math','ordinary','scalar'),required=True)
    args=parser.parse_args();C=args.control;release=read(args.release)
    assert release['status']=='RELEASED_ALL_SEVEN_V29_FIXED_TERMINAL_READOUTS'
    assert release['source_files']==sources() and release['supervisor_sha256']==sha256_file(__file__)
    reference=Path(release['references'][args.model]);p,_=load_reference(reference)
    assert p['purpose']=='science' and p['model']==release['models'][args.model]
    for name,digest in release['science_closures'].items():
        assert sha256_file(name)==digest
        closure=read(name)
        assert closure['exit_code']==0 and closure['owned_session_absent'] and closure['assigned_devices_idle']
    helper=Path(release['process_helper']['path'])
    assert sha256_file(helper)==release['process_helper']['sha256']
    spec=importlib.util.spec_from_file_location('owned_process_utilities_only',helper)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    root=Path(release['repository_root']);source=Path(release['source_root'])
    device=release['physical_slots'][args.model];base=4*device
    count=4 if args.panel=='scalar' else 1
    core_ids=list(range(base,base+count))
    assert all(not row['neuron_processes'] and row['logical_neuroncore_config']==2
               for row in h.topology() if row['neuron_device']==device)
    assert h.available()>512*2**30
    assert min(shutil.disk_usage(q).free for q in (root,Path(release['cache_filesystem'])))>35*2**30
    stage=C/'supervision'/args.model/args.panel;stage.mkdir(parents=True,exist_ok=False)
    output=C/'jobs'/args.model/args.panel
    output.parent.mkdir(parents=True,exist_ok=True)
    if args.panel=='scalar':output.mkdir(exist_ok=False)
    notes=C/'jobs'/args.model/'writer/notes-bundle.json'
    if args.panel in ('math','scalar'):
        bundle=read(notes);assert bundle['model']==p['model'] and bundle['status']=='COMPLETE_V29_INTACT_ORIGINAL_NOTES'
    seconds=release['phase_wall_seconds'][args.panel]
    deadline=datetime.fromtimestamp(time.time()+seconds,tz=timezone.utc).isoformat()
    temp=Path('/dev/shm')/f'cpt29ev-{device}-{args.panel}';temp.mkdir(mode=0o700,exist_ok=False)
    template=read(release['environment_template']);environment=os.environ.copy()
    for key in template['unset']:environment.pop(key,None)
    environment.update(template['environment'])
    for key,value in p['runtime_policy']['environment'].items():
        if value is None:environment.pop(key,None)
        else:environment[key]=value
    environment.update(PYTHONPATH=str(source)+':'+str(root/'local/test-deps'),
        NEURONCORE_NUM_DEVICES='1',TMPDIR=str(temp))
    key=args.model+'/'+args.panel;ledger_path=C/'reservations.json'
    def ledger(change=None):
        with (C/'reservations.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=read(ledger_path) if ledger_path.exists() else {}
            if change=='start':
                assert key not in value
                value[key]=dict(status='active',started_monotonic=time.monotonic(),physical_chips=1)
            elif isinstance(change,dict):value[key].update(change)
            if change is not None:atomic_json(ledger_path,value,overwrite=ledger_path.exists())
            return sum(row.get('reserved_chip_hours',(time.monotonic()-row['started_monotonic'])/3600)
                       for row in value.values())
    assert ledger()<release['reserved_chip_hours']-1
    started=time.monotonic();ledger('start');children=[];failure=None;observed=False
    try:
        for shard,core in enumerate(core_ids):
            directory=output/f'shard-{shard}' if args.panel=='scalar' else output
            current=environment.copy();current['NEURON_RT_VISIBLE_CORES']=str(core)
            command=['timeout','--signal=TERM','--kill-after=60s',str(seconds)+'s',sys.executable,'-u',
                '-m','train.real_cpt_readout_driver','run','--reference',str(reference),
                '--output',str(directory),'--panel',args.panel,'--release',str(args.release),'--deadline-utc',deadline]
            if args.panel in ('math','scalar'):command+=['--notes',str(notes)]
            if args.panel=='scalar':command+=['--shard',str(shard)]
            log=(stage/f'shard-{shard}.log').open('xb')
            process=subprocess.Popen(command,cwd=source,env=current,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            owned=h.identity(process.pid);assert owned and owned['session']==owned['pgid']==process.pid
            children.append((process,owned,log,directory))
            atomic_json(stage/f'shard-{shard}-launch.json',dict(command=command,owned=owned,
                physical_chip=device,logical_core=core,utc=utc_now(),release_sha256=sha256_file(args.release)))
        with (stage/'host-samples.jsonl').open('x') as samples:
            while any(child.poll() is None for child,_,_,_ in children):
                members=[v for _,owned,_,_ in children for v in h.members(owned['session'])]
                actual=[v for row in h.topology() if row['neuron_device']==device for v in row['neuron_processes']]
                assert {v['pid'] for v in actual}<={v['pid'] for v in members}
                if not observed and len(actual)==count:
                    atomic_json(stage/'first-live.json',dict(actual=actual,utc=utc_now()));observed=True
                free={str(q):shutil.disk_usage(q).free for q in (root,Path(release['cache_filesystem']))}
                sample=dict(utc=utc_now(),MemAvailable_bytes=h.available(),free_bytes=free,
                    processes=[dict(**v,memory=h.memory(v['pid'])) for v in members],global_reserved_chip_hours=ledger())
                samples.write(json.dumps(sample)+'\n');samples.flush()
                if (sample['MemAvailable_bytes']<512*2**30 or min(free.values())<35*2**30
                        or sample['global_reserved_chip_hours']>=release['reserved_chip_hours']-1
                        or time.monotonic()-started>=seconds-60):
                    raise RuntimeError('Declared resource/deadline boundary; preserve original partial output')
                if any(child.poll() not in (None,0) for child,_,_,_ in children):
                    raise RuntimeError('One required shard failed; no automatic relaunch')
                time.sleep(10)
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}'
        for child,owned,_,_ in children:h.kill_remaining(owned)
    codes=[]
    for child,owned,log,_ in children:
        codes.append(child.wait());log.close()
    for _ in range(30):
        if not any(h.members(owned['session']) for _,owned,_,_ in children):break
        time.sleep(1)
    for _,owned,_,_ in children:
        if h.members(owned['session']):h.kill_remaining(owned)
    absent=not any(h.members(owned['session']) for _,owned,_,_ in children)
    idle=all(not row['neuron_processes'] for row in h.topology() if row['neuron_device']==device)
    hours=(time.monotonic()-started)/3600
    closure=dict(status='SESSION_AND_PHYSICAL_DEVICE_CLOSED' if absent and idle else 'CLOSURE_FAILED',
        exit_code=0 if len(codes)==count and all(code==0 for code in codes) and failure is None else 1,
        child_exit_codes=codes,owned_session_absent=absent,physical_device_idle=idle,
        observed_expected_live_owners=observed,reserved_chip_hours=hours,
        physical_chips=1,logical_cores=core_ids,failure=failure)
    atomic_json(stage/'closure.json',closure)
    ledger(dict(status='closed',reserved_chip_hours=hours,closure=closure))
    for _,_,_,directory in children:atomic_json(directory/'closure.json',closure)
    assert closure['exit_code']==0 and observed and absent and idle
    for _,_,_,directory in children:
        assert read(directory/'run.json')['status']=='COMPLETE_ALL_DECLARED_ATTEMPTS'
    if args.panel=='writer':collect(argparse.Namespace(reference=reference,output=output))
    atomic_json(stage/'complete.json',dict(status='ALL_REQUIRED_PHASE_OUTPUTS_PHYSICALLY_CLOSED',
        closure=closure,release_sha256=sha256_file(args.release)))
    print(json.dumps(closure),flush=True)


if __name__=='__main__':main()
