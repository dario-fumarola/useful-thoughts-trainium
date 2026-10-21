"""One finite local qualification or released reader-head experiment; no retry.

Uses the existing qualified process-identity/cleanup helpers. Charges whole
physical-chip phase wall time, including startup, idle cores and closure.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from train.causal_reader_gate_driver import read,require_frozen,require_science_release,science_budget
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

R=Path('/home/ubuntu/dario')
W=Path(__file__).resolve().parents[1]


def worker_ports(cursor):
    """Two distinct loopback ports, probed before launch and never reused.

    Independent one-core runtimes still bootstrap a communicator. Letting
    simultaneous workers share implicit bootstrap defaults can cross-connect
    them. Keep allocations above this host's ephemeral range and record them.
    """
    held=[];ports=[]
    try:
        while len(ports)<2 and cursor<65535:
            candidate=cursor;cursor+=1
            probe=socket.socket()
            try:probe.bind(('127.0.0.1',candidate))
            except OSError:probe.close();continue
            held.append(probe);ports.append(candidate)
        if len(ports)!=2:raise RuntimeError('No unused fixed-range worker communication ports')
        return ports,cursor
    finally:
        for probe in held:probe.close()


class Runner:
    def __init__(self,a,p):
        helper=R/'local/diagnostics/real-source-balanced-v17-world8-target640-qualification-v2/qualify.py'
        if sha256_file(helper)!='57fe9fb1a760dd160574aa81f9210092249fa063e44c8a106cd65345b14b627f':
            raise ValueError('Qualified process helper changed')
        spec=importlib.util.spec_from_file_location('reader_process_helpers',helper)
        self.h=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.h)
        self.a=a;self.p=p;self.deadline=datetime.fromisoformat(a.deadline_utc)
        self.workers=[];self.reservations=[];self.live=[]
        self.initial_spent=p.get('prior_qualification_reserved_chip_hours',0.)
        self.ceiling=.5 if a.phase=='qualification' else science_budget(p)
        if a.phase=='science':
            release=require_science_release(p,a.protocol)
            self.initial_spent=read(release['profile_file'])['actual_qualification_reserved_chip_hours']
            if self.deadline>datetime.fromisoformat(release['deadline_utc']):
                raise ValueError('Worker deadline exceeds whole-question release')
        self.template=read(R/'local/science/control/W-seed20261018-development64-evaluation-v1/environment.template.json')
        self.namespace=hashlib.sha256(str(a.output.resolve()).encode()).hexdigest()[:10]
        self.port_cursor=61000+(int(self.namespace,16)%100)*32
        a.output.mkdir(parents=True,exist_ok=False)

    def spent(self):return self.initial_spent+sum(v['elapsed_wall_seconds']*v['reserved_physical_chips']/3600
                                                 for v in self.reservations)

    def check(self,current_seconds=0,devices=()):
        if (datetime.now(timezone.utc)>=self.deadline or (self.a.output/'STOP').exists()
                or self.h.available()<512*2**30 or shutil.disk_usage(R).free<35*2**30
                or self.spent()+(current_seconds+60)*len(devices)/3600>=self.ceiling):
            raise RuntimeError('Finite deadline/resource/reserved-chip-hour boundary reached; no retry')
        if self.a.phase=='science':
            used=sum(v.stat().st_size for v in self.a.output.rglob('*') if v.is_file())
            if used>6*2**30:raise RuntimeError('Fixed6GiB experiment-output ceiling reached')

    def stage(self,name,jobs,devices=()):
        self.check(devices=devices);top=self.h.topology()
        for device in devices:
            hardware=next(v for v in top if v['neuron_device']==device)
            if (hardware['neuron_processes'] or hardware['logical_neuroncore_config']!=2
                    or hardware['neuroncore_ids']!=list(range(device*4,device*4+4))):
                raise ValueError('Existing empty LNC2 core mapping required')
        directory=self.a.output/'control'/name;directory.mkdir(parents=True)
        phase_start=time.monotonic();launched=[];captured=not devices;last_topology=0.
        try:
            for number,(command,output,core) in enumerate(jobs):
                control=directory/str(number);control.mkdir()
                env=os.environ.copy()
                for key in self.template['unset']:env.pop(key,None)
                env.update(self.template['environment'])
                tmp=Path('/dev/shm')/f'crg-{self.namespace}-{name}-{number}';tmp.mkdir(mode=0o700)
                env.update(PYTHONPATH=str(W)+':'+str(R/'local/test-deps'),USE_TORCH_XLA='0',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(tmp),XLA_COMPILATION_CACHE_SIZE='256')
                if 'compiler_flags' in self.p:
                    if self.p['compiler_flags']!=['--model-type=transformer','--auto-cast=none']:
                        raise ValueError('Only the explicit frozen FP32 compiler settings are supported')
                    env['NEURON_CC_FLAGS']=' '.join(self.p['compiler_flags'])
                communication=None
                if core is not None:
                    if core//4 not in devices:raise ValueError('Worker outside fixed reserved physical chips')
                    env.update(NEURON_RT_VISIBLE_CORES=str(core),NEURONCORE_NUM_DEVICES='1')
                    ports,self.port_cursor=worker_ports(self.port_cursor)
                    env.update(NEURON_RT_ROOT_COMM_ID=f'127.0.0.1:{ports[0]}',
                               MASTER_ADDR='127.0.0.1',MASTER_PORT=str(ports[1]))
                    communication=dict(runtime_root=env['NEURON_RT_ROOT_COMM_ID'],
                                       torch_master=env['MASTER_ADDR']+':'+env['MASTER_PORT'])
                else:env.pop('NEURON_RT_VISIBLE_CORES',None)
                remaining=max(1,int((self.deadline-datetime.now(timezone.utc)).total_seconds()))
                if devices:
                    remaining=min(remaining,max(1,int((self.ceiling-self.spent())*3600/len(devices)
                                                     -(time.monotonic()-phase_start)-60)))
                argv=['timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s',sys.executable,'-u',*command]
                log=(control/'worker.log').open('xb');started=time.monotonic()
                proc=subprocess.Popen(argv,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                owned=self.h.identity(proc.pid)
                if not owned or owned['session']!=proc.pid or owned['pgid']!=proc.pid:
                    raise RuntimeError('Cannot identify owned worker session')
                record=dict(proc=proc,owned=owned,log=log,started=started,output=output,
                            control=control,core=core,stage=name,command=argv)
                launched.append(record);self.live.append(record)
                atomic_json(control/'launch.json',dict(command=argv,identity=owned,core=core,
                    protocol_sha256=sha256_file(self.a.protocol),supervisor_sha256=sha256_file(__file__),
                    source_directory=str(W),communication=communication,created_utc=utc_now()))
            with (directory/'host-samples.jsonl').open('x') as samples:
                while any(j['proc'].poll() is None for j in launched):
                    self.check(time.monotonic()-phase_start,devices)
                    if any(j['proc'].poll() not in (None,0) for j in launched):
                        raise RuntimeError('Worker failed; no peer continuation or automatic retry')
                    processes=[v for j in launched for v in self.h.members(j['owned']['session'])]
                    samples.write(json.dumps(dict(utc=utc_now(),available_host_bytes=self.h.available(),
                        filesystem_free_bytes=shutil.disk_usage(R).free,
                        owned_processes=[dict(**v,memory=self.h.memory(v['pid'])) for v in processes]))+'\n')
                    samples.flush()
                    if devices and time.monotonic()-last_topology>5:
                        actual=self.h.topology();last_topology=time.monotonic()
                        allocation=[v for d in actual if d['neuron_device'] in devices for v in d['neuron_processes']]
                        if allocation:
                            if not {v['pid'] for v in allocation}<={v['pid'] for v in processes}:
                                raise RuntimeError('Unexpected process on reserved physical chips')
                            if not captured:
                                atomic_json(directory/'first-live.json',dict(utc=utc_now(),topology=actual));captured=True
                    time.sleep(2)
            for j in launched:
                code=j['proc'].wait();j['log'].close()
                record=dict(execution_device='xla' if j['core'] is not None else 'cpu',
                    reserved_logical_cores=1 if j['core'] is not None else 0,
                    logical_cores_per_physical_chip=4,elapsed_wall_seconds=time.monotonic()-j['started'],
                    returncode=code,stage=name,core=j['core'])
                self.workers.append(record);atomic_json(j['control']/'exit.json',record)
                if code or not (j['output']/'complete.json').exists():
                    raise RuntimeError('Worker failed or lacks complete original artifacts')
            for _ in range(20):
                if not any(self.h.members(j['owned']['session']) for j in launched):break
                time.sleep(1)
            if (not captured or any(self.h.members(j['owned']['session']) for j in launched)
                    or any(d['neuron_processes'] for d in self.h.topology() if d['neuron_device'] in devices)):
                raise RuntimeError('Native/source processes did not physically close')
            elapsed=time.monotonic()-phase_start
            if devices:self.reservations.append(dict(stage=name,reserved_physical_chips=len(devices),
                physical_devices=list(devices),elapsed_wall_seconds=elapsed))
            atomic_json(directory/'closure.json',dict(status='ALL_WORKERS_PHYSICALLY_CLOSED',
                seconds=elapsed,reserved_physical_chips=len(devices),utc=utc_now()))
            self.live=[j for j in self.live if j not in launched]
            print('STAGE_CLOSED',name,'reserved_chip_hours',self.spent(),flush=True)
        except BaseException:
            for j in self.live:
                if self.h.same(j['owned']):os.killpg(j['owned']['pgid'],signal.SIGTERM)
            time.sleep(2)
            for j in self.live:self.h.kill_remaining(j['owned'])
            for j in self.live:j['proc'].wait();j['log'].close()
            elapsed=time.monotonic()-phase_start
            if devices:self.reservations.append(dict(stage=name,reserved_physical_chips=len(devices),
                physical_devices=list(devices),elapsed_wall_seconds=elapsed))
            atomic_json(directory/'failure.json',dict(status='INCOMPLETE_NO_RETRY',utc=utc_now(),
                native_reservations=self.reservations,reserved_chip_hours_including_prior_qualification=self.spent(),
                worker_exit_codes=[j['proc'].returncode for j in launched]))
            raise

    def workers_for(self,command,cohort,role,count,notes=None,device='xla'):
        jobs=[]
        category='notes' if command=='generate' else 'reference' if command=='reference' else 'causality' if command=='causality' else 'interventions' if command=='interventions' else 'features'
        base=self.a.output/category
        if category=='notes':base/=cohort
        elif category=='features':base=base/cohort/role
        elif category in ('reference','causality'):base/=role
        for part in range(count):
            output=base/f'part-{part}' if category in ('notes','features','interventions') else base
            output.parent.mkdir(parents=True,exist_ok=True)
            args=['-m','train.causal_reader_gate_driver',command,'--protocol',str(self.a.protocol),
                '--phase',self.a.phase,'--cohort',cohort,'--role',role,'--device',device,
                '--output',str(output),'--shard-index',str(part),'--num-shards',str(count),
                '--deadline-utc',self.a.deadline_utc]
            if notes is not None:args+=['--notes',str(notes)]
            jobs.append((args,output,32+part if device=='xla' else None))
        return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--phase',choices=('qualification','science'),required=True)
    parser.add_argument('--deadline-utc',required=True)
    a=parser.parse_args();p=read(a.protocol);require_frozen(p)
    def interrupted(signum,frame):raise InterruptedError('Explicit supervisor termination')
    signal.signal(signal.SIGTERM,interrupted)
    if a.phase=='qualification' and p['qualification_release']['status']!='RELEASED_BOUNDED_ACTUAL_MODEL_QUALIFICATION_ONLY':
        raise ValueError('Actual-model qualification held')
    runner=Runner(a,p)
    if a.phase=='qualification':
        runner.stage('generate',runner.workers_for('generate','qualification','T_note',4),[8])
        notes=a.output/'notes'/'qualification'
        for role in ('B0_raw','T_raw','T_note'):
            runner.stage('reference-'+role,runner.workers_for('reference','qualification',role,1,notes,'cpu'))
        for role in ('B0_raw','T_raw','T_note'):
            runner.stage('features-'+role,runner.workers_for('features','qualification',role,4,notes),[8])
        jobs=[]
        for index,role in enumerate(('B0_raw','T_raw','T_note')):
            args,output,_=runner.workers_for('causality','qualification',role,1,notes)[0]
            jobs.append((args,output,32+index))
        runner.stage('causality',jobs,[8])
        status='ALL_QUALIFICATION_PROCESSES_CLOSED'
    else:
        require_science_release(p,a.protocol)
        for cohort in p.get('science_cohorts',('TRAIN','DEV_calculation','DEV_ordinary')):
            runner.stage(cohort+'-generate',runner.workers_for('generate',cohort,'T_note',8),[8,9])
            notes=a.output/'notes'/cohort
            for role in ('B0_raw','T_raw','T_note'):
                runner.stage(cohort+'-'+role,runner.workers_for('features',cohort,role,8,notes),[8,9])
            if cohort=='TRAIN':
                output=a.output/'fitted'
                runner.stage('CPU-fit', [(['-m','train.causal_reader_head_driver','fit',
                    '--protocol',str(a.protocol),'--features',str(a.output/'features'),
                    '--output',str(output),'--deadline-utc',a.deadline_utc],output,None)])
        if p.get('secondary_interventions',{}).get('item_ids'):
            runner.stage('interventions',runner.workers_for('interventions','DEV_calculation','T_note',4,
                                                           a.output/'notes'/'DEV_calculation'),[8])
        status=('ALL_FROZEN_HEAD_CONFIRMATION_FEATURE_PROCESSES_CLOSED'
                if p.get('study_schema')=='frozen-reader-confirmation-v1'
                else 'ALL_SCIENCE_FEATURE_AND_TRAIN_HEAD_PROCESSES_CLOSED')
    atomic_json(a.output/'runtime.json',dict(status=status,protocol_sha256=sha256_file(a.protocol),
        workers=runner.workers,native_reservations=runner.reservations,
        prior_reserved_chip_hours=runner.initial_spent,
        total_reserved_chip_hours_including_prior_qualification=runner.spent(),
        supervisor_sha256=sha256_file(__file__),completed_utc=utc_now()))


if __name__=='__main__':main()
