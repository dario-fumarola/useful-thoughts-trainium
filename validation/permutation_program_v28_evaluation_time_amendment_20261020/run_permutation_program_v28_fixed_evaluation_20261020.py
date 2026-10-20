"""One finite fixed23-job evaluation batch, at most8 chips/30 reserved chip-hours.

All six terminal audits and CPU preparation must exist. Independent per-chip
workers use the unchanged frozen decoder. No scores are opened by this queue.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import importlib.util,json,os,signal,subprocess,sys,time,shutil
from permutation_program_v28_eval_budget_20261020 import MAX_CHIPS,MAX_CHIP_HOURS,STOP_CHIP_HOURS,reserved_hours
from train.permutation_program_science_evaluate import verify_release,sources
from weekend_thinking.artifacts import atomic_json,sha256_file,utc_now

R=Path('/home/ubuntu/dario');W=R/'local/worktrees/controlled-computation-v28-eval'
C=R/'local/diagnostics/permutation-program-v28-fixed-evaluation-v1'
CONTROLLER=R/'local/analysis/supervise_permutation_program_v28_generation_group_20261020.py'
HELPER=R/'local/diagnostics/permutation-program-v28-world4-qualification-v1/qualify.py'


def main():
    spec=importlib.util.spec_from_file_location('fixed_eval_owned_helpers',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    registry=h.read(C/'registry.json');release=verify_release(registry['release_file'])
    assert release['maximum_evaluation_reserved_chip_hours']==MAX_CHIP_HOURS
    amendment=release['operational_budget_amendment']
    assert h.sha(amendment['path'])==amendment['sha256']
    assert h.read(amendment['path'])['status']=='PROSPECTIVE_V28_EVALUATION_TIME_ONLY_AMENDMENT'
    assert registry['status']=='PREPARED_FIXED23_INPUTS_NO_ACCELERATOR_EXECUTION'
    assert registry['source_files']==sources() and registry['release_sha256']==h.sha(registry['release_file'])
    assert registry['controller_source_sha256']==h.sha(CONTROLLER)
    assert len(registry['jobs'])==23 and registry['index_sha256']==h.sha(C/'index.json')
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=W,text=True).strip()
    jobs=registry['jobs'];assert len({v['key'] for v in jobs})==23 and sum(v['attempts'] for v in jobs)==378880
    for job in jobs:
        assert h.sha(Path(job['output'])/'protocol.json')==job['protocol_sha256']
        assert not (Path(job['output'])/'answers').exists()
    # Long primary trace jobs first; then the fixed base references. Ordering
    # uses identities/counts only, never generated tokens, likelihood or scores.
    def priority(job):
        first=0 if '-trace-' in job['model'] and job['panel']=='DEV_composition' and not job['diagnostic'] else 1 if job['model']=='B0' else 2
        return first,-job['attempts'],job['key']
    pending=sorted(jobs,key=priority);active={};records=[];started=time.monotonic()
    work=C/'EXECUTION';heartbeat=work/'budget-heartbeat.json'
    owner=h.identity(os.getpid());assert owner
    now=datetime.now(timezone.utc);deadline=now+timedelta(hours=8)-timedelta(seconds=120)
    top=h.topology();assert all(not v['neuron_processes'] and v['logical_neuroncore_config']==2 for v in top if v['neuron_device'] in range(MAX_CHIPS))
    assert h.available()>768*2**30
    work.mkdir(exist_ok=False)
    budget_source=Path(__file__).with_name('permutation_program_v28_eval_budget_20261020.py')
    atomic_json(work/'plan.json',dict(status='RELEASED_FIXED_V28_BATCH_EXECUTION',utc=utc_now(),owner=owner,
        maximum_chips=MAX_CHIPS,maximum_reserved_chip_hours=MAX_CHIP_HOURS,stop_at_chip_hours=STOP_CHIP_HOURS,
        worker_deadline_utc=deadline.isoformat(),index_sha256=h.sha(C/'index.json'),registry_sha256=h.sha(C/'registry.json'),
        controller_sha256=h.sha(CONTROLLER),budget_source_sha256=h.sha(budget_source),queue_source_sha256=h.sha(__file__),
        operational_budget_amendment=amendment,
        order=[v['key'] for v in pending],qualification='Existing four independent decoder workers per chip,unchanged numerical path; each worker checks actual own-model/native/cache distributions.',
        limits='Eight is a maximum slot count,not eight reserved chips for the whole eight-hour wall bound. Charge each launched group until its controller and original device processes close. The prospective time-only amendment preserves all science/numerical inputs. No automatic retry or outcome-based scheduling.'))
    requested=[False];signal.signal(signal.SIGTERM,lambda *_:requested.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:requested.__setitem__(0,True))
    reason=None;logs=[]
    def cleanup(control):
        """Only this group's captured original process identities may be killed."""
        owned=[];captured=[]
        for path in control.glob('shard-*/launch.json'):
            owned.append(h.read(path)['identity'])
        sample=control/'host-samples.jsonl'
        if sample.exists() and sample.stat().st_size:
            with sample.open('rb') as stream:
                stream.seek(max(0,sample.stat().st_size-262144));last=json.loads(stream.read().splitlines()[-1])
            captured=last['processes']
        for value in owned:
            if h.same(value):
                try:os.kill(value['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        until=time.monotonic()+35
        while time.monotonic()<until and any(h.members(v['session']) for v in owned):time.sleep(1)
        for value in owned:
            original_live=h.same(value) or any(v['session']==value['session'] and h.same(v) for v in captured)
            if original_live:h.kill_remaining(value)
        for _ in range(10):
            if not any(h.members(v['session']) for v in owned):return
            time.sleep(1)
        assert not any(h.members(v['session']) for v in owned),'Original session closure is required'
    def beat():
        status='STOP_FIXED_V28_EVALUATION' if reason else 'COMPLETE_FIXED_V28_EVALUATION' if not pending and not active and len(records)==23 else 'ACTIVE_FIXED_V28_EVALUATION'
        atomic_json(heartbeat,dict(status=status,
            utc=utc_now(),monotonic=time.monotonic(),owner=owner,maximum_chips=MAX_CHIPS,
            maximum_reserved_chip_hours=MAX_CHIP_HOURS,reserved_chip_hours=reserved_hours(records,time.monotonic()),
            worker_deadline_utc=deadline.isoformat(),reason=reason),overwrite=True)
    try:
        while pending or active:
            for chip,child in list(active.items()):
                code=child['process'].poll()
                if code is None:continue
                if code or not (child['control']/'closure.json').exists():
                    reason=reason or 'A fixed original group failed;no retry'
                    beat();cleanup(child['control'])
                else:
                    closure=h.read(child['control']/'closure.json')
                    assert closure['assigned_physical_device_empty'] and closure['all_original_session_processes_absent']
                    assert closure['actual_attempts']==child['record']['attempts']
                child['record'].update(closed_monotonic=time.monotonic(),closed_utc=utc_now(),exit_code=code)
                atomic_json(work/(child['record']['key']+'-closed.json'),child['record'])
                del active[chip]
            spent=reserved_hours(records,time.monotonic())
            if requested[0] or (work/'STOP').exists():reason='Operator stop'
            if spent>=STOP_CHIP_HOURS:reason=reason or 'Reserved chip-hour bound'
            if datetime.now(timezone.utc)>=deadline:reason=reason or 'Finite batch wall bound'
            if h.available()<512*2**30 or min(shutil.disk_usage(v).free for v in (R,R/'local/nvme-cpt',R/'local/nvme-study'))<35*2**30:reason=reason or 'Resource floor'
            beat()
            if reason:
                if not active:break
                # The same heartbeat stop is checked by all original group
                # controllers. A stale/missing/dead-parent heartbeat also stops
                # them. Their owned TERM/KILL cleanup completes within120s.
                time.sleep(2);continue
            for chip in range(MAX_CHIPS):
                if not pending:break
                if chip in active or h.available()<768*2**30:continue
                job=pending.pop(0);control=work/job['key'];log=(work/(job['key']+'.log')).open('xb');logs.append(log)
                record=dict(key=job['key'],physical_chip=chip,attempts=job['attempts'],started_monotonic=time.monotonic(),started_utc=utc_now(),protocol_sha256=job['protocol_sha256'])
                records.append(record);beat()
                command=[sys.executable,'-u',str(CONTROLLER),job['output'],str(chip),str(control),deadline.isoformat(),str(heartbeat)]
                process=subprocess.Popen(command,cwd=W,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['controller_identity']=h.identity(process.pid);record['command']=command
                active[chip]=dict(process=process,record=record,control=control)
                atomic_json(work/(job['key']+'-launch.json'),record)
            time.sleep(2)
    finally:
        reason=reason or ('Interrupted queue' if pending or active else None);beat()
        if active:
            for child in active.values():
                cleanup(child['control'])
                if child['process'].poll() is None:
                    child['process'].terminate()
                    try:child['process'].wait(timeout=5)
                    except subprocess.TimeoutExpired:child['process'].kill();child['process'].wait()
                child['record'].update(closed_monotonic=time.monotonic(),closed_utc=utc_now(),exit_code=child['process'].returncode)
            active.clear()
        for log in logs:log.close()
    spent=reserved_hours(records,time.monotonic())
    assert spent<MAX_CHIP_HOURS
    status='COMPLETE_ALL_FIXED23_GROUPS_PHYSICALLY_CLOSED_NO_SCORES_REVIEWED' if not reason and not pending and not active else 'INCOMPLETE_FIXED_BATCH_NO_RETRY'
    atomic_json(work/'completion.json',dict(status=status,utc=utc_now(),reason=reason,groups_launched=len(records),pending=[v['key'] for v in pending],
        reserved_chip_hours=spent,wall_seconds=time.monotonic()-started,records=records,source_sha256=h.sha(__file__)))
    print(json.dumps(dict(status=status,groups=len(records),reserved_chip_hours=spent)),flush=True)
    if reason:raise SystemExit(2)


if __name__=='__main__':main()
