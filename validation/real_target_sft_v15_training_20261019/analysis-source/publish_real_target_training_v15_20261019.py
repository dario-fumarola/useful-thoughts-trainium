"""Publish completed training traces and state receipts without capability results."""
from pathlib import Path
from datetime import datetime,timezone
import gzip
import hashlib
import json
import math
import os
import shutil

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-target-sft-v15-study-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read(C/'terminal-pipeline-completion.json')['status']=='ALL_FOUR_MODELS_AUDITED_PROTECTED_AND_FROZEN'
audit=read(C/'terminal-independent-audit.json');plan=read(C/'plan.json')
assert audit['status']=='PASS_ALL_FOUR_TERMINAL320_TRAINING_STATES_NATIVE32_ORDER_AND_COUNTERS'
out=R/'validation/real_target_sft_v15_training_20261019';out.mkdir(exist_ok=False)
files={}
def copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    assert sha(source)==sha(dest)
    files[str(dest.relative_to(R))]=dict(bytes=dest.stat().st_size,sha256=sha(dest))
for name in ['plan.json','release.json','terminal-independent-audit.json','terminal-protection.json','terminal-pipeline-completion.json']:
    p=C/name
    if p.exists():copy(p,out/'evidence'/name)
models={}
for job,a in zip(plan['jobs'],audit['models']):
    assert (job['arm'],job['seed'])==(a['arm'],a['seed'])
    key=job['key'];output=Path(job['output']);metrics=output/'metrics.jsonl'
    assert sha(metrics)==a['metrics_sha256']
    values=[json.loads(v) for v in metrics.read_text().splitlines()]
    assert [v['step'] for v in values]==list(range(1,321))
    assert all(math.isfinite(v['loss']) and math.isfinite(v['gradient_norm']) for v in values)
    p=out/f'{key}-all320-training-metrics.jsonl.gz'
    p.write_bytes(gzip.compress(metrics.read_bytes(),mtime=0));files[str(p.relative_to(R))]=dict(rows=320,bytes=p.stat().st_size,sha256=sha(p))
    copy(Path(job['protocol']),out/'evidence'/key/'protocol.json')
    copy(C/'frozen-models'/(key+'.json'),out/'evidence'/key/'frozen-terminal-binding.json')
    copy(C/key/'closure.json',out/'evidence'/key/'physical-closure.json')
    copy(output/'completion.json',out/'evidence'/key/'completion.json')
    for step in [160,320]:copy(output/f'step-{step:07d}'/'COMPLETE.json',out/'evidence'/key/f'full{step}-COMPLETE.json')
    completion=read(output/'completion.json')
    models[key]=dict(arm=job['arm'],seed=job['seed'],updates=320,unique_documents=5120,document_occurrences=10240,
        supervised_target_tokens=values[-1]['counters']['supervised_tokens'],
        warm_last60_median_update_seconds=a['warm_last60_median_update_seconds'],
        summed_update_wall_seconds=math.fsum(v['elapsed_seconds'] for v in values),
        loop_including_saves_seconds=completion['loop_including_saves_seconds'],
        assigned_logical_workers=8,assigned_physical_devices=2,
        logical_worker_hours_during_loop=8*completion['loop_including_saves_seconds']/3600,
        physical_device_hours_during_loop=2*completion['loop_including_saves_seconds']/3600,
        checkpoint_walls=completion['checkpoint_walls'],
        terminal_model_sha256=a['checkpoints']['320']['model_sha256'])
for name in ['publish_real_target_training_v15_20261019.py','real_target_state_audit_v15_20261019.py',
             'audit_real_target_four_terminals_v15_20261019.py','protect_real_target_four_terminals_v15_20261019.py',
             'bind_real_target_four_terminals_v15_20261019.py','analyze_real_target_v15_20261019.py',
             'check_real_target_readout_v15_20261019.py','render_real_target_v15_20261019.py',
             'save_real_target_examples_v15_20261019.py','watch_real_target_final_readout_v15_20261019.py']:
    copy(R/'local/analysis'/name,out/'analysis-source'/name)
report=dict(status='ALL_FOUR_TERMINAL320_TRAINING_STATES_AUDITED_PROTECTED_AND_FROZEN',
    utc=datetime.now(timezone.utc).isoformat(),training_commit='829d9a2',evaluation_commit='41a5bfd',
    models=models,updates=1280,full_scientific_checkpoints=8,native_forward_checks=32,
    source_audit_sha256=sha(C/'terminal-independent-audit.json'),files=files,
    limits=['No capability outcomes inspected for this training completion report.',
            'Eight logical-worker checks per model repeat four distinct canonical contexts;not 32 independent scientific documents.',
            'All original full160/320 states retained onNVMe. EBS protects allfour terminalmodels and latestfull320;notoff-instancebackup.',
            'Different active-target counts are explicit;matched documents,updates andpackedwidth are not equal FLOPs orsupervision.',
            'Measured concurrent loop cost includes saves but excludes preparation,initialization/nativechecks andsubsequent evaluations.'])
path=out.with_suffix('.json');path.write_text(json.dumps(report,indent=2))
A=R/'local/backups/engineering/real-target-sft-v15-completed-training-evidence';A.mkdir(exist_ok=False)
receipts={}
for source in [path,*[p for p in out.rglob('*') if p.is_file()]]:
    assert shutil.disk_usage(R).free>35*2**30+source.stat().st_size
    target=A/source.relative_to(R/'validation');target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    with target.open('rb') as stream:os.fsync(stream.fileno())
    assert sha(source)==sha(target);receipts[str(target.relative_to(A))]=sha(target)
(A/'READBACK.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps(dict(status=report['status'],models=models,public_files=len(files)),indent=2),flush=True)
