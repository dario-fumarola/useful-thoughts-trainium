"""Independent ordered-source/component/native/full-state audit of one stage64."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import random
import statistics

import torch

from train.useful_note_precision import compare_distributions
from weekend_thinking.artifacts import atomic_json,sha256_file
from weekend_thinking.pretrained import load_base

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',required=True);args=parser.parse_args()
R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-cpt-scaffold-v29-science-v1'
registry=json.loads((C/'protocols-prepared.json').read_text());job=registry['jobs'][args.job]
p=json.loads(Path(job['protocol']).read_text());O=Path(job['output']);control=C/'jobs'/args.job/'stage64'
assert json.loads((control/'completed.json').read_text())['terminal']==64
assert json.loads((control/'first-step-reference.json').read_text())['passed']
torch.set_num_threads(8)
data=[json.loads(line) for line in Path(p['data_file']).open()]
assert len(data)==20480 and sha256_file(p['data_file'])==p['data_sha256']
groups={k:[i for i,row in enumerate(data) if row['kind']==k] for k in ('calculation','ordinary')}
order=[]
for epoch in range(2):
    rng=random.Random((p['seed']<<32)+epoch)
    calc,ordinary=list(groups['calculation']),list(groups['ordinary']);rng.shuffle(calc);rng.shuffle(ordinary)
    for i,index in enumerate(calc):
        group=[index]+ordinary[i*3:i*3+3];rng.shuffle(group);order.extend(group)
metrics=[json.loads(line) for line in (O/'metrics.jsonl').open()]
assert [m['step'] for m in metrics]==list(range(1,65));total=0
for step,m in enumerate(metrics):
    rows=[data[i] for i in order[step*32:step*32+32]]
    assert m['ordered_item_ids']==[r['item_id'] for r in rows]
    assert sum(row['kind']=='calculation' for row in rows)==8
    work=0 if job['arm']=='raw' else 16 if job['arm']=='format' else sum(len(row['full_calculation_ids'])+2 for row in rows if row['kind']=='calculation')
    counts=dict(source=sum(len(row['prefix_ids'])+15 for row in rows),work=work,continuation=128)
    comp=m['loss_components'];assert all(comp[k+'_labels']==v for k,v in counts.items())
    assert comp['source_weight']==comp['work_weight']==1. and comp['continuation_weight']==.5
    assert comp['source_batch_fraction']==1. and comp['work_batch_fraction']==comp['continuation_batch_fraction']==.25
    expected=comp['source_mean_loss']+.25*comp['work_mean_loss']+.125*comp['continuation_mean_loss']
    assert abs(expected-m['loss'])<1e-5
    assert math.isfinite(m['loss']) and math.isfinite(m['gradient_norm']) and m['gradient_norm']>0
    assert m['actual_master_and_replica_audit']==(step<3 or step==63)
    total+=sum(counts.values());assert m['counters']==dict(expressions=(step+1)*32,supervised_tokens=total)
base=json.loads(Path(p['base_protocol']).read_text())
ref=torch.load(Path(base['native_reference_output'])/'next-logps.pt',map_location='cpu',weights_only=True)
native=[]
for rank in range(8):
    row=json.loads((O/f'qualification-rank-{rank:05d}.json').read_text())
    actual=torch.load(O/f'forward-rank-{rank:05d}.pt',map_location='cpu',weights_only=True)
    check=compare_distributions(ref[row['reference_index']:row['reference_index']+1],actual)[0]
    assert check['passed'] and row['status']=='PASS'
    sample=torch.load(O/f'initial-master-rank-{rank:05d}.pt',map_location='cpu',weights_only=True)
    assert torch.equal(sample['expected_cpu'].view(torch.uint8),sample['actual_fp32_masters'].view(torch.uint8))
    native.append(check)
helper=R/'local/analysis/real_source_state_audit_v17_20261019.py'
spec=importlib.util.spec_from_file_location('generic_world8_full_state_audit',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
model,_=load_base(p['model_dir'])
with torch.no_grad():
    state=h.audit_checkpoint(O/'step-0000064',p,job['arm'],O.name,order,job['protocol_sha256'],metrics[-1]['counters'],model)
atomic_json(control/'independent-audit.json',dict(status='PASS_ACTUAL64_ORDER_COMPONENTS_NATIVE8_FULL_MASTER_MOMENT_AND_CAST_AUDIT',
    source_sha256=sha256_file(__file__),generic_state_audit_sha256=sha256_file(helper),job=args.job,
    protocol_sha256=job['protocol_sha256'],actual_updates=64,source_exposures=2048,
    supervised_positions=total,native=native,full_checkpoint=state,
    warm_last32_median_seconds=statistics.median(m['elapsed_seconds'] for m in metrics[-32:]),
    limits='TRAIN execution and full-state integrity,not held-out learning or indefinite memory stability. Original64 full state is in RAM; EBS protection is separate.'))
print(args.job,'PASS_FULL64',flush=True)
