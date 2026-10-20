"""Independent closed-run order/component/full-state readback; no model forward."""
from pathlib import Path
import importlib.util
import json
import math
import random

import torch
from transformers import AutoTokenizer

from train.useful_note_precision import compare_distributions
from train.useful_note_recovery import optimizer_digest
from weekend_thinking.artifacts import atomic_json, canonical_hash, sha256_file, utc_now
from weekend_thinking.pretrained import load_base

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-cpt-scaffold-v29-qualification-v2'
O=R/'local/qualification/ram-real-cpt-scaffold-v29-v2/run-v1'
W=R/'local/worktrees/real-cpt-scaffold-v29'
read=lambda p:json.loads(Path(p).read_text())
helper=R/'local/analysis/real_source_state_audit_v17_20261019.py'
spec=importlib.util.spec_from_file_location('generic_full_world8_state_audit_only',helper)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
p=read(C/'protocol.json')
assert all(sha256_file(W/k)==v for k,v in p['source_files'].items())
assert read(C/'hardware-complete.json')['total_reserved_chip_hours']<=1.
assert read(C/'fresh4/exit.json')['exit_code']==read(C/'resume8/exit.json')['exit_code']==0
assert read(C/'fresh4/initial-loss-check.json')['passed']
data=[json.loads(line) for line in Path(p['data_file']).open()]
calculation=[i for i,r in enumerate(data) if r['kind']=='calculation']
ordinary=[i for i,r in enumerate(data) if r['kind']=='ordinary']
assert len(calculation)==32 and len(ordinary)==96
order=[]
for epoch in range(2):
    rng=random.Random((p['seed']<<32)+epoch)
    a,b=list(calculation),list(ordinary);rng.shuffle(a);rng.shuffle(b)
    for n,index in enumerate(a):
        group=[index]+b[3*n:3*n+3];rng.shuffle(group);order+=group
metrics=[json.loads(line) for line in (O/'metrics.jsonl').open()]
assert [m['step'] for m in metrics]==list(range(1,9))
total=0
for step,m in enumerate(metrics):
    rows=[data[i] for i in order[32*step:32*step+32]]
    assert m['ordered_item_ids']==[r['item_id'] for r in rows]
    sources=sum(len(r['prefix_ids'])+15 for r in rows)
    work=(0 if step%3==0 else 16 if step%3==1 else
          sum(len(r['full_calculation_ids'])+2 for r in rows if r['kind']=='calculation'))
    labels=dict(source=sources,continuation=128,work=work)
    comp=m['loss_components']
    assert all(comp[name+'_labels']==count for name,count in labels.items())
    assert comp['source_weight']==comp['work_weight']==1. and comp['continuation_weight']==.5
    assert comp['source_batch_fraction']==1. and comp['work_batch_fraction']==comp['continuation_batch_fraction']==.25
    expected=comp['source_mean_loss']+.25*(comp['work_mean_loss']+.5*comp['continuation_mean_loss'])
    assert abs(expected-m['loss'])<1e-5
    assert math.isfinite(m['loss']) and math.isfinite(m['gradient_norm']) and m['gradient_norm']>0
    assert m['actual_master_and_replica_audit']==(step<3 or step+1 in (4,8))
    total+=sum(labels.values())
    assert m['counters']==dict(expressions=(step+1)*32,supervised_tokens=total)
base=read(p['base_protocol'])
reference=torch.load(Path(base['native_reference_output'])/'next-logps.pt',weights_only=True,map_location='cpu')
native=[];initial=[]
for directory in (O,O/'recoveries/resume8/initialization'):
    for rank in range(8):
        n=f'{rank:05d}';q=read(directory/f'qualification-rank-{n}.json')
        actual=torch.load(directory/f'forward-rank-{n}.pt',weights_only=True,map_location='cpu')
        check=compare_distributions(reference[q['reference_index']:q['reference_index']+1],actual)[0]
        assert check['passed'] and q['status']=='PASS'
        assert all(isinstance(v,bool) and v==q['native_check'][k]
                   or not isinstance(v,bool) and abs(v-q['native_check'][k])<=1e-10 for k,v in check.items())
        witness=torch.load(directory/f'initial-master-rank-{n}.pt',weights_only=True,map_location='cpu')
        assert torch.equal(witness['expected_cpu'].view(torch.uint8),witness['actual_fp32_masters'].view(torch.uint8))
        native.append(check);initial.append(dict(directory=str(directory),rank=rank,exact=True))
model,_=load_base(p['model_dir'])
checkpoints={}
with torch.no_grad():
    for step in (4,8):
        checkpoints[str(step)]=h.audit_checkpoint(O/f'step-{step:07d}',p,
            'scaffold_qualification',O.name,order,sha256_file(C/'protocol.json'),metrics[step-1]['counters'],model)
        print('FULL_CHECKPOINT_PASS',step,flush=True)
restored=[]
for rank in range(8):
    saved=torch.load(O/'step-0000004'/f'rank-{rank:05d}.pt',map_location='cpu',weights_only=True,mmap=True)
    report=read(O/'recoveries/resume8'/f'restored-rank-{rank:05d}.json')
    assert report['full_active_model_bytes_equal'] and report['full_optimizer_bytes_equal'] and report['tied_parameter_objects_preserved']
    assert optimizer_digest(saved['optimizer'])==report['optimizer_digest']
    restored.append(dict(rank=rank,all_actual_optimizer_digests_match_saved_state4=True))
result=dict(status='PASS_ALL8_ORDERS_COMPONENTS_NATIVE16_FULL_STATES4_8_AND_ACTUAL_RESTORE8',utc=utc_now(),
    numerical_source_files=p['source_files'],audit_source_sha256=sha256_file(__file__),
    reused_generic_state_audit_sha256=sha256_file(helper),protocol_sha256=sha256_file(C/'protocol.json'),
    checkpoints=checkpoints,actual_native_vectors_recomputed=len(native),native_checks=native,
    initial_released_FP32_master_witnesses=len(initial),actual_restored_optimizer_checks=restored,
    actual_updates=8,source_exposures=256,supervised_positions=total,metrics_sha256=sha256_file(O/'metrics.jsonl'),
    all4960_FP32_master_shards_and_moments_finite_and_cast_consistent=True,
    limits=['CPU full-state/integrity audit and actual device restoration,not learning/generalization evidence.',
            'Actual RNG reset uses the existing qualified helper;no new independent RNG readback or uninterrupted-trajectory parity claim.',
            'Eight updates do not establish multi-hour memory stability.','Terminal EBS protection is checked separately.'])
atomic_json(C/'independent-state-audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('numerical_source_files','native_checks')},indent=2))
