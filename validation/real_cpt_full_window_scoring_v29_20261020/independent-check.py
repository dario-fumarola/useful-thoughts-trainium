"""Reconstruct complete saved scoring witnesses independently; no model calls."""
from pathlib import Path
import json
import hashlib
import math
import numpy as np
import torch

C=Path('/home/ubuntu/dario/local/diagnostics/real-cpt-ordinary-full-window-v29-v2')
read=lambda name:json.loads((C/name).read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
p=read('plan.json');r=read('worker-result.json');closure=read('closure.json')
assert r['plan_sha256']==sha(C/'plan.json')
assert all(sha(Path(path))==value for path,value in p['sources'].items())
reference=np.asarray(read('HF-reference.json')['token_logps'],dtype=np.float64)
assert reference.shape==(4,527) and np.isfinite(reference).all()
errors={};conditions={}
for size in (1,4):
    rows=read(f'full-window-B{size}.json');assert len(rows)==4
    values=np.asarray([v['token_logps'] for v in rows],dtype=np.float64)
    assert values.shape==(4,527) and np.isfinite(values).all()
    for row in rows:
        assert row['tokens']==527 and row['full_logp']==sum(row['token_logps']) and row['K1_logp']==row['token_logps'][0]
    error=float(np.max(np.abs(values-reference)));assert error<=p['max_abs_logp_error']
    assert error==r['max_abs_target_error'][str(size)]
    errors[str(size)]=error;conditions[str(size)]=values
suffix=np.asarray([v['token_logps'] for v in read('suffix16-B4.json')],dtype=np.float64)
assert suffix.shape==(4,16) and np.isfinite(suffix).all()
suffix_error=float(np.max(np.abs(suffix-reference[:,-16:])))
assert suffix_error==r['max_abs_suffix_error'] and suffix_error<=p['max_abs_logp_error']
cpu=torch.load(C/'CPU-native.pt',map_location='cpu',weights_only=True).double().numpy()
device=torch.load(C/'XLA-native.pt',map_location='cpu',weights_only=True).double().numpy()
assert cpu.shape==device.shape==(12,151936) and np.isfinite(cpu).all() and np.isfinite(device).all()
cpu-=np.logaddexp.reduce(cpu,axis=1)[:,None];device-=np.logaddexp.reduce(device,axis=1)[:,None]
native=[]
for i in range(12):
    delta=device[i]-cpu[i]
    f=float(np.sum(np.exp(cpu[i])*-delta));b=float(np.sum(np.exp(device[i])*delta))
    assert min(f,b)>=-1e-10
    values=dict(reference_to_device_kl_nats=max(0.,f),device_to_reference_kl_nats=max(0.,b),
        max_bidirectional_kl_nats=max(0.,f,b),max_logp_error=float(np.max(np.abs(delta))),
        rms_logp_error=float(np.sqrt(np.mean(delta**2))),
        relative_l2_logp_error=float(np.linalg.norm(delta)/np.linalg.norm(cpu[i])))
    assert values['max_bidirectional_kl_nats']<=.01 and values['max_logp_error']<=.001
    for key,value in values.items():assert math.isclose(value,r['native_checks'][i][key],abs_tol=1e-10,rel_tol=0.)
    native.append(values)
assert r['status']=='PASS_ACTUAL_TRAIN_FULL527_AND_SUFFIX16_SCORING'
assert closure['exit_code']==0 and closure['actual_owned_device_observed'] and closure['assigned_physical_device_empty'] and closure['all_original_session_processes_absent'] and not closure['host_bounds_stop']
assert closure['reserved_chip_hours']<p['maximum_reserved_chip_hours']
result=dict(status='PASS_INDEPENDENT_NATIVE12_FULL4224_TARGETS_SUFFIX64_AND_PHYSICAL_CLOSURE',
    max_abs_target_error=errors,max_abs_suffix_error=suffix_error,native_checks=native,
    serial_batch4_max_abs_difference=float(np.max(np.abs(conditions['1']-conditions['4']))),
    source_sha256=sha(Path(__file__)),plan_sha256=sha(C/'plan.json'),
    successful_reserved_chip_hours=closure['reserved_chip_hours'],
    failed_original_reserved_chip_hours=p['original_attempt']['reserved_chip_hours'],
    total_question_reserved_chip_hours=closure['reserved_chip_hours']+p['original_attempt']['reserved_chip_hours'],
    limits='Numerical execution on four TRAIN documents under one frozen trained model; no capability, calibration or generalization result. Cold-inclusive timings do not establish warm throughput.')
with (C/'independent-check.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='native_checks'},indent=2))
