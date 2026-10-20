"""Independent NumPy reconstruction of actual fixed-TRAIN readout evidence."""
from pathlib import Path
import json
import numpy as np
import torch
from weekend_thinking.artifacts import atomic_json,sha256_file

R=Path('/home/ubuntu/dario');C=R/'local/diagnostics/real-cpt-readout-v29-qualification-v1'
read=lambda p:json.loads(Path(p).read_text())
cpu=torch.load(C/'reference/CPU-native.pt',map_location='cpu',weights_only=True)
inputs=read(C/'reference/TRAIN-inputs.json');checks={};all_files={}

def normalized(value):
    x=value.numpy().astype(np.float64);maximum=x.max(axis=1,keepdims=True)
    return x-maximum-np.log(np.exp(x-maximum).sum(axis=1,keepdims=True))

def compare(a,b,path):
    assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    x,y=normalized(a),normalized(b)
    maximum=np.abs(x-y).max(axis=1)
    kl=np.maximum(np.maximum((np.exp(x)*(x-y)).sum(axis=1),(np.exp(y)*(y-x)).sum(axis=1)),0.)
    assert np.all(maximum<=.001) and np.all(kl<=.01)
    original=read(path);original=original if isinstance(original,list) else [original]
    assert len(original)==len(maximum)
    assert max(abs(float(m)-v['max_logp_error']) for m,v in zip(maximum,original))<1e-9
    assert max(abs(float(m)-v['max_bidirectional_kl_nats']) for m,v in zip(kl,original))<1e-9
    return dict(vectors=len(maximum),max_logp_error=float(maximum.max()),max_bidirectional_KL=float(kl.max()))

complete=read(C/'complete.json')
for panel in ('writer','math','ordinary','scalar'):
    p=C/panel;run=read(p/'run.json');closure=read(C/f'{panel}-closure.json')
    assert run['status']=='PASS_FIXED_TRAIN_NATIVE_AND_PANEL_MECHANICS' and run['rows']==0 and run['qualify_only']
    assert closure['exit_code']==0 and closure['owned_session_absent'] and closure['device_idle'] and closure['observed_actual_single_process']
    actual=torch.load(p/'native-logps.pt',map_location='cpu',weights_only=True)
    checks[panel]=compare(cpu[panel],actual,p/'native-check.json')
    if panel in ('writer','scalar'):
        vectors=torch.load(p/'cache-prefill-logps.pt',map_location='cpu',weights_only=True)
        checks[panel+'_cache']=compare(vectors['ordinary'],vectors['cached'],p/'cache-prefill.json')
    else:
        score=read(p/'TRAIN-full-span.json');assert score['passed'] and len(score['rows'])==4
        reference=np.asarray([v['HF_target_logps'] for v in inputs[panel]])
        observed=np.asarray([v['token_logps'] for v in score['rows']])
        assert reference.shape==observed.shape==(4,527 if panel=='ordinary' else 16)
        maximum=float(np.abs(reference-observed).max());assert maximum<=.001
        assert abs(maximum-score['max_token_error'])<1e-10
        checks[panel+'_targets']=dict(native_targets=int(reference.size),max_logp_error=maximum)
for size in (4,8):
    actual=torch.load(C/f'writer/cached-B{size}-native.pt',map_location='cpu',weights_only=True)
    checks[f'writer_cached_B{size}']=compare(cpu['writer'][:size],actual,C/f'writer/cached-B{size}-native.json')
for p in C.rglob('*'):
    if p.is_file() and p.name not in ('independent-check-v2.json','independent-check-v2.log','independent-check.log'):all_files[str(p)]=sha256_file(p)
atomic_json(C/'independent-check-v2.json',dict(status='PASS_INDEPENDENT_TRAIN_NATIVE_CACHE_AND_ALL_TARGET_RECONSTRUCTION',
    checks=checks,reserved_chip_hours=complete['reserved_chip_hours'],source_sha256=sha256_file(__file__),
    evidence_files=all_files,limits='Independent arithmetic on the original stored witnesses, not external replication or scientific learning evidence.'))
print('PASS_ALL_TRAIN_NATIVE_CACHE_AND_TARGET_VECTORS',flush=True)
