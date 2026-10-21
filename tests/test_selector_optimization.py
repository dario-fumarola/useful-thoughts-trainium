import math

import numpy as np
import pytest
import torch

from train.causal_reader_gate import SelectiveHead
from train.selector_optimization import (SmoothSelectiveHead,INITIAL_LOGIT,
    split_sources,fixed_orders,make_head,numerical_fit,reject_completed_DEV_reads)


def test_smooth_and_hard_match_initial_value_and_learning_sensitivity():
    derivatives=[]
    for head in (SelectiveHead(1),SmoothSelectiveHead(1)):
        u=torch.tensor([INITIAL_LOGIT],dtype=torch.float64,requires_grad=True)
        g=head.gate(u);g.sum().backward()
        assert g.item()==pytest.approx(.01,abs=1e-15)
        derivatives.append(u.grad.item())
    assert derivatives[0]==pytest.approx(.019701,abs=1e-15)
    assert derivatives[1]==pytest.approx(derivatives[0],abs=1e-15)


def test_helpful_example_can_reopen_smooth_gate_below_old_dead_region():
    x=torch.ones(1,16,1,dtype=torch.float64)
    raw=torch.full((1,16),math.log(.1),dtype=torch.float64)
    assisted=torch.full_like(raw,math.log(.9))
    hard=SelectiveHead(1);smooth=SmoothSelectiveHead(1)
    with torch.no_grad():hard.bias.fill_(-6);smooth.bias.fill_(-6)
    hard.loss(x,raw,assisted).backward()
    assert hard.bias.grad==0 and hard.weight.grad==0
    before=smooth.loss(x,raw,assisted);before.backward()
    assert smooth.bias.grad<0 and smooth.weight.grad<0
    with torch.no_grad():
        smooth.bias-=smooth.bias.grad;smooth.weight-=smooth.weight.grad
    assert smooth.loss(x,raw,assisted)<before


def test_harmful_example_reduces_smooth_assistance_and_never_updates_inputs():
    x=torch.ones(1,16,1,dtype=torch.float64,requires_grad=True)
    raw=torch.full((1,16),math.log(.9),dtype=torch.float64,requires_grad=True)
    assisted=torch.full_like(raw,math.log(.1),requires_grad=True)
    head=SmoothSelectiveHead(1);head.loss(x,raw,assisted).backward()
    assert head.bias.grad>0
    assert x.grad is None and raw.grad is None and assisted.grad is None


def test_disabled_smooth_system_is_exact_base_without_an_assisted_prediction():
    head=SmoothSelectiveHead(1)
    raw=torch.randn(2,16,5,dtype=torch.float64).log_softmax(-1)
    assert head.predict_logps(None,raw,enabled=False) is raw


def test_smooth_full_vector_normalization_target_consistency_and_extremes():
    torch.manual_seed(18);head=SmoothSelectiveHead(1)
    raw=torch.randn(2,16,11,dtype=torch.float64).log_softmax(-1)
    other=torch.randn_like(raw).log_softmax(-1)
    u=torch.linspace(-30,30,32,dtype=torch.float64).view(2,16).requires_grad_()
    full=head.predict_logps(u,raw,other)
    assert torch.allclose(full.exp().sum(-1),torch.ones(2,16,dtype=torch.float64),atol=1e-12)
    assert torch.allclose(full[:,:,3],head.target_logps(u,raw[:,:,3],other[:,:,3]),atol=1e-12)
    full.sum().backward();assert torch.isfinite(u.grad).all()
    extreme=torch.tensor([-1000.,1000.],dtype=torch.float64,requires_grad=True)
    lp=head.target_logps(extreme,torch.tensor([-1.,-1000.]),torch.tensor([-1000.,-1.]))
    lp.sum().backward();assert torch.isfinite(lp).all() and torch.isfinite(extreme.grad).all()


def labels():
    return [dict(document_id=f'doc-{i:05}',kind='calculation' if i<2560 else 'ordinary') for i in range(5120)]


def test_source_only_split_is_balanced_disjoint_and_probe_is_fit_only():
    rows=labels();split=split_sources(rows)
    assert len(split['fit'])==4096 and len(split['hold'])==1024 and len(split['probe'])==256
    assert not set(split['fit'])&set(split['hold'])
    assert set(split['fit'])|set(split['hold'])==set(range(5120))
    assert set(split['probe'])<=set(split['fit'])
    for name,n in [('fit',2048),('hold',512),('probe',128)]:
        assert sum(rows[i]['kind']=='calculation' for i in split[name])==n
    changed=[dict(r,answer='changed future',target_logp=-999) for r in rows]
    assert split_sources(changed)==split
    assert split_sources(rows)==split
    rows[-1]['document_id']=rows[0]['document_id']
    with pytest.raises(ValueError):split_sources(rows)


def test_fixed_orders_cover_all_fit_sources_without_selecting_seeds():
    a=fixed_orders(4096,2026102141)
    assert np.array_equal(a,fixed_orders(4096,2026102141))
    assert not np.array_equal(a,fixed_orders(4096,2026102142))
    assert np.bincount(a).tolist()==[10]*4096
    for seed in (2026102131,2026102147):
        with pytest.raises(ValueError):fixed_orders(4096,seed)
    with pytest.raises(ValueError):fixed_orders(5120,2026102141)
    with pytest.raises(ValueError):make_head('extra_recipe',1)


def test_fitter_rejects_a_full_train_or_validation_tensor_before_any_update():
    head=SmoothSelectiveHead(1);features=torch.zeros(1024,16,1,dtype=torch.float64)
    lp=torch.full((1024,16),-1.,dtype=torch.float64)
    with pytest.raises(ValueError):
        numerical_fit(head,features,lp,lp,np.arange(40960),lambda:None,lambda _:None,lambda *_:None)


def test_completed_DEV_cache_and_label_paths_are_denied():
    for path in ('/tmp/features/DEV_calculation/T_note/hidden.npy',
                 '/tmp/features/DEV_ordinary/T_raw/target_logps.npy',
                 '/tmp/data/DEV_calculation.jsonl'):
        with pytest.raises(PermissionError):reject_completed_DEV_reads('open',(path,))
    reject_completed_DEV_reads('open',('/tmp/features/TRAIN/T_note/hidden.npy',))
    reject_completed_DEV_reads('open',(5,))
