import math

import numpy as np
import pytest
import torch

from train.causal_reader_fit import constant_logps, fit_constant, fit_position_constants, fixed_order


def test_scalar_control_recovers_convex_optimum_rather_than_token_oracle():
    raw=torch.tensor([[math.log(.9),math.log(.1)]],dtype=torch.float64)
    other=raw.flip(-1)
    result=fit_constant(raw,other)
    assert result['weight']==pytest.approx(.5,abs=1e-8)
    assert result['TRAIN_token_NLL']==pytest.approx(math.log(2),abs=1e-12)
    assert constant_logps(raw,other,0) is raw
    assert constant_logps(raw,other,1) is other


def test_scalar_control_does_not_recover_an_unpredictable_coin_oracle():
    raw=torch.full((1,2),math.log(.5),dtype=torch.float64)
    other=torch.tensor([[math.log(.9),math.log(.1)]],dtype=torch.float64)
    result=fit_constant(raw,other)
    assert result['weight']<1e-8
    assert result['TRAIN_token_NLL']==pytest.approx(math.log(2),abs=1e-12)
    # Very large probability ratios must not overflow the derivative solver.
    extreme=fit_constant(torch.tensor([[-1e6,-.1]],dtype=torch.float64),
                         torch.tensor([[-.1,-1e6]],dtype=torch.float64))
    assert math.isfinite(extreme['TRAIN_token_NLL'])


def test_matching_orders_cover_train_and_cannot_substitute_dev_size():
    first=fixed_order(5120,2026102131)
    assert np.array_equal(first,fixed_order(5120,2026102131))
    assert not np.array_equal(first,fixed_order(5120,2026102132))
    assert np.bincount(first).tolist()==[10]*5120
    with pytest.raises(ValueError):fixed_order(4096,2026102131)


def test_position_schedule_can_explain_gain_without_contextual_selection():
    raw=torch.full((4,16),math.log(.5),dtype=torch.float64)
    assisted=torch.tensor([math.log(.9)]*8+[math.log(.1)]*8,dtype=torch.float64)[None].expand(4,-1)
    global_fit=fit_constant(raw,assisted)
    positional=fit_position_constants(raw,assisted)
    assert global_fit['weight']<1e-8
    assert positional['weights'][:8]==[1.]*8
    assert max(positional['weights'][8:])<1e-8
    assert global_fit['TRAIN_token_NLL']-positional['TRAIN_token_NLL']>.25
