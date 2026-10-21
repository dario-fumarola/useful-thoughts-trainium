from types import SimpleNamespace
import json

import pytest

from train.causal_reader_gate_driver import protocol


@pytest.mark.parametrize('command',('generate','features','reference','causality','interventions'))
def test_cpu_preparation_cannot_release_an_actual_model_worker(tmp_path,command):
    path=tmp_path/'preparation.json'
    path.write_text(json.dumps(dict(status='CPU_FEASIBILITY_PLAN_NOT_MODEL_OR_LEARNING_RELEASE')))
    # No loader or asset path is even reachable from an unreleased CPU plan.
    args=SimpleNamespace(protocol=path,command=command,phase='qualification',cohort='qualification')
    with pytest.raises(ValueError,match='Exact frozen reader protocol'):
        protocol(args)


def test_cpu_preparation_cannot_release_head_fitting_or_dev_readout(tmp_path):
    from train.causal_reader_head_driver import require_release
    path=tmp_path/'preparation.json'
    path.write_text(json.dumps(dict(status='CPU_FEASIBILITY_PLAN_NOT_MODEL_OR_LEARNING_RELEASE')))
    with pytest.raises(ValueError,match='Exact frozen head protocol'):
        require_release(path)


def test_greedy_reference_distinguishes_numerical_ties_from_real_policy_errors():
    import torch
    from train.causal_reader_gate_driver import greedy_margin
    tied=torch.tensor([[1.,1.0001,-2.]])
    result=greedy_margin(tied,[0])
    assert result['numerical_near_tie_actions']==1
    assert result['maximum_argmax_logit_regret']<.001
    assert greedy_margin(tied,[1])['maximum_argmax_logit_regret']==0
    with pytest.raises(ValueError,match='argmax numerical tolerance'):
        greedy_margin(tied,[2])
