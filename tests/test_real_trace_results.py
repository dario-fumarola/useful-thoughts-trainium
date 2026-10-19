import pytest
from train.real_trace_results import contrast,decisions


def rows(value):
    return [dict(item_id=str(i),document_id=str(i),logps16=[value]*16) for i in range(256)]


def test_likelihood_and_regression_have_opposite_signs():
    # A constant +0.01 LL/token gives +0.16 LL/span and -0.01 NLL/token.
    a,b=rows(-1.),rows(-.99)
    gain=contrast(a,b);regression=contrast(a,b,token_nll=True)
    assert gain['difference_b_minus_a']==pytest.approx(.16)
    assert regression['difference_b_minus_a']==pytest.approx(-.01)
    assert gain['documents']==256 and gain['bootstrap_draws']==20000
    with pytest.raises(ValueError):contrast(a,b[:-1])


def test_note_execution_failure_cannot_erase_raw_training_benefit():
    good=dict(difference_b_minus_a=.4,ci95=[.1,.7]);stable=dict(difference_b_minus_a=.01,ci95=[-.01,.03])
    result=decisions(good,stable,good,good,good,inference_execution_complete=False)
    assert result['training_benefit_pass'] and not result['inference_benefit_pass']
    bad_training=dict(difference_b_minus_a=-.1,ci95=[-.3,.1])
    result=decisions(bad_training,stable,good,good,good,inference_execution_complete=True)
    assert not result['training_benefit_pass'] and result['inference_benefit_pass']


def test_positive_point_estimate_alone_cannot_pass():
    uncertain=dict(difference_b_minus_a=.4,ci95=[0.,.7]);good=dict(difference_b_minus_a=.4,ci95=[.1,.7])
    stable=dict(difference_b_minus_a=.01,ci95=[-.01,.03])
    result=decisions(uncertain,stable,good,good,uncertain,inference_execution_complete=True)
    assert not result['training_benefit_pass'] and not result['inference_benefit_pass']
