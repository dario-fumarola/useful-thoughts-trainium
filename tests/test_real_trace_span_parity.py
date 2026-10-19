import pytest
import torch
from train.real_trace_cpu_scores import span_parity


def test_fixed_fullspan_parity_rejects_one_bad_token_even_when_means_cancel():
    reference = torch.full((20,16),-1.,dtype=torch.float64)
    actual = reference.clone()
    actual[0,0] += .002
    actual[1,0] -= .002
    report = span_parity(reference,actual)
    assert report['absolute_mean_span_drift'] == 0
    assert report['max_token_error'] > .001 and not report['passed']


def test_fixed_fullspan_parity_rejects_accumulated_span_drift():
    reference = torch.full((20,16),-1.,dtype=torch.float64)
    report = span_parity(reference,reference+.0001)
    assert report['max_token_error'] < .001
    assert report['absolute_mean_span_drift'] > .001 and not report['passed']


def test_actual_finite_paired_rows_are_required():
    reference = torch.full((20,16),-1.,dtype=torch.float64)
    actual = reference.clone()
    actual[:,0] += .0002
    assert span_parity(reference,actual)['passed']
    actual[0,0] = float('nan')
    assert not span_parity(reference,actual)['passed']
    with pytest.raises(ValueError):
        span_parity(reference,reference[:,:1])
